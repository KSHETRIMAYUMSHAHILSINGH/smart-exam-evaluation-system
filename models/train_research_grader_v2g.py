"""Research Grader V2G — strict evidence verification.

V2G fixes the three failure modes found in V2F:

1. FALSE CLEAN GATE
   A valid paraphrased Q4 answer triggered because evidence_word_ratio was low.
   V2G requires:
       low evidence ratio
       AND
       an explicit gaming signal
   before padding defense activates.

2. FAKE EVIDENCE SURVIVED V2F
   V2E/V2F allowed a non-zero NLI floor even when entailment was poor.
   V2G removes that floor. Suspicious sentences count as evidence only when
   they strongly entail an atomic rubric criterion.

3. Q3 CONTRADICTIONS WERE MISSED
   V2G combines:
       forward NLI contradiction
       reverse NLI contradiction
       explicit meta-negation detection
   for high-confidence contradiction handling.

Architecture
------------
Normal response:
    V2B -> final

Suspicious response:
    V2B
      -> strict rubric evidence verification
      -> evidence-only V2B ceiling
      -> strict coverage ceiling
      -> contradiction penalty
      -> final

The guard can NEVER raise V2B.

Inputs
------
data/training_dataset_v2c.csv
data/rubric_concepts_v2c.json
models/train_research_grader_v2b.py
models/train_research_grader_v2c.py
models/train_research_grader_v2e.py

Outputs
-------
models/artifacts/research_grader_v2g.joblib
data/research_grader_v2g_human_oof.csv
data/research_grader_v2g_weak_oof.csv
data/research_grader_v2g_adversarial.csv
data/research_grader_v2g_summary.csv

Previous graders are not modified.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import train_research_grader_v2b as v2b  # noqa: E402
import train_research_grader_v2c as v2c  # noqa: E402
import train_research_grader_v2e as v2e  # noqa: E402


DATA_PATH = ROOT / "data" / "training_dataset_v2c.csv"
RUBRIC_PATH = ROOT / "data" / "rubric_concepts_v2c.json"

ARTIFACT_PATH = ROOT / "models" / "artifacts" / "research_grader_v2g.joblib"
HUMAN_PATH = ROOT / "data" / "research_grader_v2g_human_oof.csv"
WEAK_PATH = ROOT / "data" / "research_grader_v2g_weak_oof.csv"
ADV_PATH = ROOT / "data" / "research_grader_v2g_adversarial.csv"
SUMMARY_PATH = ROOT / "data" / "research_grader_v2g_summary.csv"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

LENIENT_EVIDENCE_RATIO_GATE = 0.70

# Padding gate now needs BOTH low evidence ratio and an explicit gaming signal.
MIN_SENTENCES_FOR_VERBOSE_SIGNAL = 4
UNSUPPORTED_RATIO_FOR_VERBOSE = 0.30
QUESTION_COPY_THRESHOLD = 0.25
KEYWORD_STUFFING_THRESHOLD = 0.10
BIGRAM_REPETITION_THRESHOLD = 0.10

# ---------------------------------------------------------------------------
# Strict evidence verifier
# ---------------------------------------------------------------------------

STRICT_COSINE_MIN = 0.32
STRICT_ENTAILMENT_MIN = 0.62
STRICT_CONTRADICTION_MAX = 0.35

# Evidence ceiling around the V2B score of strictly verified evidence.
STRICT_EVIDENCE_SLACK = 0.25

# Strict concept coverage backstop.
STRICT_COVERAGE_SLACK = 0.75
MIN_STRICT_CAP = 1.0

# Near duplicate evidence text is collapsed.
EVIDENCE_DEDUP_COSINE = 0.94

# ---------------------------------------------------------------------------
# Contradiction
# ---------------------------------------------------------------------------

# NLI contradiction is checked in BOTH pair directions.
NLI_CONTRADICTION_THRESHOLD = 0.85

# Explicit metalinguistic negation such as:
# "It is false that NLP processes language."
META_NEGATION_PATTERNS = [
    r"^\s*it\s+is\s+false\s+that\b",
    r"^\s*it\s+is\s+not\s+true\s+that\b",
    r"^\s*it\s+is\s+incorrect\s+that\b",
    r"^\s*it\s+is\s+wrong\s+that\b",
    r"^\s*false\s*:\s*",
]

META_NEGATION_COSINE_MIN = 0.45

PENALTY_PER_CONTRADICTED_CONCEPT = 0.75
MAX_CONTRADICTION_PENALTY = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(value: object) -> str:
    return v2c.clean_text(value)


def metric_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred))
        if len(np.unique(y_true)) >= 2
        else float("nan"),
        "within_1_mark": float(np.mean(np.abs(y_true - y_pred) <= 1.0)),
        "within_2_marks": float(np.mean(np.abs(y_true - y_pred) <= 2.0)),
    }


def print_metrics(title, y_true, y_pred):
    m = metric_dict(y_true, y_pred)

    print(f"\n## {title}")
    print(f"MAE:            {m['mae']:.3f}")
    print(f"RMSE:           {m['rmse']:.3f}")
    print(f"R2:             {m['r2']:.3f}")
    print(f"Within 1 mark:  {m['within_1_mark'] * 100:.1f}%")
    print(f"Within 2 marks: {m['within_2_marks'] * 100:.1f}%")

    return m


def assign_human_folds(df):
    fold = pd.Series(-1, index=df.index, dtype=int)

    human = df[df["source"].astype(str).eq("human_group")]
    counts = human.groupby("question_id").size()

    if not (counts == 4).all():
        raise RuntimeError(
            f"Expected 4 human answers per question, found {counts.to_dict()}"
        )

    for _, group in human.groupby("question_id", sort=True):
        for fold_id, idx in enumerate(group.index):
            fold.loc[idx] = fold_id

    return fold


def meta_negation(sentence: str) -> bool:
    text = clean_text(sentence).lower()
    return any(re.search(pattern, text) for pattern in META_NEGATION_PATTERNS)


def strip_meta_negation(sentence: str) -> str:
    text = clean_text(sentence)

    for pattern in META_NEGATION_PATTERNS:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

    return text


# ---------------------------------------------------------------------------
# Strict evidence
# ---------------------------------------------------------------------------

@dataclass
class StrictProfile:
    evidence_text: str
    supported_count: int
    coverage_index: float
    contradicted_count: int
    max_contradiction: float
    selected_sentence_count: int
    original_sentence_count: int


def strict_profile(
    row: pd.Series,
    rubric_concepts,
    embedder,
    nli,
) -> StrictProfile:
    qid = str(row["question_id"])

    criteria = [
        clean_text(x)
        for x in rubric_concepts[qid]["criteria"]
        if clean_text(x)
    ]

    sentences = [
        clean_text(s)
        for s in v2c.split_sentences(
            clean_text(row["student_answer"])
        )
        if clean_text(s)
    ]

    if not sentences:
        sentences = [""]

    criterion_vecs = embedder.encode(criteria)
    sentence_vecs = embedder.encode(sentences)

    cosine = criterion_vecs @ sentence_vecs.T

    # Forward NLI: student sentence -> rubric criterion
    forward_pairs = [
        (sentence, criterion)
        for criterion in criteria
        for sentence in sentences
    ]

    forward_probs = nli.probabilities(forward_pairs).reshape(
        len(criteria),
        len(sentences),
        3,
    )

    # Reverse NLI is used ONLY for contradiction robustness.
    reverse_pairs = [
        (criterion, sentence)
        for criterion in criteria
        for sentence in sentences
    ]

    reverse_probs = nli.probabilities(reverse_pairs).reshape(
        len(criteria),
        len(sentences),
        3,
    )

    entail = forward_probs[:, :, v2c.ENTAILMENT_INDEX]

    contradiction = np.maximum(
        forward_probs[:, :, v2c.CONTRADICTION_INDEX],
        reverse_probs[:, :, v2c.CONTRADICTION_INDEX],
    )

    support_matrix = np.zeros_like(entail, dtype=np.float32)
    contradicted_concepts = np.zeros(len(criteria), dtype=bool)

    # High-confidence NLI contradiction.
    contradicted_concepts |= (
        contradiction.max(axis=1)
        >= NLI_CONTRADICTION_THRESHOLD
    )

    # Explicit meta-negation fallback.
    for si, sentence in enumerate(sentences):
        if not meta_negation(sentence):
            continue

        stripped = strip_meta_negation(sentence)

        if stripped:
            stripped_vec = embedder.encode([stripped])[0]
            stripped_cosine = criterion_vecs @ stripped_vec

            contradicted_concepts |= (
                stripped_cosine >= META_NEGATION_COSINE_MIN
            )

    # Strict evidence:
    # no entailment floor; the student sentence must actually entail the
    # criterion and must not strongly contradict it.
    for ci in range(len(criteria)):
        for si in range(len(sentences)):
            if meta_negation(sentences[si]):
                continue

            if cosine[ci, si] < STRICT_COSINE_MIN:
                continue

            if entail[ci, si] < STRICT_ENTAILMENT_MIN:
                continue

            if contradiction[ci, si] > STRICT_CONTRADICTION_MAX:
                continue

            support_matrix[ci, si] = float(
                entail[ci, si]
                * max(0.0, cosine[ci, si])
            )

    best_support = support_matrix.max(axis=1)

    supported = best_support > 0.0
    supported_count = int(supported.sum())

    # Select only sentences that verified at least one rubric concept.
    sentence_verified = support_matrix.max(axis=0) > 0.0

    selected_sentences = [
        sentence
        for sentence, keep in zip(sentences, sentence_verified)
        if keep
    ]

    selected_sentences = v2e.deduplicate_evidence_sentences(
        selected_sentences,
        embedder,
    )

    evidence_text = clean_text(" ".join(selected_sentences))

    hard_coverage = supported_count / max(1, len(criteria))

    mean_support = float(best_support.mean())

    # Hard concept count dominates.
    coverage_index = float(
        0.80 * hard_coverage
        + 0.20 * np.clip(mean_support, 0.0, 1.0)
    )

    return StrictProfile(
        evidence_text=evidence_text,
        supported_count=supported_count,
        coverage_index=coverage_index,
        contradicted_count=int(contradicted_concepts.sum()),
        max_contradiction=float(contradiction.max()),
        selected_sentence_count=len(selected_sentences),
        original_sentence_count=len(sentences),
    )


# ---------------------------------------------------------------------------
# Gaming gate
# ---------------------------------------------------------------------------

def v2b_feature_dict(row, embedder):
    return v2b.make_features(
        question_text=row["question_text"],
        model_answer=row["model_answer"],
        student_answer=row["student_answer"],
        rubric_keywords=row["rubric_keywords"],
        embedder=embedder,
    )


def gaming_gate(
    row,
    lenient_profile,
    base_features,
):
    sentence_count = len(
        v2b.split_sentences(
            clean_text(row["student_answer"])
        )
    )

    unsupported = float(
        base_features.get("unsupported_word_ratio", 0.0)
    )
    question_copy = float(
        base_features.get("question_copy_ratio", 0.0)
    )
    stuffing = float(
        base_features.get("keyword_stuffing_score", 0.0)
    )
    repeated = float(
        base_features.get("repeated_bigram_ratio", 0.0)
    )

    low_evidence = (
        lenient_profile.evidence_word_ratio
        < LENIENT_EVIDENCE_RATIO_GATE
    )

    verbose_signal = (
        sentence_count >= MIN_SENTENCES_FOR_VERBOSE_SIGNAL
        and unsupported >= UNSUPPORTED_RATIO_FOR_VERBOSE
    )

    question_copy_signal = (
        question_copy >= QUESTION_COPY_THRESHOLD
    )

    stuffing_signal = (
        stuffing >= KEYWORD_STUFFING_THRESHOLD
    )

    repetition_signal = (
        repeated >= BIGRAM_REPETITION_THRESHOLD
    )

    explicit_signal = any(
        [
            verbose_signal,
            question_copy_signal,
            stuffing_signal,
            repetition_signal,
        ]
    )

    active = low_evidence and explicit_signal

    return {
        "active": bool(active),
        "low_evidence": bool(low_evidence),
        "verbose_signal": bool(verbose_signal),
        "question_copy_signal": bool(question_copy_signal),
        "stuffing_signal": bool(stuffing_signal),
        "repetition_signal": bool(repetition_signal),
        "sentence_count": int(sentence_count),
        "unsupported_word_ratio": unsupported,
        "question_copy_ratio": question_copy,
        "keyword_stuffing_score": stuffing,
        "repeated_bigram_ratio": repeated,
    }


# ---------------------------------------------------------------------------
# Strict coverage calibration
# ---------------------------------------------------------------------------

def fit_strict_coverage_guard(
    train_df,
    strict_profiles,
):
    x = np.asarray(
        [
            strict_profiles[int(row_id)].coverage_index
            for row_id in train_df["_row_id"]
        ],
        dtype=float,
    )

    y = train_df["human_score"].to_numpy(dtype=float)

    weights = v2b.sample_weights(
        train_df,
        adversarial=False,
    )

    model = IsotonicRegression(
        y_min=0.0,
        y_max=10.0,
        increasing=True,
        out_of_bounds="clip",
    )

    model.fit(
        x,
        y,
        sample_weight=weights,
    )

    return model


def strict_coverage_ceiling(
    guard,
    coverage_index,
):
    expected = float(
        guard.predict(
            np.asarray([coverage_index], dtype=float)
        )[0]
    )

    return float(
        np.clip(
            max(
                MIN_STRICT_CAP,
                expected + STRICT_COVERAGE_SLACK,
            ),
            0.0,
            10.0,
        )
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def v2g_predict(
    base_model,
    strict_guard,
    row,
    rubric_concepts,
    embedder,
    nli,
):
    base_features = v2b_feature_dict(
        row,
        embedder,
    )

    base_X = pd.DataFrame(
        [base_features],
        columns=v2b.FEATURE_NAMES,
    )

    base_score = float(
        np.clip(
            base_model.predict(base_X)[0],
            0.0,
            10.0,
        )
    )

    # Lenient profile is ONLY used for suspicious-padding detection.
    lenient = v2e.extract_evidence_profile(
        row=row,
        rubric_concepts=rubric_concepts,
        embedder=embedder,
        nli=nli,
    )

    gate = gaming_gate(
        row=row,
        lenient_profile=lenient,
        base_features=base_features,
    )

    strict = strict_profile(
        row=row,
        rubric_concepts=rubric_concepts,
        embedder=embedder,
        nli=nli,
    )

    contradiction_gate = strict.contradicted_count > 0
    gate_active = gate["active"] or contradiction_gate

    evidence_score = base_score
    evidence_ceiling = 10.0
    coverage_ceiling = 10.0

    if gate_active:
        if strict.evidence_text:
            evidence_row = row.copy()
            evidence_row["student_answer"] = strict.evidence_text

            evidence_score = v2e.v2b_predict_one(
                model=base_model,
                row=evidence_row,
                embedder=embedder,
            )
        else:
            evidence_score = 0.0

        evidence_ceiling = float(
            np.clip(
                evidence_score + STRICT_EVIDENCE_SLACK,
                0.0,
                10.0,
            )
        )

        coverage_ceiling = strict_coverage_ceiling(
            strict_guard,
            strict.coverage_index,
        )

        guarded = min(
            base_score,
            evidence_ceiling,
            coverage_ceiling,
        )
    else:
        guarded = base_score

    contradiction_penalty = min(
        MAX_CONTRADICTION_PENALTY,
        strict.contradicted_count
        * PENALTY_PER_CONTRADICTED_CONCEPT,
    )

    final_score = float(
        np.clip(
            guarded - contradiction_penalty,
            0.0,
            10.0,
        )
    )

    return {
        "base_score": base_score,
        "final_score": final_score,
        "gate_active": bool(gate_active),
        "padding_gate": bool(gate["active"]),
        "contradiction_gate": bool(contradiction_gate),
        "evidence_word_ratio": float(lenient.evidence_word_ratio),
        "strict_supported_count": int(strict.supported_count),
        "strict_coverage_index": float(strict.coverage_index),
        "strict_evidence_text": strict.evidence_text,
        "strict_evidence_score": float(evidence_score),
        "strict_evidence_ceiling": float(evidence_ceiling),
        "strict_coverage_ceiling": float(coverage_ceiling),
        "contradicted_concepts": int(strict.contradicted_count),
        "max_contradiction": float(strict.max_contradiction),
        "contradiction_penalty": float(contradiction_penalty),
        **gate,
    }


# ---------------------------------------------------------------------------
# V2B training helpers
# ---------------------------------------------------------------------------

def build_v2b_feature_frame(df, embedder, label):
    return v2e.build_v2b_feature_frame(
        df=df,
        embedder=embedder,
        label=label,
    )


def train_v2b_fold(
    clean_df,
    clean_X,
    attack_df,
    attack_X,
    heldout_ids,
):
    return v2e.train_v2b_fold(
        clean_df=clean_df,
        clean_X=clean_X,
        attack_df=attack_df,
        attack_X=attack_X,
        heldout_ids=heldout_ids,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    name,
    folds,
    df,
    clean_X,
    attack_df,
    attack_X,
    strict_profiles,
    rubric_concepts,
    embedder,
    nli,
):
    prediction_records = []
    attack_records = []

    for fold_no, (heldout_ids, test_df) in enumerate(folds, start=1):
        base_model, train_clean = train_v2b_fold(
            clean_df=df,
            clean_X=clean_X,
            attack_df=attack_df,
            attack_X=attack_X,
            heldout_ids=heldout_ids,
        )

        strict_guard = fit_strict_coverage_guard(
            train_df=train_clean,
            strict_profiles=strict_profiles,
        )

        for _, row in test_df.iterrows():
            result = v2g_predict(
                base_model=base_model,
                strict_guard=strict_guard,
                row=row,
                rubric_concepts=rubric_concepts,
                embedder=embedder,
                nli=nli,
            )

            truth = float(row["human_score"])

            for model_name, pred in [
                ("V2B", result["base_score"]),
                ("V2G", result["final_score"]),
            ]:
                prediction_records.append({
                    "evaluation": name,
                    "fold": fold_no,
                    "model": model_name,
                    "_row_id": int(row["_row_id"]),
                    "question_id": row["question_id"],
                    "source": row["source"],
                    "human_score": truth,
                    "prediction": float(pred),
                    "absolute_error": abs(truth - float(pred)),
                    "gate_active": result["gate_active"],
                    "padding_gate": result["padding_gate"],
                    "contradiction_gate": result["contradiction_gate"],
                    "evidence_word_ratio": result["evidence_word_ratio"],
                    "strict_supported_count": result["strict_supported_count"],
                    "contradicted_concepts": result["contradicted_concepts"],
                    "contradiction_penalty": result["contradiction_penalty"],
                    "strict_evidence_text": result["strict_evidence_text"],
                    "student_answer": row["student_answer"],
                })

            attacks = v2e.all_eval_attacks(
                row=row,
                df=df,
                rubric_concepts=rubric_concepts,
            )

            base_v2b = result["base_score"]
            base_v2g = result["final_score"]

            for attack_name, attacked_answer in attacks.items():
                attacked_row = row.copy()
                attacked_row["student_answer"] = attacked_answer

                attacked = v2g_predict(
                    base_model=base_model,
                    strict_guard=strict_guard,
                    row=attacked_row,
                    rubric_concepts=rubric_concepts,
                    embedder=embedder,
                    nli=nli,
                )

                for model_name, clean_pred, attacked_pred in [
                    ("V2B", base_v2b, attacked["base_score"]),
                    ("V2G", base_v2g, attacked["final_score"]),
                ]:
                    attack_records.append({
                        "evaluation": name,
                        "fold": fold_no,
                        "model": model_name,
                        "_row_id": int(row["_row_id"]),
                        "question_id": row["question_id"],
                        "human_score": truth,
                        "attack": attack_name,
                        "base_prediction": float(clean_pred),
                        "attacked_prediction": float(attacked_pred),
                        "score_delta": float(
                            attacked_pred - clean_pred
                        ),
                        "gate_active": attacked["gate_active"],
                        "padding_gate": attacked["padding_gate"],
                        "contradiction_gate": attacked["contradiction_gate"],
                        "evidence_word_ratio": attacked["evidence_word_ratio"],
                        "strict_supported_count":
                            attacked["strict_supported_count"],
                        "contradicted_concepts":
                            attacked["contradicted_concepts"],
                        "contradiction_penalty":
                            attacked["contradiction_penalty"],
                        "strict_evidence_text":
                            attacked["strict_evidence_text"],
                    })

        print(
            f"{name}: fold {fold_no}/{len(folds)} complete "
            f"(held out={len(test_df)})"
        )

    return (
        pd.DataFrame(prediction_records),
        pd.DataFrame(attack_records),
    )


def adversarial_report(title, adv_df, model_name):
    part = adv_df[adv_df["model"].eq(model_name)].copy()

    padding = part[
        ~part["attack"].eq("contradiction_injection")
    ].copy()

    contradiction = part[
        part["attack"].eq("contradiction_injection")
    ].copy()

    positive = padding["score_delta"].clip(lower=0.0)

    print(f"\n## {title} — {model_name}")
    print(
        f"Padding mean delta:          "
        f"{padding['score_delta'].mean():+.3f}"
    )
    print(
        f"Padding mean positive delta: "
        f"{positive.mean():+.3f}"
    )
    print(
        f"Padding worst increase:      "
        f"{padding['score_delta'].max():+.3f}"
    )
    print(
        f"Failures > +0.5 mark:        "
        f"{(padding['score_delta'] > 0.5).mean() * 100:.1f}%"
    )
    print(
        f"Failures > +1.0 mark:        "
        f"{(padding['score_delta'] > 1.0).mean() * 100:.1f}%"
    )

    print("\nBy padding attack:")
    print(
        padding.groupby("attack")["score_delta"]
        .agg(["mean", "max", "count"])
        .sort_values("mean", ascending=False)
        .round(3)
        .to_string()
    )

    print("\nContradiction injection:")
    print(
        f"Mean delta:                  "
        f"{contradiction['score_delta'].mean():+.3f}"
    )
    print(
        f"Worst increase:              "
        f"{contradiction['score_delta'].max():+.3f}"
    )
    print(
        f"Raised score > +0.5:         "
        f"{(contradiction['score_delta'] > 0.5).mean() * 100:.1f}%"
    )

    if model_name == "V2G":
        print(
            f"Attack gate activation:      "
            f"{part['gate_active'].mean() * 100:.1f}%"
        )

    return {
        "padding_mean_delta": float(padding["score_delta"].mean()),
        "padding_mean_positive_delta": float(positive.mean()),
        "padding_worst_increase": float(padding["score_delta"].max()),
        "padding_fail_gt_0_5": float(
            np.mean(padding["score_delta"] > 0.5)
        ),
        "padding_fail_gt_1_0": float(
            np.mean(padding["score_delta"] > 1.0)
        ),
        "contradiction_mean_delta": float(
            contradiction["score_delta"].mean()
        ),
        "contradiction_worst_increase": float(
            contradiction["score_delta"].max()
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("RESEARCH GRADER V2G — STRICT EVIDENCE VERIFICATION")
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if not RUBRIC_PATH.exists():
        raise FileNotFoundError(RUBRIC_PATH)

    df = pd.read_csv(DATA_PATH).reset_index(drop=True)

    with RUBRIC_PATH.open("r", encoding="utf-8") as f:
        rubric_concepts = json.load(f)

    df["_row_id"] = np.arange(len(df), dtype=int)
    df["human_score"] = pd.to_numeric(
        df["human_score"],
        errors="raise",
    ).astype(float)

    human = df[df["source"].astype(str).eq("human_group")]
    weak = df[df["human_score"] <= 4.0]

    print(f"Dataset rows:                {len(df)}")
    print(f"Human gold rows:             {len(human)}")
    print(f"Weak rows <=4:               {len(weak)}")
    print(
        f"Strict entailment threshold: {STRICT_ENTAILMENT_MIN:.2f}"
    )
    print(
        "Padding gate:                low evidence AND explicit gaming signal"
    )
    print(
        "Contradiction:               bidirectional NLI + meta-negation"
    )
    print(
        "Guard direction:             can lower V2B only"
    )

    print("\nLoading sentence encoder...")
    embedder = v2b.EmbeddingCache(v2b.ENCODER_NAME)

    print("\nLoading NLI validator...")
    nli = v2c.NLICache(
        model_name=v2c.NLI_MODEL_NAME,
        device=v2c.NLI_DEVICE,
        batch_size=v2c.NLI_BATCH_SIZE,
    )

    print("\nExtracting V2B clean features...")
    clean_X = build_v2b_feature_frame(
        df=df,
        embedder=embedder,
        label="V2B clean feature extraction",
    )

    print("\nExtracting strict clean rubric profiles...")
    strict_profiles = {}

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        strict_profiles[int(row["_row_id"])] = strict_profile(
            row=row,
            rubric_concepts=rubric_concepts,
            embedder=embedder,
            nli=nli,
        )

        if i == 1 or i % 25 == 0 or i == len(df):
            print(f"Strict profile extraction: {i}/{len(df)}")

    print("\nBuilding V2B invariance attacks...")
    attack_df = v2e.build_v2b_attack_frame(df)

    attack_X = build_v2b_feature_frame(
        df=attack_df,
        embedder=embedder,
        label="V2B attack feature extraction",
    )

    # Human OOF
    human_fold_series = assign_human_folds(df)
    human_folds = []

    for fold_id in range(4):
        test_df = df[human_fold_series.eq(fold_id)].copy()
        heldout = set(test_df["_row_id"].astype(int).tolist())
        human_folds.append((heldout, test_df))

    human_pred, human_adv = evaluate(
        name="HUMAN_OOF",
        folds=human_folds,
        df=df,
        clean_X=clean_X,
        attack_df=attack_df,
        attack_X=attack_X,
        strict_profiles=strict_profiles,
        rubric_concepts=rubric_concepts,
        embedder=embedder,
        nli=nli,
    )

    # Weak OOF
    weak_indices = weak.index.to_numpy()
    weak_strata = weak["question_id"].astype(str).to_numpy()

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    weak_folds = []

    for _, test_pos in skf.split(weak_indices, weak_strata):
        test_indices = weak_indices[test_pos]
        test_df = df.loc[test_indices].copy()
        heldout = set(test_df["_row_id"].astype(int).tolist())
        weak_folds.append((heldout, test_df))

    weak_pred, weak_adv = evaluate(
        name="WEAK_OOF",
        folds=weak_folds,
        df=df,
        clean_X=clean_X,
        attack_df=attack_df,
        attack_X=attack_X,
        strict_profiles=strict_profiles,
        rubric_concepts=rubric_concepts,
        embedder=embedder,
        nli=nli,
    )

    summary_records = []

    for evaluation, pred_df, adv_df in [
        ("HUMAN_OOF", human_pred, human_adv),
        ("WEAK_OOF", weak_pred, weak_adv),
    ]:
        print("\n" + "=" * 72)
        print(evaluation)
        print("=" * 72)

        for model_name in ["V2B", "V2G"]:
            part = pred_df[pred_df["model"].eq(model_name)]

            y_true = part["human_score"].to_numpy(dtype=float)
            y_pred = part["prediction"].to_numpy(dtype=float)

            m = print_metrics(
                f"{evaluation} — {model_name}",
                y_true,
                y_pred,
            )

            a = adversarial_report(
                f"{evaluation} ADVERSARIAL",
                adv_df,
                model_name,
            )

            clean_gate_rate = (
                float(part["gate_active"].mean())
                if model_name == "V2G"
                else 0.0
            )

            summary_records.append({
                "evaluation": evaluation,
                "model": model_name,
                "clean_gate_rate": clean_gate_rate,
                **m,
                **a,
            })

    print("\n## V2G clean gate activation")
    for name, pred_df in [
        ("HUMAN_OOF", human_pred),
        ("WEAK_OOF", weak_pred),
    ]:
        part = pred_df[pred_df["model"].eq("V2G")]
        print(
            f"{name}: "
            f"{part['gate_active'].mean() * 100:.1f}%"
        )

    print("\nTraining final V2G artifact...")

    final_base_model, final_clean = train_v2b_fold(
        clean_df=df,
        clean_X=clean_X,
        attack_df=attack_df,
        attack_X=attack_X,
        heldout_ids=set(),
    )

    final_strict_guard = fit_strict_coverage_guard(
        train_df=final_clean,
        strict_profiles=strict_profiles,
    )

    artifact = {
        "version": "research_grader_v2g",
        "architecture": (
            "V2B normal path + explicit-gaming gate + strict NLI rubric "
            "verification + bidirectional contradiction guard"
        ),
        "base_model": final_base_model,
        "strict_coverage_guard": final_strict_guard,
        "base_feature_names": v2b.FEATURE_NAMES,
        "embedding_model_name": v2b.ENCODER_NAME,
        "nli_model_name": v2c.NLI_MODEL_NAME,
        "rubric_concepts": rubric_concepts,
        "parameters": {
            "lenient_evidence_ratio_gate":
                LENIENT_EVIDENCE_RATIO_GATE,
            "min_sentences_for_verbose_signal":
                MIN_SENTENCES_FOR_VERBOSE_SIGNAL,
            "unsupported_ratio_for_verbose":
                UNSUPPORTED_RATIO_FOR_VERBOSE,
            "question_copy_threshold":
                QUESTION_COPY_THRESHOLD,
            "keyword_stuffing_threshold":
                KEYWORD_STUFFING_THRESHOLD,
            "bigram_repetition_threshold":
                BIGRAM_REPETITION_THRESHOLD,
            "strict_cosine_min":
                STRICT_COSINE_MIN,
            "strict_entailment_min":
                STRICT_ENTAILMENT_MIN,
            "strict_contradiction_max":
                STRICT_CONTRADICTION_MAX,
            "strict_evidence_slack":
                STRICT_EVIDENCE_SLACK,
            "strict_coverage_slack":
                STRICT_COVERAGE_SLACK,
            "nli_contradiction_threshold":
                NLI_CONTRADICTION_THRESHOLD,
            "meta_negation_cosine_min":
                META_NEGATION_COSINE_MIN,
            "penalty_per_contradicted_concept":
                PENALTY_PER_CONTRADICTED_CONCEPT,
            "max_contradiction_penalty":
                MAX_CONTRADICTION_PENALTY,
        },
        "design_properties": {
            "normal_path_is_v2b": True,
            "padding_requires_explicit_gaming_signal": True,
            "strict_evidence_requires_entailment": True,
            "nli_entailment_floor": False,
            "bidirectional_contradiction": True,
            "meta_negation_fallback": True,
            "guard_can_raise_score": False,
        },
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)

    human_pred.to_csv(HUMAN_PATH, index=False)
    weak_pred.to_csv(WEAK_PATH, index=False)

    combined_adv = pd.concat(
        [human_adv, weak_adv],
        ignore_index=True,
    )

    combined_adv.to_csv(ADV_PATH, index=False)

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\n## Side-by-side summary")
    print(
        summary_df[
            [
                "evaluation",
                "model",
                "clean_gate_rate",
                "mae",
                "rmse",
                "r2",
                "within_1_mark",
                "within_2_marks",
                "padding_mean_positive_delta",
                "padding_worst_increase",
                "padding_fail_gt_0_5",
                "padding_fail_gt_1_0",
                "contradiction_mean_delta",
                "contradiction_worst_increase",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\n## Saved")
    print(f"Artifact:    {ARTIFACT_PATH}")
    print(f"Human OOF:   {HUMAN_PATH}")
    print(f"Weak OOF:    {WEAK_PATH}")
    print(f"Adversarial: {ADV_PATH}")
    print(f"Summary:     {SUMMARY_PATH}")

    print(
        "\nPrevious graders and the original dataset were not modified."
    )


if __name__ == "__main__":
    main()
