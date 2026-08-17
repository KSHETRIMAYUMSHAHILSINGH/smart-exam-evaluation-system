"""Research Grader V2H — contrastive strict evidence guard.

Fixes remaining V2G failures:
1. Removes reverse-NLI contradiction gating, which falsely penalized a valid
   human Q1 answer.
2. Keeps explicit meta-negation fallback for "It is false that ..." attacks.
3. Requires forward-NLI contradiction plus an explicit negation cue for
   ordinary contradiction detection.
4. Adds contrastive rubric evidence filtering:
      target rubric similarity must beat non-target rubric similarity.
5. Rejects copied-question evidence and repeated keyword-list evidence.

Normal path remains V2B. The guard can only lower scores.

Inputs:
    data/training_dataset_v2c.csv
    data/rubric_concepts_v2c.json
    models/train_research_grader_v2b.py
    models/train_research_grader_v2c.py
    models/train_research_grader_v2e.py
    models/train_research_grader_v2g.py

Outputs:
    models/artifacts/research_grader_v2h.joblib
    data/research_grader_v2h_human_oof.csv
    data/research_grader_v2h_weak_oof.csv
    data/research_grader_v2h_adversarial.csv
    data/research_grader_v2h_summary.csv
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
import train_research_grader_v2g as v2g  # noqa: E402


DATA_PATH = ROOT / "data" / "training_dataset_v2c.csv"
RUBRIC_PATH = ROOT / "data" / "rubric_concepts_v2c.json"

ARTIFACT_PATH = ROOT / "models" / "artifacts" / "research_grader_v2h.joblib"
HUMAN_PATH = ROOT / "data" / "research_grader_v2h_human_oof.csv"
WEAK_PATH = ROOT / "data" / "research_grader_v2h_weak_oof.csv"
ADV_PATH = ROOT / "data" / "research_grader_v2h_adversarial.csv"
SUMMARY_PATH = ROOT / "data" / "research_grader_v2h_summary.csv"

RANDOM_STATE = 42

# Padding gate: keep V2G's selective design.
LENIENT_EVIDENCE_RATIO_GATE = 0.70
MIN_SENTENCES_FOR_VERBOSE_SIGNAL = 4
UNSUPPORTED_RATIO_FOR_VERBOSE = 0.30
QUESTION_COPY_THRESHOLD = 0.25
KEYWORD_STUFFING_THRESHOLD = 0.10
BIGRAM_REPETITION_THRESHOLD = 0.10

# Strict evidence.
STRICT_COSINE_MIN = 0.32
STRICT_ENTAILMENT_MIN = 0.62
STRICT_CONTRADICTION_MAX = 0.35

# Target rubric should beat the best non-target rubric.
MIN_TARGET_MARGIN = 0.035

# Sentence-quality rejection.
MAX_SENTENCE_BIGRAM_REPETITION = 0.12
MIN_SENTENCE_UNIQUE_RATIO = 0.55
MAX_QUESTION_LEXICAL_OVERLAP = 0.82
MIN_EVIDENCE_TOKENS = 4

STRICT_EVIDENCE_SLACK = 0.25
STRICT_COVERAGE_SLACK = 0.75
MIN_STRICT_CAP = 1.0

# Contradiction.
FORWARD_CONTRADICTION_THRESHOLD = 0.85
META_NEGATION_COSINE_MIN = 0.45
PENALTY_PER_CONTRADICTED_CONCEPT = 0.75
MAX_CONTRADICTION_PENALTY = 2.0

META_NEGATION_PATTERNS = [
    r"^\s*it\s+is\s+false\s+that\b",
    r"^\s*it\s+is\s+not\s+true\s+that\b",
    r"^\s*it\s+is\s+incorrect\s+that\b",
    r"^\s*it\s+is\s+wrong\s+that\b",
    r"^\s*false\s*:\s*",
]

NEGATION_PATTERNS = [
    r"\bnot\b",
    r"\bnever\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bdoes\s+not\b",
    r"\bdoesn't\b",
    r"\bis\s+not\b",
    r"\bisn't\b",
    r"\bare\s+not\b",
    r"\baren't\b",
    r"\bdo\s+not\b",
    r"\bdon't\b",
]


@dataclass
class StrictProfile:
    evidence_text: str
    supported_count: int
    coverage_index: float
    contradicted_count: int
    max_contradiction: float
    selected_sentence_count: int
    original_sentence_count: int


def clean_text(value):
    return v2c.clean_text(value)


def metric_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred))
        if len(np.unique(y_true)) >= 2 else float("nan"),
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


def meta_negation(sentence):
    text = clean_text(sentence).lower()
    return any(re.search(p, text) for p in META_NEGATION_PATTERNS)


def strip_meta_negation(sentence):
    text = clean_text(sentence)

    for pattern in META_NEGATION_PATTERNS:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

    return text


def has_explicit_negation(sentence):
    text = clean_text(sentence).lower()

    if meta_negation(text):
        return True

    return any(
        re.search(pattern, text)
        for pattern in NEGATION_PATTERNS
    )


def sentence_quality_ok(sentence, question):
    tokens = v2b.word_tokens(sentence)

    if len(tokens) < MIN_EVIDENCE_TOKENS:
        return False

    unique_ratio = len(set(tokens)) / max(1, len(tokens))

    if unique_ratio < MIN_SENTENCE_UNIQUE_RATIO:
        return False

    if (
        v2b.repeated_bigram_ratio(tokens)
        > MAX_SENTENCE_BIGRAM_REPETITION
    ):
        return False

    overlap = v2e.lexical_question_copy_ratio(
        sentence,
        question,
    )

    if overlap > MAX_QUESTION_LEXICAL_OVERLAP:
        return False

    return True


def strict_profile(
    row,
    rubric_concepts,
    embedder,
    nli,
):
    qid = str(row["question_id"])
    question = clean_text(row["question_text"])

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

    target_vecs = embedder.encode(criteria)
    sentence_vecs = embedder.encode(sentences)

    target_cosine = target_vecs @ sentence_vecs.T

    # All non-target rubric criteria provide contrastive negatives.
    negative_criteria = []

    for other_qid, payload in rubric_concepts.items():
        if str(other_qid) == qid:
            continue

        negative_criteria.extend(
            [
                clean_text(x)
                for x in payload["criteria"]
                if clean_text(x)
            ]
        )

    if negative_criteria:
        negative_vecs = embedder.encode(negative_criteria)
        negative_cosine = negative_vecs @ sentence_vecs.T
        best_negative_by_sentence = negative_cosine.max(axis=0)
    else:
        best_negative_by_sentence = np.zeros(
            len(sentences),
            dtype=float,
        )

    forward_pairs = [
        (sentence, criterion)
        for criterion in criteria
        for sentence in sentences
    ]

    probs = nli.probabilities(forward_pairs).reshape(
        len(criteria),
        len(sentences),
        3,
    )

    entail = probs[:, :, v2c.ENTAILMENT_INDEX]
    contradiction = probs[:, :, v2c.CONTRADICTION_INDEX]

    support_matrix = np.zeros_like(entail, dtype=np.float32)
    contradicted = np.zeros(len(criteria), dtype=bool)

    # Explicit meta-negation fallback.
    for si, sentence in enumerate(sentences):
        if not meta_negation(sentence):
            continue

        stripped = strip_meta_negation(sentence)

        if stripped:
            stripped_vec = embedder.encode([stripped])[0]
            stripped_cosine = target_vecs @ stripped_vec

            contradicted |= (
                stripped_cosine >= META_NEGATION_COSINE_MIN
            )

    # Ordinary contradiction:
    # forward NLI only + explicit negation cue.
    for si, sentence in enumerate(sentences):
        if not has_explicit_negation(sentence):
            continue

        strong = (
            contradiction[:, si]
            >= FORWARD_CONTRADICTION_THRESHOLD
        )

        semantic = (
            target_cosine[:, si]
            >= STRICT_COSINE_MIN
        )

        contradicted |= strong & semantic

    # Strict target evidence.
    for ci in range(len(criteria)):
        for si, sentence in enumerate(sentences):
            if meta_negation(sentence):
                continue

            if not sentence_quality_ok(
                sentence,
                question,
            ):
                continue

            target = float(target_cosine[ci, si])
            negative = float(best_negative_by_sentence[si])
            margin = target - negative

            if target < STRICT_COSINE_MIN:
                continue

            if margin < MIN_TARGET_MARGIN:
                continue

            if entail[ci, si] < STRICT_ENTAILMENT_MIN:
                continue

            if contradiction[ci, si] > STRICT_CONTRADICTION_MAX:
                continue

            support_matrix[ci, si] = float(
                entail[ci, si]
                * max(0.0, target)
                * np.clip(
                    0.5 + margin,
                    0.0,
                    1.0,
                )
            )

    best_support = support_matrix.max(axis=1)
    supported = best_support > 0.0
    supported_count = int(supported.sum())

    verified_sentence = support_matrix.max(axis=0) > 0.0

    selected_sentences = [
        sentence
        for sentence, keep in zip(
            sentences,
            verified_sentence,
        )
        if keep
    ]

    selected_sentences = v2e.deduplicate_evidence_sentences(
        selected_sentences,
        embedder,
    )

    evidence_text = clean_text(
        " ".join(selected_sentences)
    )

    hard_coverage = supported_count / max(1, len(criteria))
    mean_support = float(best_support.mean())

    coverage_index = float(
        0.85 * hard_coverage
        + 0.15 * np.clip(mean_support, 0.0, 1.0)
    )

    return StrictProfile(
        evidence_text=evidence_text,
        supported_count=supported_count,
        coverage_index=coverage_index,
        contradicted_count=int(contradicted.sum()),
        max_contradiction=float(
            contradiction.max()
            if contradiction.size else 0.0
        ),
        selected_sentence_count=len(selected_sentences),
        original_sentence_count=len(sentences),
    )


def v2b_feature_dict(row, embedder):
    return v2b.make_features(
        question_text=row["question_text"],
        model_answer=row["model_answer"],
        student_answer=row["student_answer"],
        rubric_keywords=row["rubric_keywords"],
        embedder=embedder,
    )


def gaming_gate(row, lenient_profile, base_features):
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

    return bool(
        low_evidence and explicit_signal
    )


def fit_strict_guard(train_df, strict_profiles):
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


def strict_coverage_ceiling(guard, coverage_index):
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


def v2h_predict(
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

    lenient = v2e.extract_evidence_profile(
        row=row,
        rubric_concepts=rubric_concepts,
        embedder=embedder,
        nli=nli,
    )

    padding_gate = gaming_gate(
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

    contradiction_gate = (
        strict.contradicted_count > 0
    )

    gate_active = (
        padding_gate or contradiction_gate
    )

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
        "padding_gate": bool(padding_gate),
        "contradiction_gate": bool(contradiction_gate),
        "evidence_word_ratio": float(
            lenient.evidence_word_ratio
        ),
        "strict_supported_count": int(
            strict.supported_count
        ),
        "strict_coverage_index": float(
            strict.coverage_index
        ),
        "contradicted_concepts": int(
            strict.contradicted_count
        ),
        "contradiction_penalty": float(
            contradiction_penalty
        ),
        "strict_evidence_text": strict.evidence_text,
        "evidence_score": float(evidence_score),
        "evidence_ceiling": float(evidence_ceiling),
        "coverage_ceiling": float(coverage_ceiling),
    }


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
    preds = []
    attacks = []

    for fold_no, (
        heldout_ids,
        test_df,
    ) in enumerate(
        folds,
        start=1,
    ):
        base_model, train_clean = v2e.train_v2b_fold(
            clean_df=df,
            clean_X=clean_X,
            attack_df=attack_df,
            attack_X=attack_X,
            heldout_ids=heldout_ids,
        )

        strict_guard = fit_strict_guard(
            train_clean,
            strict_profiles,
        )

        for _, row in test_df.iterrows():
            result = v2h_predict(
                base_model=base_model,
                strict_guard=strict_guard,
                row=row,
                rubric_concepts=rubric_concepts,
                embedder=embedder,
                nli=nli,
            )

            truth = float(row["human_score"])

            for model_name, prediction in [
                ("V2B", result["base_score"]),
                ("V2H", result["final_score"]),
            ]:
                preds.append({
                    "evaluation": name,
                    "fold": fold_no,
                    "model": model_name,
                    "_row_id": int(row["_row_id"]),
                    "question_id": row["question_id"],
                    "source": row["source"],
                    "human_score": truth,
                    "prediction": float(prediction),
                    "absolute_error": abs(
                        truth - float(prediction)
                    ),
                    "gate_active": result["gate_active"],
                    "padding_gate": result["padding_gate"],
                    "contradiction_gate":
                        result["contradiction_gate"],
                    "evidence_word_ratio":
                        result["evidence_word_ratio"],
                    "strict_supported_count":
                        result["strict_supported_count"],
                    "contradicted_concepts":
                        result["contradicted_concepts"],
                    "contradiction_penalty":
                        result["contradiction_penalty"],
                    "strict_evidence_text":
                        result["strict_evidence_text"],
                    "student_answer": row["student_answer"],
                })

            variants = v2e.all_eval_attacks(
                row=row,
                df=df,
                rubric_concepts=rubric_concepts,
            )

            base_v2b = result["base_score"]
            base_v2h = result["final_score"]

            for attack_name, attacked_answer in variants.items():
                attacked_row = row.copy()
                attacked_row["student_answer"] = attacked_answer

                attacked = v2h_predict(
                    base_model=base_model,
                    strict_guard=strict_guard,
                    row=attacked_row,
                    rubric_concepts=rubric_concepts,
                    embedder=embedder,
                    nli=nli,
                )

                for model_name, clean_pred, attacked_pred in [
                    ("V2B", base_v2b, attacked["base_score"]),
                    ("V2H", base_v2h, attacked["final_score"]),
                ]:
                    attacks.append({
                        "evaluation": name,
                        "fold": fold_no,
                        "model": model_name,
                        "_row_id": int(row["_row_id"]),
                        "question_id": row["question_id"],
                        "human_score": truth,
                        "attack": attack_name,
                        "base_prediction": float(clean_pred),
                        "attacked_prediction":
                            float(attacked_pred),
                        "score_delta": float(
                            attacked_pred - clean_pred
                        ),
                        "gate_active": attacked["gate_active"],
                        "padding_gate": attacked["padding_gate"],
                        "contradiction_gate":
                            attacked["contradiction_gate"],
                        "evidence_word_ratio":
                            attacked["evidence_word_ratio"],
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

    return pd.DataFrame(preds), pd.DataFrame(attacks)


def adversarial_report(title, adv_df, model_name):
    part = adv_df[
        adv_df["model"].eq(model_name)
    ].copy()

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

    return {
        "padding_mean_delta":
            float(padding["score_delta"].mean()),
        "padding_mean_positive_delta":
            float(positive.mean()),
        "padding_worst_increase":
            float(padding["score_delta"].max()),
        "padding_fail_gt_0_5":
            float(np.mean(padding["score_delta"] > 0.5)),
        "padding_fail_gt_1_0":
            float(np.mean(padding["score_delta"] > 1.0)),
        "contradiction_mean_delta":
            float(contradiction["score_delta"].mean()),
        "contradiction_worst_increase":
            float(contradiction["score_delta"].max()),
    }


def main():
    print("=" * 80)
    print("RESEARCH GRADER V2H — CONTRASTIVE STRICT EVIDENCE")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH).reset_index(drop=True)

    with RUBRIC_PATH.open("r", encoding="utf-8") as f:
        rubric_concepts = json.load(f)

    df["_row_id"] = np.arange(len(df), dtype=int)
    df["human_score"] = pd.to_numeric(
        df["human_score"],
        errors="raise",
    ).astype(float)

    human = df[
        df["source"].astype(str).eq("human_group")
    ]
    weak = df[df["human_score"] <= 4.0]

    print(f"Dataset rows:           {len(df)}")
    print(f"Human gold rows:        {len(human)}")
    print(f"Weak rows <=4:          {len(weak)}")
    print("Contradiction:          forward NLI + explicit negation")
    print("Reverse NLI:            REMOVED from contradiction gating")
    print(
        f"Target-vs-other margin: >= {MIN_TARGET_MARGIN:.3f}"
    )
    print("Keyword/list rejection: enabled")
    print("Guard direction:        can lower V2B only")

    print("\nLoading sentence encoder...")
    embedder = v2b.EmbeddingCache(v2b.ENCODER_NAME)

    print("\nLoading NLI validator...")
    nli = v2c.NLICache(
        model_name=v2c.NLI_MODEL_NAME,
        device=v2c.NLI_DEVICE,
        batch_size=v2c.NLI_BATCH_SIZE,
    )

    clean_X = v2e.build_v2b_feature_frame(
        df=df,
        embedder=embedder,
        label="V2B clean feature extraction",
    )

    print("\nExtracting strict clean profiles...")
    strict_profiles = {}

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        strict_profiles[
            int(row["_row_id"])
        ] = strict_profile(
            row=row,
            rubric_concepts=rubric_concepts,
            embedder=embedder,
            nli=nli,
        )

        if i == 1 or i % 25 == 0 or i == len(df):
            print(
                f"Strict profile extraction: "
                f"{i}/{len(df)}"
            )

    print("\nBuilding V2B invariance attacks...")
    attack_df = v2e.build_v2b_attack_frame(df)

    attack_X = v2e.build_v2b_feature_frame(
        df=attack_df,
        embedder=embedder,
        label="V2B attack feature extraction",
    )

    # Human OOF
    human_fold_series = assign_human_folds(df)
    human_folds = []

    for fold_id in range(4):
        test_df = df[
            human_fold_series.eq(fold_id)
        ].copy()

        heldout = set(
            test_df["_row_id"]
            .astype(int)
            .tolist()
        )

        human_folds.append(
            (heldout, test_df)
        )

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
    weak_strata = weak[
        "question_id"
    ].astype(str).to_numpy()

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    weak_folds = []

    for _, test_pos in skf.split(
        weak_indices,
        weak_strata,
    ):
        test_indices = weak_indices[
            test_pos
        ]

        test_df = df.loc[
            test_indices
        ].copy()

        heldout = set(
            test_df["_row_id"]
            .astype(int)
            .tolist()
        )

        weak_folds.append(
            (heldout, test_df)
        )

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

        for model_name in [
            "V2B",
            "V2H",
        ]:
            part = pred_df[
                pred_df["model"].eq(model_name)
            ]

            y_true = part[
                "human_score"
            ].to_numpy(dtype=float)

            y_pred = part[
                "prediction"
            ].to_numpy(dtype=float)

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
                float(
                    part["gate_active"].mean()
                )
                if model_name == "V2H"
                else 0.0
            )

            summary_records.append({
                "evaluation": evaluation,
                "model": model_name,
                "clean_gate_rate": clean_gate_rate,
                **m,
                **a,
            })

    print("\n## V2H clean gate activation")
    for name, pred_df in [
        ("HUMAN_OOF", human_pred),
        ("WEAK_OOF", weak_pred),
    ]:
        part = pred_df[
            pred_df["model"].eq("V2H")
        ]

        print(
            f"{name}: "
            f"{part['gate_active'].mean() * 100:.1f}%"
        )

    print("\nTraining final V2H artifact...")

    final_base_model, final_clean = v2e.train_v2b_fold(
        clean_df=df,
        clean_X=clean_X,
        attack_df=attack_df,
        attack_X=attack_X,
        heldout_ids=set(),
    )

    final_guard = fit_strict_guard(
        final_clean,
        strict_profiles,
    )

    artifact = {
        "version": "research_grader_v2h",
        "architecture": (
            "V2B normal path + explicit-gaming gate + "
            "contrastive strict evidence verifier + "
            "forward-NLI/meta-negation contradiction guard"
        ),
        "base_model": final_base_model,
        "strict_coverage_guard": final_guard,
        "base_feature_names": v2b.FEATURE_NAMES,
        "embedding_model_name": v2b.ENCODER_NAME,
        "nli_model_name": v2c.NLI_MODEL_NAME,
        "rubric_concepts": rubric_concepts,
        "parameters": {
            "lenient_evidence_ratio_gate":
                LENIENT_EVIDENCE_RATIO_GATE,
            "strict_cosine_min":
                STRICT_COSINE_MIN,
            "strict_entailment_min":
                STRICT_ENTAILMENT_MIN,
            "strict_contradiction_max":
                STRICT_CONTRADICTION_MAX,
            "min_target_margin":
                MIN_TARGET_MARGIN,
            "max_sentence_bigram_repetition":
                MAX_SENTENCE_BIGRAM_REPETITION,
            "min_sentence_unique_ratio":
                MIN_SENTENCE_UNIQUE_RATIO,
            "max_question_lexical_overlap":
                MAX_QUESTION_LEXICAL_OVERLAP,
            "forward_contradiction_threshold":
                FORWARD_CONTRADICTION_THRESHOLD,
            "meta_negation_cosine_min":
                META_NEGATION_COSINE_MIN,
            "penalty_per_contradicted_concept":
                PENALTY_PER_CONTRADICTED_CONCEPT,
            "max_contradiction_penalty":
                MAX_CONTRADICTION_PENALTY,
        },
        "design_properties": {
            "normal_path_is_v2b": True,
            "reverse_nli_contradiction_removed": True,
            "target_vs_negative_rubric_margin": True,
            "keyword_list_rejection": True,
            "question_copy_rejection": True,
            "strict_entailment_required": True,
            "guard_can_raise_score": False,
        },
    }

    ARTIFACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        ARTIFACT_PATH,
    )

    human_pred.to_csv(
        HUMAN_PATH,
        index=False,
    )

    weak_pred.to_csv(
        WEAK_PATH,
        index=False,
    )

    combined_adv = pd.concat(
        [human_adv, weak_adv],
        ignore_index=True,
    )

    combined_adv.to_csv(
        ADV_PATH,
        index=False,
    )

    summary_df = pd.DataFrame(
        summary_records
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

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
