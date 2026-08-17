"""Research Grader V2F — selective evidence bottleneck.

Architecture
------------
Normal answer:
    V2B score -> final

Suspiciously padded answer:
    V2B score
      + rubric evidence extraction
      -> evidence-only V2B ceiling
      -> rubric coverage ceiling
      -> final cannot exceed either ceiling

High-confidence contradiction:
    activates the evidence bottleneck
    + subtracts marks per contradicted rubric concept

Why selective gating?
---------------------
V2E gate diagnostic showed:
- evidence_word_ratio < 0.70 triggers only 5% of clean human answers
- triggers 99.5% of weak padding attacks
- triggers 100% of weak combined_long_weak attacks

This should preserve V2B human accuracy while retaining V2E's weak-answer defense.

Inputs
------
data/training_dataset_v2c.csv
data/rubric_concepts_v2c.json
models/train_research_grader_v2b.py
models/train_research_grader_v2c.py
models/train_research_grader_v2e.py

Outputs
-------
models/artifacts/research_grader_v2f.joblib
data/research_grader_v2f_human_oof.csv
data/research_grader_v2f_weak_oof.csv
data/research_grader_v2f_adversarial.csv
data/research_grader_v2f_summary.csv

Nothing from previous graders is overwritten.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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

ARTIFACT_PATH = ROOT / "models" / "artifacts" / "research_grader_v2f.joblib"
HUMAN_PATH = ROOT / "data" / "research_grader_v2f_human_oof.csv"
WEAK_PATH = ROOT / "data" / "research_grader_v2f_weak_oof.csv"
ADV_PATH = ROOT / "data" / "research_grader_v2f_adversarial.csv"
SUMMARY_PATH = ROOT / "data" / "research_grader_v2f_summary.csv"

RANDOM_STATE = 42

# Gate chosen from the V2E diagnostic.
EVIDENCE_RATIO_GATE = 0.70

# Because the bottleneck now activates selectively, it can be stricter than V2E.
EVIDENCE_SCORE_SLACK = 0.35

# High-confidence concept contradiction.
CONTRADICTION_THRESHOLD = 0.85
PENALTY_PER_CONTRADICTED_CONCEPT = 1.50
MAX_CONTRADICTION_PENALTY = 3.50


def metrics(y_true, y_pred):
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
    m = metrics(y_true, y_pred)

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


def contradiction_profile(row, rubric_concepts, nli):
    qid = str(row["question_id"])

    criteria = [
        v2c.clean_text(x)
        for x in rubric_concepts[qid]["criteria"]
        if v2c.clean_text(x)
    ]

    sentences = v2c.split_sentences(
        v2c.clean_text(row["student_answer"])
    )

    pairs = [
        (sentence, criterion)
        for criterion in criteria
        for sentence in sentences
    ]

    if not pairs:
        return 0, 0.0

    probs = nli.probabilities(pairs)
    probs = probs.reshape(len(criteria), len(sentences), 3)

    max_per_concept = probs[:, :, v2c.CONTRADICTION_INDEX].max(axis=1)

    count = int(
        np.sum(max_per_concept >= CONTRADICTION_THRESHOLD)
    )

    maximum = float(max_per_concept.max())

    return count, maximum


def v2f_predict(
    base_model,
    coverage_guard,
    row,
    rubric_concepts,
    embedder,
    nli,
):
    base_score = v2e.v2b_predict_one(
        model=base_model,
        row=row,
        embedder=embedder,
    )

    profile = v2e.extract_evidence_profile(
        row=row,
        rubric_concepts=rubric_concepts,
        embedder=embedder,
        nli=nli,
    )

    contradicted_count, max_contradiction = contradiction_profile(
        row=row,
        rubric_concepts=rubric_concepts,
        nli=nli,
    )

    padding_gate = (
        profile.evidence_word_ratio < EVIDENCE_RATIO_GATE
    )

    contradiction_gate = contradicted_count > 0

    gate_active = padding_gate or contradiction_gate

    evidence_score = base_score
    evidence_ceiling = 10.0
    rubric_ceiling = 10.0

    if gate_active:
        if profile.evidence_text:
            evidence_row = row.copy()
            evidence_row["student_answer"] = profile.evidence_text

            evidence_score = v2e.v2b_predict_one(
                model=base_model,
                row=evidence_row,
                embedder=embedder,
            )
        else:
            evidence_score = 0.0

        evidence_ceiling = float(
            np.clip(
                evidence_score + EVIDENCE_SCORE_SLACK,
                0.0,
                10.0,
            )
        )

        rubric_ceiling = v2e.coverage_ceiling(
            guard=coverage_guard,
            coverage_index=profile.coverage_index,
        )

        guarded_score = min(
            base_score,
            evidence_ceiling,
            rubric_ceiling,
        )
    else:
        guarded_score = base_score

    contradiction_penalty = min(
        MAX_CONTRADICTION_PENALTY,
        contradicted_count * PENALTY_PER_CONTRADICTED_CONCEPT,
    )

    final_score = float(
        np.clip(
            guarded_score - contradiction_penalty,
            0.0,
            10.0,
        )
    )

    return {
        "base_score": float(base_score),
        "final_score": final_score,
        "gate_active": bool(gate_active),
        "padding_gate": bool(padding_gate),
        "contradiction_gate": bool(contradiction_gate),
        "evidence_word_ratio": float(profile.evidence_word_ratio),
        "coverage_index": float(profile.coverage_index),
        "supported_count": int(profile.supported_count),
        "evidence_score": float(evidence_score),
        "evidence_ceiling": float(evidence_ceiling),
        "rubric_ceiling": float(rubric_ceiling),
        "contradicted_concepts": int(contradicted_count),
        "max_contradiction": float(max_contradiction),
        "contradiction_penalty": float(contradiction_penalty),
        "evidence_text": profile.evidence_text,
    }


def evaluate(
    name,
    folds,
    df,
    clean_X,
    attack_df,
    attack_X,
    clean_profiles,
    rubric_concepts,
    embedder,
    nli,
):
    predictions = []
    attacks = []

    coverage_by_row_id = {
        row_id: profile.coverage_index
        for row_id, profile in clean_profiles.items()
    }

    for fold_no, (heldout_ids, test_df) in enumerate(folds, start=1):
        base_model, train_clean = v2e.train_v2b_fold(
            clean_df=df,
            clean_X=clean_X,
            attack_df=attack_df,
            attack_X=attack_X,
            heldout_ids=heldout_ids,
        )

        coverage_guard = v2e.fit_coverage_guard(
            train_df=train_clean,
            coverage_by_row_id=coverage_by_row_id,
        )

        for _, row in test_df.iterrows():
            result = v2f_predict(
                base_model=base_model,
                coverage_guard=coverage_guard,
                row=row,
                rubric_concepts=rubric_concepts,
                embedder=embedder,
                nli=nli,
            )

            truth = float(row["human_score"])

            for model_name, pred in [
                ("V2B", result["base_score"]),
                ("V2F", result["final_score"]),
            ]:
                predictions.append({
                    "evaluation": name,
                    "fold": fold_no,
                    "model": model_name,
                    "_row_id": int(row["_row_id"]),
                    "question_id": row["question_id"],
                    "source": row["source"],
                    "human_score": truth,
                    "prediction": pred,
                    "absolute_error": abs(truth - pred),
                    "gate_active": result["gate_active"],
                    "padding_gate": result["padding_gate"],
                    "contradiction_gate": result["contradiction_gate"],
                    "evidence_word_ratio": result["evidence_word_ratio"],
                    "coverage_index": result["coverage_index"],
                    "supported_count": result["supported_count"],
                    "contradicted_concepts": result["contradicted_concepts"],
                    "contradiction_penalty": result["contradiction_penalty"],
                    "evidence_text": result["evidence_text"],
                    "student_answer": row["student_answer"],
                })

            variants = v2e.all_eval_attacks(
                row=row,
                df=df,
                rubric_concepts=rubric_concepts,
            )

            base_v2b = result["base_score"]
            base_v2f = result["final_score"]

            for attack_name, attacked_answer in variants.items():
                attacked_row = row.copy()
                attacked_row["student_answer"] = attacked_answer

                attacked = v2f_predict(
                    base_model=base_model,
                    coverage_guard=coverage_guard,
                    row=attacked_row,
                    rubric_concepts=rubric_concepts,
                    embedder=embedder,
                    nli=nli,
                )

                for model_name, clean_pred, attacked_pred in [
                    ("V2B", base_v2b, attacked["base_score"]),
                    ("V2F", base_v2f, attacked["final_score"]),
                ]:
                    attacks.append({
                        "evaluation": name,
                        "fold": fold_no,
                        "model": model_name,
                        "_row_id": int(row["_row_id"]),
                        "question_id": row["question_id"],
                        "human_score": truth,
                        "attack": attack_name,
                        "base_prediction": clean_pred,
                        "attacked_prediction": attacked_pred,
                        "score_delta": attacked_pred - clean_pred,
                        "gate_active": attacked["gate_active"],
                        "padding_gate": attacked["padding_gate"],
                        "contradiction_gate": attacked["contradiction_gate"],
                        "evidence_word_ratio": attacked["evidence_word_ratio"],
                        "contradicted_concepts": attacked["contradicted_concepts"],
                        "contradiction_penalty": attacked["contradiction_penalty"],
                    })

        print(
            f"{name}: fold {fold_no}/{len(folds)} complete "
            f"(held out={len(test_df)})"
        )

    return pd.DataFrame(predictions), pd.DataFrame(attacks)


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
    print(f"Padding mean delta:          {padding['score_delta'].mean():+.3f}")
    print(f"Padding mean positive delta: {positive.mean():+.3f}")
    print(f"Padding worst increase:      {padding['score_delta'].max():+.3f}")
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
    print(f"Mean delta:                  {contradiction['score_delta'].mean():+.3f}")
    print(f"Worst increase:              {contradiction['score_delta'].max():+.3f}")
    print(
        f"Raised score > +0.5:         "
        f"{(contradiction['score_delta'] > 0.5).mean() * 100:.1f}%"
    )

    if model_name == "V2F":
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


def main():
    print("=" * 78)
    print("RESEARCH GRADER V2F — SELECTIVE EVIDENCE BOTTLENECK")
    print("=" * 78)

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

    print(f"Dataset rows:             {len(df)}")
    print(f"Human gold rows:          {len(human)}")
    print(f"Weak rows <=4:            {len(weak)}")
    print(f"Evidence ratio gate:      < {EVIDENCE_RATIO_GATE:.2f}")
    print(f"Selective evidence slack: +{EVIDENCE_SCORE_SLACK:.2f}")
    print(f"Contradiction threshold:  {CONTRADICTION_THRESHOLD:.2f}")
    print(
        f"Penalty / contradicted concept: "
        f"{PENALTY_PER_CONTRADICTED_CONCEPT:.2f}"
    )

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

    print("\nExtracting clean evidence profiles...")
    clean_profiles = {}

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        clean_profiles[int(row["_row_id"])] = v2e.extract_evidence_profile(
            row=row,
            rubric_concepts=rubric_concepts,
            embedder=embedder,
            nli=nli,
        )

        if i == 1 or i % 25 == 0 or i == len(df):
            print(f"Clean evidence extraction: {i}/{len(df)}")

    print("\nBuilding V2B invariance attacks...")
    attack_df = v2e.build_v2b_attack_frame(df)

    attack_X = v2e.build_v2b_feature_frame(
        df=attack_df,
        embedder=embedder,
        label="V2B attack feature extraction",
    )

    # Human folds
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
        clean_profiles=clean_profiles,
        rubric_concepts=rubric_concepts,
        embedder=embedder,
        nli=nli,
    )

    # Weak folds
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
        clean_profiles=clean_profiles,
        rubric_concepts=rubric_concepts,
        embedder=embedder,
        nli=nli,
    )

    summary_records = []

    for evaluation, pred_df, adv_df in [
        ("HUMAN_OOF", human_pred, human_adv),
        ("WEAK_OOF", weak_pred, weak_adv),
    ]:
        print("\n" + "=" * 70)
        print(evaluation)
        print("=" * 70)

        for model_name in ["V2B", "V2F"]:
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
                if model_name == "V2F"
                else 0.0
            )

            summary_records.append({
                "evaluation": evaluation,
                "model": model_name,
                "clean_gate_rate": clean_gate_rate,
                **m,
                **a,
            })

    print("\n## V2F clean gate activation")
    for name, pred_df in [
        ("HUMAN_OOF", human_pred),
        ("WEAK_OOF", weak_pred),
    ]:
        part = pred_df[pred_df["model"].eq("V2F")]
        print(
            f"{name}: "
            f"{part['gate_active'].mean() * 100:.1f}%"
        )

    # Final model
    print("\nTraining final V2F artifact...")

    final_model, final_clean = v2e.train_v2b_fold(
        clean_df=df,
        clean_X=clean_X,
        attack_df=attack_df,
        attack_X=attack_X,
        heldout_ids=set(),
    )

    coverage_by_row_id = {
        row_id: profile.coverage_index
        for row_id, profile in clean_profiles.items()
    }

    final_guard = v2e.fit_coverage_guard(
        train_df=final_clean,
        coverage_by_row_id=coverage_by_row_id,
    )

    artifact = {
        "version": "research_grader_v2f",
        "architecture": (
            "V2B base + selective V2E evidence bottleneck "
            "+ per-concept contradiction penalty"
        ),
        "base_model": final_model,
        "coverage_guard": final_guard,
        "base_feature_names": v2b.FEATURE_NAMES,
        "embedding_model_name": v2b.ENCODER_NAME,
        "nli_model_name": v2c.NLI_MODEL_NAME,
        "rubric_concepts": rubric_concepts,
        "parameters": {
            "evidence_ratio_gate": EVIDENCE_RATIO_GATE,
            "evidence_score_slack": EVIDENCE_SCORE_SLACK,
            "contradiction_threshold": CONTRADICTION_THRESHOLD,
            "penalty_per_contradicted_concept":
                PENALTY_PER_CONTRADICTED_CONCEPT,
            "max_contradiction_penalty": MAX_CONTRADICTION_PENALTY,
        },
        "design_properties": {
            "normal_path_is_v2b": True,
            "guard_can_raise_score": False,
            "padding_gate_is_selective": True,
            "contradiction_is_per_rubric_concept": True,
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
