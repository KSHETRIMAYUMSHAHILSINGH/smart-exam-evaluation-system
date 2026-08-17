"""Fair human-only benchmark for Research Grader V2B vs V2C.

Why this benchmark exists
-------------------------
Previous reported metrics are not directly comparable:
- V2B used a random 25% holdout from the full dataset, mostly synthetic.
- V2C evaluated all 20 human_group answers out-of-fold.

This script puts BOTH graders on exactly the same evaluation protocol:
- data/training_dataset_v2c.csv (cleaned 135-row dataset)
- 4 human OOF folds
- every fold holds out exactly one human answer per question
- all 20 human answers receive one out-of-fold prediction
- padding/gaming robustness is tested on ALL held-out human answers
  (not only answers with score <= 4)
- contradiction injection is also tested for both models

No production artifacts are modified.

Required local files
--------------------
models/train_research_grader_v2b.py
models/train_research_grader_v2c.py
data/training_dataset_v2c.csv
data/rubric_concepts_v2c.json

Outputs
-------
data/research_graders_v2_human_fair_benchmark.csv
data/research_graders_v2_human_fair_adversarial.csv
data/research_graders_v2_human_fair_summary.csv
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import train_research_grader_v2b as v2b  # noqa: E402
import train_research_grader_v2c as v2c  # noqa: E402


DATA_PATH = ROOT / "data" / "training_dataset_v2c.csv"
RUBRIC_PATH = ROOT / "data" / "rubric_concepts_v2c.json"

PREDICTIONS_PATH = (
    ROOT / "data" / "research_graders_v2_human_fair_benchmark.csv"
)
ADVERSARIAL_PATH = (
    ROOT / "data" / "research_graders_v2_human_fair_adversarial.csv"
)
SUMMARY_PATH = (
    ROOT / "data" / "research_graders_v2_human_fair_summary.csv"
)


def clean_text(value: object) -> str:
    return v2c.clean_text(value)


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    result = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(
            math.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "within_1_mark": float(
            np.mean(np.abs(y_true - y_pred) <= 1.0)
        ),
        "within_2_marks": float(
            np.mean(np.abs(y_true - y_pred) <= 2.0)
        ),
    }

    if len(np.unique(y_true)) >= 2:
        result["r2"] = float(r2_score(y_true, y_pred))
    else:
        result["r2"] = float("nan")

    return result


def print_metrics(
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    m = metrics(y_true, y_pred)

    print(f"\n## {title}")
    print(f"MAE:            {m['mae']:.3f}")
    print(f"RMSE:           {m['rmse']:.3f}")
    print(f"R2:             {m['r2']:.3f}")
    print(
        f"Within 1 mark:  "
        f"{m['within_1_mark'] * 100:.1f}%"
    )
    print(
        f"Within 2 marks: "
        f"{m['within_2_marks'] * 100:.1f}%"
    )

    return m


def assign_human_folds(df: pd.DataFrame) -> pd.Series:
    """Use the same protocol as V2C."""
    fold = pd.Series(-1, index=df.index, dtype=int)

    human = df[
        df["source"].astype(str).eq("human_group")
    ]

    counts = human.groupby("question_id").size()

    if not (counts == 4).all():
        raise RuntimeError(
            "Expected exactly 4 human_group rows per question. "
            f"Found: {counts.to_dict()}"
        )

    for _, group in human.groupby(
        "question_id",
        sort=True,
    ):
        indices = list(group.index)

        for fold_id, idx in enumerate(indices):
            fold.loc[idx] = fold_id

    return fold


def build_v2b_attack_frame(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build V2B's five same-score attacks and retain parent row IDs."""
    records: list[dict[str, object]] = []

    for _, row in df.iterrows():
        variants = v2b.adversarial_variants(
            row,
            df,
        )

        for attack_name, attacked_answer in variants.items():
            record = row.to_dict()
            record["student_answer"] = attacked_answer
            record["_attack"] = attack_name
            record["_parent_row_id"] = int(
                row["_row_id"]
            )
            records.append(record)

    return pd.DataFrame(records)


def v2b_feature_frame(
    df: pd.DataFrame,
    embedder: v2b.EmbeddingCache,
    label: str,
) -> pd.DataFrame:
    out = v2b.build_feature_frame(
        df.reset_index(drop=True),
        embedder=embedder,
        show_progress=True,
        label=label,
    )
    out.index = df.index
    return out


def build_v2c_attack_frame(
    df: pd.DataFrame,
    rubric_concepts: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Use V2C's weak-answer invariance training policy."""
    return v2c.build_invariance_attack_frame(
        df=df,
        pool_df=df,
        rubric_concepts=rubric_concepts,
    )


def training_data_v2b(
    clean_df: pd.DataFrame,
    clean_X: pd.DataFrame,
    attack_df: pd.DataFrame,
    attack_X: pd.DataFrame,
    heldout_ids: set[int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    clean_mask = ~clean_df["_row_id"].isin(
        heldout_ids
    )

    train_clean = clean_df.loc[clean_mask]
    X_clean = clean_X.loc[train_clean.index]

    attack_mask = ~attack_df[
        "_parent_row_id"
    ].isin(heldout_ids)

    train_attack = attack_df.loc[attack_mask]
    X_attack = attack_X.loc[
        train_attack.index
    ]

    X_train = pd.concat(
        [
            X_clean.reset_index(drop=True),
            X_attack.reset_index(drop=True),
        ],
        ignore_index=True,
    )

    y_train = np.concatenate(
        [
            train_clean[
                "human_score"
            ].to_numpy(dtype=float),
            train_attack[
                "human_score"
            ].to_numpy(dtype=float),
        ]
    )

    weights = np.concatenate(
        [
            v2b.sample_weights(
                train_clean,
                adversarial=False,
            ),
            v2b.sample_weights(
                train_attack,
                adversarial=True,
            ),
        ]
    )

    return X_train, y_train, weights


def training_data_v2c(
    clean_df: pd.DataFrame,
    clean_X: pd.DataFrame,
    attack_df: pd.DataFrame,
    attack_X: pd.DataFrame,
    heldout_ids: set[int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    return v2c.training_matrix_for_fold(
        clean_df=clean_df,
        clean_X=clean_X,
        adversarial_df=attack_df,
        adversarial_X=attack_X,
        heldout_row_ids=heldout_ids,
    )


def all_eval_attacks(
    row: pd.Series,
    df: pd.DataFrame,
    rubric_concepts: dict[str, dict[str, object]],
) -> dict[str, str]:
    """Exactly five padding attacks + one contradiction attack."""
    attacks = v2c.adversarial_variants(
        row=row,
        pool_df=df,
        rubric_concepts=rubric_concepts,
        include_contradiction=True,
    )
    return attacks


def v2b_one_prediction(
    model,
    row: pd.Series,
    embedder: v2b.EmbeddingCache,
) -> float:
    feature = v2b.make_features(
        question_text=row["question_text"],
        model_answer=row["model_answer"],
        student_answer=row["student_answer"],
        rubric_keywords=row["rubric_keywords"],
        embedder=embedder,
    )

    X = pd.DataFrame(
        [feature],
        columns=v2b.FEATURE_NAMES,
    )

    return float(
        np.clip(model.predict(X)[0], 0.0, 10.0)
    )


def v2c_one_prediction(
    model,
    row: pd.Series,
    nli: v2c.NLICache,
    rubric_concepts: dict[str, dict[str, object]],
) -> float:
    result = v2c.make_features(
        row=row,
        rubric_concepts=rubric_concepts,
        nli=nli,
    )

    X = pd.DataFrame(
        [result.features],
        columns=v2c.FEATURE_NAMES,
    )

    return float(
        np.clip(model.predict(X)[0], 0.0, 10.0)
    )


def adversarial_summary(
    df: pd.DataFrame,
    model_name: str,
) -> dict[str, float]:
    model_df = df[
        df["model"].eq(model_name)
    ].copy()

    padding = model_df[
        ~model_df["attack"].eq(
            "contradiction_injection"
        )
    ].copy()

    positive = padding[
        "score_delta"
    ].clip(lower=0.0)

    contradiction = model_df[
        model_df["attack"].eq(
            "contradiction_injection"
        )
    ]

    result = {
        "model": model_name,
        "padding_mean_delta": float(
            padding["score_delta"].mean()
        ),
        "padding_mean_positive_delta": float(
            positive.mean()
        ),
        "padding_worst_increase": float(
            padding["score_delta"].max()
        ),
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

    return result


def print_adversarial(
    adversarial_df: pd.DataFrame,
    model_name: str,
) -> dict[str, float]:
    model_df = adversarial_df[
        adversarial_df["model"].eq(model_name)
    ].copy()

    padding = model_df[
        ~model_df["attack"].eq(
            "contradiction_injection"
        )
    ].copy()

    positive = padding[
        "score_delta"
    ].clip(lower=0.0)

    contradiction = model_df[
        model_df["attack"].eq(
            "contradiction_injection"
        )
    ].copy()

    print(
        f"\n## {model_name} — ALL HUMAN OOF ADVERSARIAL"
    )
    print(
        f"Human base answers:          "
        f"{padding['_row_id'].nunique()}"
    )
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
        padding.groupby("attack")[
            "score_delta"
        ]
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

    return adversarial_summary(
        adversarial_df,
        model_name,
    )


def main() -> None:
    print("=" * 78)
    print("RESEARCH GRADER V2 — FAIR HUMAN BENCHMARK")
    print("=" * 78)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {DATA_PATH}"
        )

    if not RUBRIC_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {RUBRIC_PATH}"
        )

    df = pd.read_csv(DATA_PATH).reset_index(
        drop=True
    )

    with RUBRIC_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        rubric_concepts = json.load(f)

    df["_row_id"] = np.arange(
        len(df),
        dtype=int,
    )
    df["human_score"] = pd.to_numeric(
        df["human_score"],
        errors="raise",
    ).astype(float)

    folds = assign_human_folds(df)
    df["_human_fold"] = folds

    human = df[
        df["source"].astype(str).eq(
            "human_group"
        )
    ]

    print(f"Dataset rows:    {len(df)}")
    print(f"Human rows:      {len(human)}")
    print(
        f"Human score min: "
        f"{human['human_score'].min():.1f}"
    )
    print(
        f"Human score max: "
        f"{human['human_score'].max():.1f}"
    )
    print("Protocol:        same 4 human OOF folds for both models")

    # ---------------------------------------------------------------
    # Feature engines
    # ---------------------------------------------------------------
    print("\nLoading V2B embedding encoder...")
    embedder = v2b.EmbeddingCache(
        v2b.ENCODER_NAME
    )

    print("\nLoading V2C NLI encoder...")
    nli = v2c.NLICache(
        model_name=v2c.NLI_MODEL_NAME,
        device=v2c.NLI_DEVICE,
        batch_size=v2c.NLI_BATCH_SIZE,
    )

    # ---------------------------------------------------------------
    # Precompute clean features.
    # ---------------------------------------------------------------
    v2b_clean_X = v2b_feature_frame(
        df,
        embedder,
        "V2B clean feature extraction",
    )

    v2c_clean_X = v2c.build_feature_frame(
        df=df,
        rubric_concepts=rubric_concepts,
        nli=nli,
        label="V2C clean NLI extraction",
    )

    # ---------------------------------------------------------------
    # Precompute attack-training features.
    # ---------------------------------------------------------------
    print("\nBuilding V2B invariance attacks...")
    v2b_attack_df = build_v2b_attack_frame(
        df
    )
    v2b_attack_df = v2b_attack_df.reset_index(
        drop=True
    )
    v2b_attack_X = v2b_feature_frame(
        v2b_attack_df,
        embedder,
        "V2B attack feature extraction",
    )

    print("\nBuilding V2C weak-answer invariance attacks...")
    v2c_attack_df = build_v2c_attack_frame(
        df,
        rubric_concepts,
    ).reset_index(drop=True)

    if len(v2c_attack_df):
        v2c_attack_X = v2c.build_feature_frame(
            df=v2c_attack_df,
            rubric_concepts=rubric_concepts,
            nli=nli,
            label="V2C attack NLI extraction",
        )
    else:
        v2c_attack_X = pd.DataFrame(
            columns=v2c.FEATURE_NAMES
        )

    # ---------------------------------------------------------------
    # Exact same folds for both models.
    # ---------------------------------------------------------------
    prediction_records: list[
        dict[str, object]
    ] = []
    adversarial_records: list[
        dict[str, object]
    ] = []

    for fold_id in range(4):
        test_df = df[
            df["_human_fold"].eq(fold_id)
        ].copy()

        if len(test_df) != 5:
            raise RuntimeError(
                f"Fold {fold_id} expected 5 human rows, "
                f"found {len(test_df)}"
            )

        heldout_ids = set(
            test_df["_row_id"].astype(
                int
            ).tolist()
        )

        # ---------------- V2B ----------------
        X_b, y_b, w_b = training_data_v2b(
            clean_df=df,
            clean_X=v2b_clean_X,
            attack_df=v2b_attack_df,
            attack_X=v2b_attack_X,
            heldout_ids=heldout_ids,
        )

        model_b = v2b.make_regressor()
        model_b.fit(
            X_b,
            y_b,
            sample_weight=w_b,
        )

        # ---------------- V2C ----------------
        X_c, y_c, w_c = training_data_v2c(
            clean_df=df,
            clean_X=v2c_clean_X,
            attack_df=v2c_attack_df,
            attack_X=v2c_attack_X,
            heldout_ids=heldout_ids,
        )

        model_c = v2c.make_calibrator()
        model_c.fit(
            X_c,
            y_c,
            ridge__sample_weight=w_c,
        )

        for _, row in test_df.iterrows():
            base_b = v2b_one_prediction(
                model_b,
                row,
                embedder,
            )
            base_c = v2c_one_prediction(
                model_c,
                row,
                nli,
                rubric_concepts,
            )

            for model_name, pred in [
                ("V2B", base_b),
                ("V2C", base_c),
            ]:
                prediction_records.append(
                    {
                        "model": model_name,
                        "fold": fold_id,
                        "_row_id": int(
                            row["_row_id"]
                        ),
                        "question_id": row[
                            "question_id"
                        ],
                        "human_score": float(
                            row["human_score"]
                        ),
                        "prediction": pred,
                        "absolute_error": abs(
                            float(
                                row["human_score"]
                            )
                            - pred
                        ),
                        "student_answer": row[
                            "student_answer"
                        ],
                    }
                )

            attacks = all_eval_attacks(
                row=row,
                df=df,
                rubric_concepts=rubric_concepts,
            )

            for attack_name, attacked_answer in attacks.items():
                attacked_row = row.copy()
                attacked_row[
                    "student_answer"
                ] = attacked_answer

                attacked_b = v2b_one_prediction(
                    model_b,
                    attacked_row,
                    embedder,
                )
                attacked_c = v2c_one_prediction(
                    model_c,
                    attacked_row,
                    nli,
                    rubric_concepts,
                )

                for model_name, base_pred, attacked_pred in [
                    (
                        "V2B",
                        base_b,
                        attacked_b,
                    ),
                    (
                        "V2C",
                        base_c,
                        attacked_c,
                    ),
                ]:
                    adversarial_records.append(
                        {
                            "model": model_name,
                            "fold": fold_id,
                            "_row_id": int(
                                row["_row_id"]
                            ),
                            "question_id": row[
                                "question_id"
                            ],
                            "human_score": float(
                                row["human_score"]
                            ),
                            "attack": attack_name,
                            "base_prediction": (
                                base_pred
                            ),
                            "attacked_prediction": (
                                attacked_pred
                            ),
                            "score_delta": (
                                attacked_pred
                                - base_pred
                            ),
                            "base_answer": row[
                                "student_answer"
                            ],
                            "attacked_answer": (
                                attacked_answer
                            ),
                        }
                    )

        print(
            f"Fold {fold_id + 1}/4 complete."
        )

    pred_df = pd.DataFrame(
        prediction_records
    )
    adv_df = pd.DataFrame(
        adversarial_records
    )

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------
    summary_records: list[
        dict[str, object]
    ] = []

    for model_name in ["V2B", "V2C"]:
        part = pred_df[
            pred_df["model"].eq(
                model_name
            )
        ]

        y_true = part[
            "human_score"
        ].to_numpy(dtype=float)
        y_pred = part[
            "prediction"
        ].to_numpy(dtype=float)

        m = print_metrics(
            f"{model_name} — HUMAN GOLD OOF",
            y_true,
            y_pred,
        )

        adv = print_adversarial(
            adv_df,
            model_name,
        )

        summary_records.append(
            {
                "model": model_name,
                **m,
                **{
                    key: value
                    for key, value in adv.items()
                    if key != "model"
                },
            }
        )

    print("\n## Side-by-side")
    summary_df = pd.DataFrame(
        summary_records
    )

    display_cols = [
        "model",
        "mae",
        "rmse",
        "r2",
        "within_1_mark",
        "within_2_marks",
        "padding_mean_positive_delta",
        "padding_worst_increase",
        "padding_fail_gt_0_5",
        "contradiction_mean_delta",
        "contradiction_worst_increase",
    ]

    print(
        summary_df[
            display_cols
        ].round(3).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Per-question MAE
    # ---------------------------------------------------------------
    print("\n## Human OOF MAE by question")
    per_q = (
        pred_df.groupby(
            ["model", "question_id"]
        )
        .apply(
            lambda g: mean_absolute_error(
                g["human_score"],
                g["prediction"],
            ),
            include_groups=False,
        )
        .rename("mae")
        .reset_index()
    )

    print(
        per_q.pivot(
            index="question_id",
            columns="model",
            values="mae",
        )
        .round(3)
        .to_string()
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )
    adv_df.to_csv(
        ADVERSARIAL_PATH,
        index=False,
    )
    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\n## Saved")
    print(
        f"Predictions: {PREDICTIONS_PATH}"
    )
    print(
        f"Adversarial: {ADVERSARIAL_PATH}"
    )
    print(
        f"Summary:     {SUMMARY_PATH}"
    )

    print(
        "\nNo production artifact was modified."
    )


if __name__ == "__main__":
    main()
