"""Benchmark existing pretrained ASAG models on our grading dataset.

Models
------
1. kenzykhaled/XLENT_ASAG
2. kenzykhaled/xlnet-regression

Both are XLNet regression models trained on Meyerger/ASAG2024 and expect:
    text      = student answer
    text_pair = reference answer
Their model cards describe the output as a normalized 0-1 grade.

This script:
- DOES NOT train anything.
- DOES NOT modify any existing model/dataset.
- Downloads pretrained models from Hugging Face on first run.
- Evaluates clean overall, human_group, and weak <=4 subsets.
- Evaluates the already-known V2H diagnostic attack families if the local
  blind-test helper exists. This is called a DIAGNOSTIC robustness test, not a
  new blind test.

Outputs
-------
data/pretrained_asag_predictions.csv
data/pretrained_asag_summary.csv
data/pretrained_asag_robustness.csv
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

DATA_PATH = ROOT / "data" / "training_dataset_v2c.csv"

PRED_PATH = ROOT / "data" / "pretrained_asag_predictions.csv"
SUMMARY_PATH = ROOT / "data" / "pretrained_asag_summary.csv"
ROBUST_PATH = ROOT / "data" / "pretrained_asag_robustness.csv"

MODEL_NAMES = [
    "kenzykhaled/XLENT_ASAG",
    "kenzykhaled/xlnet-regression",
]

BATCH_SIZE = 16
MAX_LENGTH = 512


def device_name():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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


class ASAGModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = device_name()

        print(f"\nLoading: {model_name}")
        print(f"Device:  {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, students, references):
        outputs_all = []

        for start in range(0, len(students), BATCH_SIZE):
            batch_students = students[start:start + BATCH_SIZE]
            batch_refs = references[start:start + BATCH_SIZE]

            encoded = self.tokenizer(
                batch_students,
                batch_refs,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            with torch.no_grad():
                outputs = self.model(**encoded)

            raw = (
                outputs.logits
                .detach()
                .float()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            # Model card specifies normalized 0-1 regression output.
            normalized = np.clip(raw, 0.0, 1.0)
            marks = normalized * 10.0

            outputs_all.extend(marks.tolist())

        return np.asarray(outputs_all, dtype=float)


def evaluate_clean(model, df):
    pred = model.predict_batch(
        df["student_answer"].fillna("").astype(str).tolist(),
        df["model_answer"].fillna("").astype(str).tolist(),
    )

    out = df[
        [
            "question_id",
            "source",
            "human_score",
            "student_answer",
            "model_answer",
        ]
    ].copy()

    out["model"] = model.model_name
    out["prediction"] = pred
    out["absolute_error"] = np.abs(
        out["human_score"].astype(float) - pred
    )

    return out


def print_slice(model_name, label, frame):
    m = metrics(
        frame["human_score"].astype(float),
        frame["prediction"].astype(float),
    )

    print(f"\n## {model_name} — {label}")
    print(f"Rows:            {len(frame)}")
    print(f"MAE:             {m['mae']:.3f}")
    print(f"RMSE:            {m['rmse']:.3f}")
    print(f"R2:              {m['r2']:.3f}")
    print(f"Within 1 mark:   {m['within_1_mark'] * 100:.1f}%")
    print(f"Within 2 marks:  {m['within_2_marks'] * 100:.1f}%")

    return {
        "model": model_name,
        "evaluation": label,
        "rows": len(frame),
        **m,
    }


def load_diagnostic_attack_helper():
    """Reuse known V2H attack generator only if available.

    This is NOT treated as a fresh blind benchmark.
    """
    try:
        import blind_test_research_grader_v2h as blind
        return blind
    except Exception as exc:
        print(
            "\nKnown robustness helper could not be imported; "
            "clean benchmark will still run."
        )
        print(f"Reason: {exc}")
        return None


def evaluate_known_robustness(model, df, helper):
    records = []

    if helper is None:
        return pd.DataFrame()

    print(
        "\nRunning known V2H diagnostic attacks "
        "(NOT a fresh blind test)..."
    )

    # Focus robustness on weak answers because that is our core failure region.
    weak = df[df["human_score"].astype(float) <= 4.0].copy()

    clean_pred = model.predict_batch(
        weak["student_answer"].fillna("").astype(str).tolist(),
        weak["model_answer"].fillna("").astype(str).tolist(),
    )

    for local_index, (_, row) in enumerate(weak.iterrows()):
        variants = helper.blind_variants(row, df)

        names = list(variants.keys())
        attacked_answers = [variants[name] for name in names]
        refs = [str(row["model_answer"])] * len(names)

        attacked_pred = model.predict_batch(
            attacked_answers,
            refs,
        )

        for attack_name, prediction in zip(names, attacked_pred):
            records.append({
                "model": model.model_name,
                "question_id": row["question_id"],
                "human_score": float(row["human_score"]),
                "attack": attack_name,
                "family": helper.attack_family(attack_name),
                "base_prediction": float(clean_pred[local_index]),
                "attacked_prediction": float(prediction),
                "score_delta": float(
                    prediction - clean_pred[local_index]
                ),
            })

    return pd.DataFrame(records)


def print_robustness(frame):
    if frame.empty:
        return

    print("\n## Known robustness diagnostic — WEAK <=4")

    summary = (
        frame.groupby(["model", "family"])
        .agg(
            mean_delta=("score_delta", "mean"),
            mean_positive_delta=(
                "score_delta",
                lambda x: float(np.maximum(x, 0).mean()),
            ),
            worst_increase=("score_delta", "max"),
            fail_gt_0_5=(
                "score_delta",
                lambda x: float(np.mean(x > 0.5)),
            ),
            fail_gt_1_0=(
                "score_delta",
                lambda x: float(np.mean(x > 1.0)),
            ),
            count=("score_delta", "size"),
        )
        .reset_index()
    )

    print(summary.round(3).to_string(index=False))


def main():
    print("=" * 78)
    print("PRETRAINED ASAG BENCHMARK")
    print("=" * 78)
    print("No training. No existing artifact is modified.")

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH).reset_index(drop=True)
    df["_row_id"] = np.arange(len(df), dtype=int)
    df["human_score"] = pd.to_numeric(
        df["human_score"],
        errors="raise",
    ).astype(float)

    print(f"\nDataset rows: {len(df)}")
    print(
        f"Human rows:   "
        f"{(df['source'].astype(str) == 'human_group').sum()}"
    )
    print(f"Weak <=4:     {(df['human_score'] <= 4).sum()}")

    all_predictions = []
    summary_records = []
    all_robustness = []

    helper = load_diagnostic_attack_helper()

    for model_name in MODEL_NAMES:
        grader = ASAGModel(model_name)

        clean = evaluate_clean(grader, df)
        all_predictions.append(clean)

        slices = {
            "ALL_135": clean,
            "HUMAN_20": clean[
                clean["source"].astype(str).eq("human_group")
            ],
            "WEAK_40": clean[
                clean["human_score"].astype(float) <= 4.0
            ],
        }

        for label, frame in slices.items():
            summary_records.append(
                print_slice(model_name, label, frame)
            )

        robustness = evaluate_known_robustness(
            grader,
            df,
            helper,
        )

        if not robustness.empty:
            all_robustness.append(robustness)

        # Free memory before loading next checkpoint.
        del grader
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    summary = pd.DataFrame(summary_records)

    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(PRED_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n" + "=" * 78)
    print("SIDE-BY-SIDE CLEAN SUMMARY")
    print("=" * 78)
    print(
        summary[
            [
                "model",
                "evaluation",
                "mae",
                "rmse",
                "r2",
                "within_1_mark",
                "within_2_marks",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    if all_robustness:
        robustness = pd.concat(
            all_robustness,
            ignore_index=True,
        )
        robustness.to_csv(ROBUST_PATH, index=False)
        print_robustness(robustness)
        print(f"\nRobustness CSV: {ROBUST_PATH}")
    else:
        print(
            "\nRobustness diagnostic skipped because the helper "
            "could not be loaded."
        )

    print("\n## Saved")
    print(f"Predictions: {PRED_PATH}")
    print(f"Summary:     {SUMMARY_PATH}")
    print(
        "\nPrevious datasets/models/app files were not modified."
    )


if __name__ == "__main__":
    main()
