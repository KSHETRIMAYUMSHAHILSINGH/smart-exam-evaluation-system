"""Research Grader V3 — lightweight Difference-Aware Contrastive Learning.

Research basis
--------------
Adapted from:
Chen et al., "DiffCL: Difference-Aware Contrastive Learning for Automatic
Answer Grading with Multi-Level Semantic Modeling", ACL Findings 2026.

Original DiffCL ideas preserved:
- explicit difference information between student/reference answers
- same-score InfoNCE
- hierarchical score-gap contrastive loss:
    gap <= 1  -> pull together
    gap > 1   -> keep cosine similarity <= 0.5
- joint supervised + contrastive representation learning

Adaptations for this project
----------------------------
The original paper fine-tunes much larger encoders on datasets with thousands
of examples. We have only 135 examples, so V3 intentionally uses:

1. Frozen sentence-transformers/all-MiniLM-L6-v2 embeddings.
2. A small trainable projection head only.
3. Five-level auxiliary classification (0-2, 3-4, 5-6, 7-8, 9-10).
4. Weighted Ridge regression on the learned representation for the final
   continuous 0-10 mark.
5. Strict fold leakage control:
   every pair containing a held-out row is removed before fitting that fold.
6. Two models trained under identical folds:
   V3_CE      = auxiliary score classification only
   V3_DIFFCL  = classification + InfoNCE + hierarchical loss

The previous V2H blind benchmark is NOT loaded or used here.
It has already been observed and must no longer be treated as a blind test.

Inputs
------
data/v3_contrastive_examples.csv
data/v3_infonce_pairs.csv
data/v3_hierarchical_pairs.csv
data/rubric_concepts_v2c.json

Outputs
-------
models/artifacts/research_grader_v3_ce.joblib
models/artifacts/research_grader_v3_diffcl.joblib
data/research_grader_v3_human_oof.csv
data/research_grader_v3_weak_oof.csv
data/research_grader_v3_summary.csv
data/research_grader_v3_geometry.csv

No V1/V2 artifact is modified.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]

EXAMPLES_PATH = ROOT / "data" / "v3_contrastive_examples.csv"
INFONCE_PATH = ROOT / "data" / "v3_infonce_pairs.csv"
HIER_PATH = ROOT / "data" / "v3_hierarchical_pairs.csv"
RUBRIC_PATH = ROOT / "data" / "rubric_concepts_v2c.json"

HUMAN_PATH = ROOT / "data" / "research_grader_v3_human_oof.csv"
WEAK_PATH = ROOT / "data" / "research_grader_v3_weak_oof.csv"
SUMMARY_PATH = ROOT / "data" / "research_grader_v3_summary.csv"
GEOMETRY_PATH = ROOT / "data" / "research_grader_v3_geometry.csv"

CE_ARTIFACT = (
    ROOT / "models" / "artifacts" / "research_grader_v3_ce.joblib"
)
DIFFCL_ARTIFACT = (
    ROOT / "models" / "artifacts" / "research_grader_v3_diffcl.joblib"
)

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RANDOM_STATE = 42

PROJECTION_DIM = 48
DROPOUT = 0.08
EPOCHS = 220
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-2

TEMPERATURE = 0.10
CONTRASTIVE_LAMBDA = 0.30

RIDGE_ALPHA = 6.0
HUMAN_SAMPLE_WEIGHT = 3.0

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    examples = pd.read_csv(EXAMPLES_PATH)
    infonce = pd.read_csv(INFONCE_PATH)
    hierarchical = pd.read_csv(HIER_PATH)

    with RUBRIC_PATH.open("r", encoding="utf-8") as f:
        rubric_concepts = json.load(f)

    examples["_row_id"] = pd.to_numeric(
        examples["_row_id"],
        errors="raise",
    ).astype(int)

    examples["human_score"] = pd.to_numeric(
        examples["human_score"],
        errors="raise",
    ).astype(float)

    examples["score_level_5"] = pd.to_numeric(
        examples["score_level_5"],
        errors="raise",
    ).astype(int)

    return examples, infonce, hierarchical, rubric_concepts


def rubric_text(qid: str, rubric_concepts: dict) -> str:
    criteria = rubric_concepts[str(qid)]["criteria"]
    return " ".join(str(x).strip() for x in criteria if str(x).strip())


# ---------------------------------------------------------------------------
# Frozen semantic relation features
# ---------------------------------------------------------------------------

@dataclass
class FrozenFeaturePack:
    matrix: np.ndarray
    row_ids: np.ndarray
    input_dim: int


def encode_unique(model, texts):
    return model.encode(
        list(texts),
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def build_frozen_features(
    examples: pd.DataFrame,
    rubric_concepts: dict,
) -> FrozenFeaturePack:
    print(f"Loading frozen encoder: {ENCODER_NAME}")
    encoder = SentenceTransformer(
        ENCODER_NAME,
        device="cpu",
    )

    students = examples["student_answer"].fillna("").astype(str).tolist()
    references = examples["model_answer"].fillna("").astype(str).tolist()
    rubrics = [
        rubric_text(qid, rubric_concepts)
        for qid in examples["question_id"].astype(str)
    ]

    print("Encoding student answers...")
    student_vec = encode_unique(encoder, students)

    print("Encoding reference answers...")
    reference_vec = encode_unique(encoder, references)

    print("Encoding atomic rubrics...")
    rubric_vec = encode_unique(encoder, rubrics)

    # Semantic relation representation:
    # student content
    # absolute student-reference difference
    # element-wise student-reference agreement
    # absolute student-rubric difference
    semantic = np.concatenate(
        [
            student_vec,
            np.abs(student_vec - reference_vec),
            student_vec * reference_vec,
            np.abs(student_vec - rubric_vec),
        ],
        axis=1,
    ).astype(np.float32)

    scalars = np.stack(
        [
            examples["jaccard_similarity"].to_numpy(dtype=np.float32),
            examples["edit_similarity"].to_numpy(dtype=np.float32),
            examples["diffcl_similarity"].to_numpy(dtype=np.float32),
            (
                examples["difference_level"].to_numpy(dtype=np.float32)
                / 4.0
            ),
            examples["negation_mismatch"].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )

    matrix = np.concatenate(
        [semantic, scalars],
        axis=1,
    ).astype(np.float32)

    return FrozenFeaturePack(
        matrix=matrix,
        row_ids=examples["_row_id"].to_numpy(dtype=int),
        input_dim=int(matrix.shape[1]),
    )


# ---------------------------------------------------------------------------
# Projection model
# ---------------------------------------------------------------------------

class ProjectionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, PROJECTION_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.LayerNorm(PROJECTION_DIM),
        )

        self.classifier = nn.Linear(
            PROJECTION_DIM,
            5,
        )

    def forward(self, x):
        z = self.projection(x)
        logits = self.classifier(z)
        return z, logits


def source_weights(train_df: pd.DataFrame) -> np.ndarray:
    weights = np.ones(len(train_df), dtype=np.float32)

    human_mask = (
        train_df["source"].astype(str).eq("human_group").to_numpy()
    )

    weights[human_mask] = HUMAN_SAMPLE_WEIGHT

    # Mild score-frequency balancing.
    counts = (
        train_df["score_level_5"]
        .value_counts()
        .to_dict()
    )

    balance = np.asarray(
        [
            1.0 / math.sqrt(max(1, counts[int(level)]))
            for level in train_df["score_level_5"]
        ],
        dtype=np.float32,
    )

    balance /= max(balance.mean(), 1e-8)

    return weights * balance


def filter_pairs(
    pair_df: pd.DataFrame,
    train_ids: set[int],
    left_col: str,
    right_col: str,
) -> pd.DataFrame:
    return pair_df[
        pair_df[left_col].astype(int).isin(train_ids)
        & pair_df[right_col].astype(int).isin(train_ids)
    ].copy()


def build_positive_weight_matrix(
    train_ids: list[int],
    train_infonce: pd.DataFrame,
) -> torch.Tensor:
    n = len(train_ids)

    id_to_pos = {
        row_id: i
        for i, row_id in enumerate(train_ids)
    }

    matrix = np.zeros(
        (n, n),
        dtype=np.float32,
    )

    for row in train_infonce.itertuples(index=False):
        a = int(row.anchor_row_id)
        b = int(row.positive_row_id)

        if a not in id_to_pos or b not in id_to_pos:
            continue

        matrix[
            id_to_pos[a],
            id_to_pos[b],
        ] = max(
            matrix[id_to_pos[a], id_to_pos[b]],
            float(row.pair_weight),
        )

    return torch.tensor(
        matrix,
        dtype=torch.float32,
        device=DEVICE,
    )


def build_hierarchical_tensors(
    train_ids: list[int],
    train_hier: pd.DataFrame,
):
    id_to_pos = {
        row_id: i
        for i, row_id in enumerate(train_ids)
    }

    left = []
    right = []
    close = []
    pair_weight = []

    for row in train_hier.itertuples(index=False):
        a = int(row.row_id_a)
        b = int(row.row_id_b)

        if a not in id_to_pos or b not in id_to_pos:
            continue

        left.append(id_to_pos[a])
        right.append(id_to_pos[b])
        close.append(
            1.0 if str(row.relation) == "close" else 0.0
        )

        # Slightly emphasize a relation involving real human data,
        # without allowing the 20 human rows to dominate.
        pair_weight.append(
            1.25 if int(row.contains_human) else 1.0
        )

    return (
        torch.tensor(left, dtype=torch.long, device=DEVICE),
        torch.tensor(right, dtype=torch.long, device=DEVICE),
        torch.tensor(close, dtype=torch.float32, device=DEVICE),
        torch.tensor(pair_weight, dtype=torch.float32, device=DEVICE),
    )


def weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    raw = F.cross_entropy(
        logits,
        labels,
        reduction="none",
    )

    return (
        raw * sample_weight
    ).sum() / sample_weight.sum().clamp_min(1e-8)


def infonce_loss(
    z: torch.Tensor,
    positive_weights: torch.Tensor,
) -> torch.Tensor:
    """Supervised InfoNCE with explicit positive pairs from the pair file."""

    z = F.normalize(z, dim=1)

    similarities = (
        z @ z.T
    ) / TEMPERATURE

    n = similarities.shape[0]

    eye = torch.eye(
        n,
        dtype=torch.bool,
        device=similarities.device,
    )

    # Exclude self from denominator.
    denom_logits = similarities.masked_fill(
        eye,
        float("-inf"),
    )

    valid_anchor = positive_weights.sum(dim=1) > 0

    if not torch.any(valid_anchor):
        return torch.tensor(
            0.0,
            device=z.device,
        )

    log_denom = torch.logsumexp(
        denom_logits,
        dim=1,
    )

    # Weighted positive numerator:
    # log(sum_j weight_ij * exp(sim_ij / tau))
    positive_logits = similarities + torch.log(
        positive_weights.clamp_min(1e-12)
    )

    positive_logits = positive_logits.masked_fill(
        positive_weights <= 0,
        float("-inf"),
    )

    log_num = torch.logsumexp(
        positive_logits,
        dim=1,
    )

    loss = -(
        log_num[valid_anchor]
        - log_denom[valid_anchor]
    )

    return loss.mean()


def hierarchical_loss(
    z: torch.Tensor,
    left_idx: torch.Tensor,
    right_idx: torch.Tensor,
    close_mask: torch.Tensor,
    pair_weight: torch.Tensor,
) -> torch.Tensor:
    if left_idx.numel() == 0:
        return torch.tensor(
            0.0,
            device=z.device,
        )

    z = F.normalize(z, dim=1)

    cosine = (
        z[left_idx]
        * z[right_idx]
    ).sum(dim=1)

    close_loss = 1.0 - cosine

    far_loss = torch.relu(
        cosine - 0.5
    )

    raw = (
        close_mask * close_loss
        + (1.0 - close_mask) * far_loss
    )

    return (
        raw * pair_weight
    ).sum() / pair_weight.sum().clamp_min(1e-8)


def train_projection(
    X_train: np.ndarray,
    train_df: pd.DataFrame,
    train_infonce: pd.DataFrame,
    train_hier: pd.DataFrame,
    input_dim: int,
    contrastive_lambda: float,
    seed: int,
):
    set_seed(seed)

    model = ProjectionModel(
        input_dim=input_dim,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    x = torch.tensor(
        X_train,
        dtype=torch.float32,
        device=DEVICE,
    )

    y_level = torch.tensor(
        train_df["score_level_5"].to_numpy(dtype=np.int64),
        dtype=torch.long,
        device=DEVICE,
    )

    sample_weight_np = source_weights(
        train_df
    )

    sample_weight = torch.tensor(
        sample_weight_np,
        dtype=torch.float32,
        device=DEVICE,
    )

    train_ids = (
        train_df["_row_id"]
        .astype(int)
        .tolist()
    )

    positive_weights = build_positive_weight_matrix(
        train_ids,
        train_infonce,
    )

    (
        hier_left,
        hier_right,
        hier_close,
        hier_weight,
    ) = build_hierarchical_tensors(
        train_ids,
        train_hier,
    )

    last = {}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        z, logits = model(x)

        ce = weighted_ce_loss(
            logits,
            y_level,
            sample_weight,
        )

        if contrastive_lambda > 0:
            nce = infonce_loss(
                z,
                positive_weights,
            )

            dist = hierarchical_loss(
                z,
                hier_left,
                hier_right,
                hier_close,
                hier_weight,
            )
        else:
            nce = torch.tensor(
                0.0,
                device=DEVICE,
            )
            dist = torch.tensor(
                0.0,
                device=DEVICE,
            )

        loss = ce + contrastive_lambda * (
            nce + dist
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=2.0,
        )

        optimizer.step()

        if epoch in {1, 50, 100, 150, 200, EPOCHS}:
            last = {
                "epoch": epoch,
                "total": float(loss.detach().cpu()),
                "ce": float(ce.detach().cpu()),
                "nce": float(nce.detach().cpu()),
                "dist": float(dist.detach().cpu()),
            }

    model.eval()

    return model, last


def project_numpy(
    model: ProjectionModel,
    X: np.ndarray,
) -> np.ndarray:
    model.eval()

    with torch.no_grad():
        x = torch.tensor(
            X,
            dtype=torch.float32,
            device=DEVICE,
        )

        z, _ = model(x)

        z = F.normalize(
            z,
            dim=1,
        )

    return z.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Continuous 0-10 regression
# ---------------------------------------------------------------------------

SCALAR_COLUMNS = [
    "jaccard_similarity",
    "edit_similarity",
    "diffcl_similarity",
    "difference_level",
    "negation_mismatch",
]


def ridge_features(
    z: np.ndarray,
    frame: pd.DataFrame,
) -> np.ndarray:
    scalars = frame[
        SCALAR_COLUMNS
    ].to_numpy(dtype=np.float32).copy()

    scalars[:, 3] /= 4.0

    return np.concatenate(
        [z, scalars],
        axis=1,
    ).astype(np.float32)


def fit_ridge(
    z_train: np.ndarray,
    train_df: pd.DataFrame,
) -> Ridge:
    X = ridge_features(
        z_train,
        train_df,
    )

    y = train_df[
        "human_score"
    ].to_numpy(dtype=float)

    weights = np.ones(
        len(train_df),
        dtype=float,
    )

    weights[
        train_df["source"]
        .astype(str)
        .eq("human_group")
        .to_numpy()
    ] = HUMAN_SAMPLE_WEIGHT

    model = Ridge(
        alpha=RIDGE_ALPHA,
    )

    model.fit(
        X,
        y,
        sample_weight=weights,
    )

    return model


def predict_ridge(
    ridge: Ridge,
    z: np.ndarray,
    frame: pd.DataFrame,
) -> np.ndarray:
    pred = ridge.predict(
        ridge_features(
            z,
            frame,
        )
    )

    return np.clip(
        pred,
        0.0,
        10.0,
    )


# ---------------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------------

def metric_dict(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            math.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        )
        if len(np.unique(y_true)) >= 2
        else float("nan"),
        "within_1_mark": float(
            np.mean(
                np.abs(
                    y_true - y_pred
                ) <= 1.0
            )
        ),
        "within_2_marks": float(
            np.mean(
                np.abs(
                    y_true - y_pred
                ) <= 2.0
            )
        ),
    }


def print_metrics(title, y_true, y_pred):
    m = metric_dict(
        y_true,
        y_pred,
    )

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


def row_index_lookup(
    examples: pd.DataFrame,
):
    return {
        int(row_id): i
        for i, row_id in enumerate(
            examples["_row_id"].astype(int)
        )
    }


def fit_and_predict_fold(
    variant_name: str,
    contrastive_lambda: float,
    fold_no: int,
    heldout_ids: set[int],
    test_df: pd.DataFrame,
    examples: pd.DataFrame,
    frozen: FrozenFeaturePack,
    infonce: pd.DataFrame,
    hierarchical: pd.DataFrame,
):
    train_df = examples[
        ~examples["_row_id"]
        .astype(int)
        .isin(heldout_ids)
    ].copy()

    train_ids = set(
        train_df["_row_id"]
        .astype(int)
        .tolist()
    )

    train_infonce = filter_pairs(
        infonce,
        train_ids,
        "anchor_row_id",
        "positive_row_id",
    )

    train_hier = filter_pairs(
        hierarchical,
        train_ids,
        "row_id_a",
        "row_id_b",
    )

    lookup = row_index_lookup(
        examples
    )

    train_positions = [
        lookup[int(row_id)]
        for row_id in train_df["_row_id"]
    ]

    test_positions = [
        lookup[int(row_id)]
        for row_id in test_df["_row_id"]
    ]

    X_train = frozen.matrix[
        train_positions
    ]

    X_test = frozen.matrix[
        test_positions
    ]

    model, losses = train_projection(
        X_train=X_train,
        train_df=train_df,
        train_infonce=train_infonce,
        train_hier=train_hier,
        input_dim=frozen.input_dim,
        contrastive_lambda=contrastive_lambda,
        seed=(
            RANDOM_STATE
            + fold_no * 100
            + (1 if contrastive_lambda > 0 else 0)
        ),
    )

    z_train = project_numpy(
        model,
        X_train,
    )

    z_test = project_numpy(
        model,
        X_test,
    )

    ridge = fit_ridge(
        z_train,
        train_df,
    )

    pred = predict_ridge(
        ridge,
        z_test,
        test_df,
    )

    records = []

    # Use iterrows() here because pandas.itertuples() renames columns that
    # begin with an underscore (such as "_row_id"), making row._row_id
    # unavailable.
    for (_, row), prediction in zip(
        test_df.iterrows(),
        pred,
    ):
        records.append({
            "variant": variant_name,
            "fold": fold_no,
            "_row_id": int(row["_row_id"]),
            "question_id": row["question_id"],
            "source": row["source"],
            "human_score": float(row["human_score"]),
            "prediction": float(prediction),
            "absolute_error": abs(
                float(row["human_score"])
                - float(prediction)
            ),
        })

    return (
        pd.DataFrame(records),
        model,
        ridge,
        losses,
        len(train_infonce),
        len(train_hier),
    )


def evaluate_folds(
    evaluation_name,
    folds,
    examples,
    frozen,
    infonce,
    hierarchical,
):
    all_records = []

    for fold_no, (
        heldout_ids,
        test_df,
    ) in enumerate(
        folds,
        start=1,
    ):
        for variant_name, lam in [
            ("V3_CE", 0.0),
            ("V3_DIFFCL", CONTRASTIVE_LAMBDA),
        ]:
            (
                pred_df,
                _,
                _,
                losses,
                n_info,
                n_hier,
            ) = fit_and_predict_fold(
                variant_name=variant_name,
                contrastive_lambda=lam,
                fold_no=fold_no,
                heldout_ids=heldout_ids,
                test_df=test_df,
                examples=examples,
                frozen=frozen,
                infonce=infonce,
                hierarchical=hierarchical,
            )

            pred_df["evaluation"] = evaluation_name

            all_records.append(
                pred_df
            )

            print(
                f"{evaluation_name} fold {fold_no}/{len(folds)} "
                f"{variant_name}: heldout={len(test_df)}, "
                f"InfoNCE pairs={n_info}, hierarchical={n_hier}, "
                f"last loss={losses.get('total', float('nan')):.3f}"
            )

    return pd.concat(
        all_records,
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Representation geometry diagnostic
# ---------------------------------------------------------------------------

def geometry_summary(
    z: np.ndarray,
    examples: pd.DataFrame,
    label: str,
):
    z = z / np.clip(
        np.linalg.norm(
            z,
            axis=1,
            keepdims=True,
        ),
        1e-8,
        None,
    )

    same_score = []
    close_score = []
    far_score = []

    scores = examples[
        "human_score"
    ].to_numpy(dtype=float)

    qids = examples[
        "question_id"
    ].astype(str).to_numpy()

    for i in range(len(examples)):
        for j in range(i + 1, len(examples)):
            # Geometry is assessed within question so topic identity
            # cannot masquerade as score structure.
            if qids[i] != qids[j]:
                continue

            sim = float(
                np.dot(
                    z[i],
                    z[j],
                )
            )

            gap = abs(
                scores[i] - scores[j]
            )

            if gap == 0:
                same_score.append(sim)

            if gap <= 1:
                close_score.append(sim)
            else:
                far_score.append(sim)

    return {
        "variant": label,
        "same_score_mean_cosine": float(
            np.mean(same_score)
        ),
        "close_gap_le_1_mean_cosine": float(
            np.mean(close_score)
        ),
        "far_gap_gt_1_mean_cosine": float(
            np.mean(far_score)
        ),
        "close_minus_far": float(
            np.mean(close_score)
            - np.mean(far_score)
        ),
    }


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def numpy_state_dict(model: ProjectionModel):
    return {
        key: value.detach().cpu().numpy()
        for key, value in model.state_dict().items()
    }


def train_final_variant(
    variant_name,
    contrastive_lambda,
    examples,
    frozen,
    infonce,
    hierarchical,
):
    train_ids = set(
        examples["_row_id"]
        .astype(int)
        .tolist()
    )

    train_infonce = filter_pairs(
        infonce,
        train_ids,
        "anchor_row_id",
        "positive_row_id",
    )

    train_hier = filter_pairs(
        hierarchical,
        train_ids,
        "row_id_a",
        "row_id_b",
    )

    model, losses = train_projection(
        X_train=frozen.matrix,
        train_df=examples,
        train_infonce=train_infonce,
        train_hier=train_hier,
        input_dim=frozen.input_dim,
        contrastive_lambda=contrastive_lambda,
        seed=(
            RANDOM_STATE
            + 999
            + (1 if contrastive_lambda > 0 else 0)
        ),
    )

    z = project_numpy(
        model,
        frozen.matrix,
    )

    ridge = fit_ridge(
        z,
        examples,
    )

    return model, ridge, z, losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("RESEARCH GRADER V3 — LIGHTWEIGHT DIFFERENCE-AWARE CONTRASTIVE LEARNING")
    print("=" * 80)

    examples, infonce, hierarchical, rubric_concepts = load_data()

    human = examples[
        examples["source"]
        .astype(str)
        .eq("human_group")
    ]

    weak = examples[
        examples["human_score"] <= 4.0
    ]

    print(f"Examples:             {len(examples)}")
    print(f"Human gold:           {len(human)}")
    print(f"Weak <=4:             {len(weak)}")
    print(f"InfoNCE pairs:         {len(infonce)}")
    print(f"Hierarchical pairs:    {len(hierarchical)}")
    print(f"Projection dimension:  {PROJECTION_DIM}")
    print(f"Contrastive lambda:    {CONTRASTIVE_LAMBDA}")
    print(f"Frozen encoder:        {ENCODER_NAME}")
    print("Blind V2H attack file: NOT USED")

    print("\nDifference marker distribution:")
    print(
        examples["difference_marker"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    if (
        examples["difference_marker"]
        .value_counts(normalize=True)
        .max()
        > 0.60
    ):
        print(
            "\nNOTE: difference markers are strongly imbalanced; "
            "they are treated only as auxiliary input features."
        )

    frozen = build_frozen_features(
        examples,
        rubric_concepts,
    )

    print(
        f"\nFrozen relation feature dimension: "
        f"{frozen.input_dim}"
    )

    # HUMAN OOF: one human response/question/fold.
    human_fold_series = pd.Series(
        -1,
        index=examples.index,
        dtype=int,
    )

    counts = human.groupby(
        "question_id"
    ).size()

    if not (counts == 4).all():
        raise RuntimeError(
            f"Expected four human rows/question; found {counts.to_dict()}"
        )

    for _, group in human.groupby(
        "question_id",
        sort=True,
    ):
        for fold_id, idx in enumerate(
            group.index
        ):
            human_fold_series.loc[
                idx
            ] = fold_id

    human_folds = []

    for fold_id in range(4):
        test_df = examples[
            human_fold_series.eq(
                fold_id
            )
        ].copy()

        heldout_ids = set(
            test_df["_row_id"]
            .astype(int)
            .tolist()
        )

        human_folds.append(
            (
                heldout_ids,
                test_df,
            )
        )

    print("\nTraining HUMAN_OOF...")
    human_oof = evaluate_folds(
        evaluation_name="HUMAN_OOF",
        folds=human_folds,
        examples=examples,
        frozen=frozen,
        infonce=infonce,
        hierarchical=hierarchical,
    )

    # WEAK OOF.
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

        test_df = examples.loc[
            test_indices
        ].copy()

        heldout_ids = set(
            test_df["_row_id"]
            .astype(int)
            .tolist()
        )

        weak_folds.append(
            (
                heldout_ids,
                test_df,
            )
        )

    print("\nTraining WEAK_OOF...")
    weak_oof = evaluate_folds(
        evaluation_name="WEAK_OOF",
        folds=weak_folds,
        examples=examples,
        frozen=frozen,
        infonce=infonce,
        hierarchical=hierarchical,
    )

    print("\n" + "=" * 80)
    print("OOF RESULTS")
    print("=" * 80)

    summary_records = []

    for evaluation_name, frame in [
        ("HUMAN_OOF", human_oof),
        ("WEAK_OOF", weak_oof),
    ]:
        for variant in [
            "V3_CE",
            "V3_DIFFCL",
        ]:
            part = frame[
                frame["variant"].eq(
                    variant
                )
            ]

            metrics = print_metrics(
                f"{evaluation_name} — {variant}",
                part["human_score"],
                part["prediction"],
            )

            summary_records.append({
                "evaluation": evaluation_name,
                "variant": variant,
                **metrics,
            })

    print("\nHuman OOF MAE per question:")
    human_per_q = (
        human_oof.assign(
            abs_error=lambda x: np.abs(
                x["human_score"]
                - x["prediction"]
            )
        )
        .groupby(
            ["variant", "question_id"]
        )["abs_error"]
        .mean()
        .unstack(0)
        .round(3)
    )

    print(
        human_per_q.to_string()
    )

    # Final models on all repaired data.
    print("\nTraining final V3_CE...")
    (
        ce_model,
        ce_ridge,
        ce_z,
        ce_losses,
    ) = train_final_variant(
        variant_name="V3_CE",
        contrastive_lambda=0.0,
        examples=examples,
        frozen=frozen,
        infonce=infonce,
        hierarchical=hierarchical,
    )

    print("\nTraining final V3_DIFFCL...")
    (
        diff_model,
        diff_ridge,
        diff_z,
        diff_losses,
    ) = train_final_variant(
        variant_name="V3_DIFFCL",
        contrastive_lambda=CONTRASTIVE_LAMBDA,
        examples=examples,
        frozen=frozen,
        infonce=infonce,
        hierarchical=hierarchical,
    )

    geometry = pd.DataFrame(
        [
            geometry_summary(
                ce_z,
                examples,
                "V3_CE",
            ),
            geometry_summary(
                diff_z,
                examples,
                "V3_DIFFCL",
            ),
        ]
    )

    print("\n## Representation geometry")
    print(
        geometry.round(3).to_string(
            index=False
        )
    )

    common_metadata = {
        "architecture": (
            "frozen MiniLM semantic relation features -> "
            "small projection -> weighted Ridge 0-10"
        ),
        "encoder_name": ENCODER_NAME,
        "input_dim": frozen.input_dim,
        "projection_dim": PROJECTION_DIM,
        "ridge_alpha": RIDGE_ALPHA,
        "human_sample_weight": HUMAN_SAMPLE_WEIGHT,
        "feature_definition": [
            "student_embedding",
            "abs(student-reference)",
            "student*reference",
            "abs(student-rubric)",
            "jaccard_similarity",
            "edit_similarity",
            "diffcl_similarity",
            "difference_level/4",
            "negation_mismatch",
        ],
        "rubric_concepts": rubric_concepts,
        "training": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "temperature": TEMPERATURE,
        },
        "leakage_control": (
            "all contrastive pairs containing held-out rows "
            "removed before every OOF fold"
        ),
        "blind_v2h_benchmark_used": False,
    }

    CE_ARTIFACT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "version": "research_grader_v3_ce",
            **common_metadata,
            "contrastive_lambda": 0.0,
            "projection_state":
                numpy_state_dict(ce_model),
            "ridge": ce_ridge,
            "final_losses": ce_losses,
        },
        CE_ARTIFACT,
    )

    joblib.dump(
        {
            "version": "research_grader_v3_diffcl",
            **common_metadata,
            "contrastive_lambda":
                CONTRASTIVE_LAMBDA,
            "projection_state":
                numpy_state_dict(diff_model),
            "ridge": diff_ridge,
            "final_losses": diff_losses,
        },
        DIFFCL_ARTIFACT,
    )

    human_oof.to_csv(
        HUMAN_PATH,
        index=False,
    )

    weak_oof.to_csv(
        WEAK_PATH,
        index=False,
    )

    summary = pd.DataFrame(
        summary_records
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    geometry.to_csv(
        GEOMETRY_PATH,
        index=False,
    )

    print("\n## Side-by-side summary")
    print(
        summary.round(3).to_string(
            index=False
        )
    )

    print("\n## Saved")
    print(f"V3 CE artifact:     {CE_ARTIFACT}")
    print(f"V3 DiffCL artifact: {DIFFCL_ARTIFACT}")
    print(f"Human OOF:          {HUMAN_PATH}")
    print(f"Weak OOF:           {WEAK_PATH}")
    print(f"Summary:            {SUMMARY_PATH}")
    print(f"Geometry:           {GEOMETRY_PATH}")

    print(
        "\nIMPORTANT: do not run the old V2H blind benchmark "
        "as a 'blind' V3 test. Its cases are already known."
    )

    print(
        "Previous models and datasets were not modified."
    )


if __name__ == "__main__":
    main()
