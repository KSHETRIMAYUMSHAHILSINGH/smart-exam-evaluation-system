"""Diagnose the remaining V2F failures.

Reads existing V2F outputs only. No training.

Shows:
1. Clean human answers where V2F activated the gate.
2. Human answers most harmed vs V2B.
3. Worst weak padding failures.
4. Worst contradiction failures.

Inputs
------
data/research_grader_v2f_human_oof.csv
data/research_grader_v2f_weak_oof.csv
data/research_grader_v2f_adversarial.csv
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

HUMAN = ROOT / "data" / "research_grader_v2f_human_oof.csv"
WEAK = ROOT / "data" / "research_grader_v2f_weak_oof.csv"
ADV = ROOT / "data" / "research_grader_v2f_adversarial.csv"


def paired_predictions(df):
    base = df[df["model"] == "V2B"].copy()
    guarded = df[df["model"] == "V2F"].copy()

    cols = [
        "_row_id",
        "question_id",
        "human_score",
        "prediction",
    ]

    base = base[cols].rename(
        columns={"prediction": "v2b_prediction"}
    )

    keep = [
        "_row_id",
        "prediction",
        "gate_active",
        "padding_gate",
        "contradiction_gate",
        "evidence_word_ratio",
        "coverage_index",
        "supported_count",
        "contradicted_concepts",
        "contradiction_penalty",
        "evidence_text",
        "student_answer",
    ]

    guarded = guarded[keep].rename(
        columns={"prediction": "v2f_prediction"}
    )

    out = base.merge(
        guarded,
        on="_row_id",
        how="inner",
    )

    out["v2f_minus_v2b"] = (
        out["v2f_prediction"]
        - out["v2b_prediction"]
    )

    out["v2b_abs_error"] = (
        out["human_score"]
        - out["v2b_prediction"]
    ).abs()

    out["v2f_abs_error"] = (
        out["human_score"]
        - out["v2f_prediction"]
    ).abs()

    out["error_change"] = (
        out["v2f_abs_error"]
        - out["v2b_abs_error"]
    )

    return out


def print_case(row):
    print("-" * 78)
    print(f"row_id:               {row['_row_id']}")
    print(f"question:             {row['question_id']}")
    print(f"human score:          {row['human_score']:.2f}")
    print(f"V2B prediction:       {row['v2b_prediction']:.3f}")
    print(f"V2F prediction:       {row['v2f_prediction']:.3f}")
    print(f"V2F - V2B:           {row['v2f_minus_v2b']:+.3f}")
    print(f"error change:         {row['error_change']:+.3f}")
    print(f"gate active:          {row['gate_active']}")
    print(f"padding gate:         {row['padding_gate']}")
    print(f"contradiction gate:   {row['contradiction_gate']}")
    print(f"evidence word ratio:  {row['evidence_word_ratio']:.3f}")
    print(f"coverage index:       {row['coverage_index']:.3f}")
    print(f"supported concepts:   {row['supported_count']}")
    print(f"contradicted concepts:{row['contradicted_concepts']}")
    print(f"contradiction penalty:{row['contradiction_penalty']:.3f}")
    print(f"\nANSWER:\n{row['student_answer']}")
    print(f"\nEVIDENCE:\n{row['evidence_text']}")


def print_attack_case(row):
    print("-" * 78)
    print(f"evaluation:           {row['evaluation']}")
    print(f"row_id:               {row['_row_id']}")
    print(f"question:             {row['question_id']}")
    print(f"human score:          {row['human_score']:.2f}")
    print(f"attack:               {row['attack']}")
    print(f"base prediction:      {row['base_prediction']:.3f}")
    print(f"attacked prediction:  {row['attacked_prediction']:.3f}")
    print(f"score delta:          {row['score_delta']:+.3f}")
    print(f"gate active:          {row['gate_active']}")
    print(f"padding gate:         {row['padding_gate']}")
    print(f"contradiction gate:   {row['contradiction_gate']}")
    print(f"evidence word ratio:  {row['evidence_word_ratio']:.3f}")
    print(f"contradicted concepts:{row['contradicted_concepts']}")
    print(f"contradiction penalty:{row['contradiction_penalty']:.3f}")


def main():
    human = pd.read_csv(HUMAN)
    weak = pd.read_csv(WEAK)
    adv = pd.read_csv(ADV)

    human_pair = paired_predictions(human)
    weak_pair = paired_predictions(weak)

    print("=" * 78)
    print("V2F FAILURE DIAGNOSTIC")
    print("=" * 78)

    print("\n## Clean HUMAN answers where the V2F gate activated")
    gated = human_pair[human_pair["gate_active"] == True].copy()

    if gated.empty:
        print("None.")
    else:
        for _, row in gated.sort_values(
            "error_change",
            ascending=False,
        ).iterrows():
            print_case(row)

    print("\n## Human answers most harmed by V2F vs V2B")
    harmed = human_pair.sort_values(
        "error_change",
        ascending=False,
    ).head(5)

    for _, row in harmed.iterrows():
        print_case(row)

    print("\n## Weak clean answers where gate activated")
    weak_gated = weak_pair[
        weak_pair["gate_active"] == True
    ].sort_values(
        "error_change",
        ascending=False,
    )

    print(
        weak_gated[
            [
                "_row_id",
                "question_id",
                "human_score",
                "v2b_prediction",
                "v2f_prediction",
                "error_change",
                "evidence_word_ratio",
                "supported_count",
                "contradicted_concepts",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    v2f_adv = adv[adv["model"] == "V2F"].copy()

    print("\n## Worst WEAK padding failures")
    weak_padding = v2f_adv[
        (v2f_adv["evaluation"] == "WEAK_OOF")
        & (v2f_adv["attack"] != "contradiction_injection")
    ].sort_values(
        "score_delta",
        ascending=False,
    ).head(12)

    for _, row in weak_padding.iterrows():
        print_attack_case(row)

    print("\n## Worst WEAK contradiction failures")
    weak_contra = v2f_adv[
        (v2f_adv["evaluation"] == "WEAK_OOF")
        & (v2f_adv["attack"] == "contradiction_injection")
    ].sort_values(
        "score_delta",
        ascending=False,
    ).head(12)

    for _, row in weak_contra.iterrows():
        print_attack_case(row)

    print("\n## Failure counts by attack (> +0.5)")
    failures = v2f_adv[
        (v2f_adv["evaluation"] == "WEAK_OOF")
        & (v2f_adv["score_delta"] > 0.5)
    ]

    if failures.empty:
        print("None.")
    else:
        print(
            failures.groupby("attack")
            .size()
            .sort_values(ascending=False)
            .to_string()
        )


if __name__ == "__main__":
    main()
