"""Diagnose the final V2G errors.

Reads existing V2G CSV outputs only.
No training and no model loading.

Shows:
1. Clean human answers where V2G activated.
2. Human answers harmed most vs V2B.
3. Remaining weak padding failures > +0.5.
4. Remaining weak contradiction failures > 0.

Inputs
------
data/research_grader_v2g_human_oof.csv
data/research_grader_v2g_weak_oof.csv
data/research_grader_v2g_adversarial.csv
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

HUMAN = ROOT / "data" / "research_grader_v2g_human_oof.csv"
WEAK = ROOT / "data" / "research_grader_v2g_weak_oof.csv"
ADV = ROOT / "data" / "research_grader_v2g_adversarial.csv"


def paired(df):
    b = (
        df[df["model"] == "V2B"]
        [["_row_id", "question_id", "human_score", "prediction"]]
        .rename(columns={"prediction": "v2b_prediction"})
    )

    g = (
        df[df["model"] == "V2G"]
        [
            [
                "_row_id",
                "prediction",
                "gate_active",
                "padding_gate",
                "contradiction_gate",
                "evidence_word_ratio",
                "strict_supported_count",
                "contradicted_concepts",
                "contradiction_penalty",
                "strict_evidence_text",
                "student_answer",
            ]
        ]
        .rename(columns={"prediction": "v2g_prediction"})
    )

    out = b.merge(g, on="_row_id")

    out["v2g_minus_v2b"] = (
        out["v2g_prediction"] - out["v2b_prediction"]
    )

    out["v2b_error"] = (
        out["human_score"] - out["v2b_prediction"]
    ).abs()

    out["v2g_error"] = (
        out["human_score"] - out["v2g_prediction"]
    ).abs()

    out["error_change"] = (
        out["v2g_error"] - out["v2b_error"]
    )

    return out


def print_clean_case(row):
    print("-" * 78)
    print(f"row_id:                {row['_row_id']}")
    print(f"question:              {row['question_id']}")
    print(f"human score:           {row['human_score']:.2f}")
    print(f"V2B prediction:        {row['v2b_prediction']:.3f}")
    print(f"V2G prediction:        {row['v2g_prediction']:.3f}")
    print(f"V2G - V2B:            {row['v2g_minus_v2b']:+.3f}")
    print(f"error change:          {row['error_change']:+.3f}")
    print(f"gate active:           {row['gate_active']}")
    print(f"padding gate:          {row['padding_gate']}")
    print(f"contradiction gate:    {row['contradiction_gate']}")
    print(f"evidence word ratio:   {row['evidence_word_ratio']:.3f}")
    print(f"strict concepts:       {row['strict_supported_count']}/6")
    print(f"contradicted concepts: {row['contradicted_concepts']}")
    print(f"contradiction penalty: {row['contradiction_penalty']:.3f}")
    print(f"\nANSWER:\n{row['student_answer']}")
    print(f"\nSTRICT EVIDENCE:\n{row['strict_evidence_text']}")


def print_attack(row):
    print("-" * 78)
    print(f"row_id:                {row['_row_id']}")
    print(f"question:              {row['question_id']}")
    print(f"human score:           {row['human_score']:.2f}")
    print(f"attack:                {row['attack']}")
    print(f"base prediction:       {row['base_prediction']:.3f}")
    print(f"attacked prediction:   {row['attacked_prediction']:.3f}")
    print(f"score delta:           {row['score_delta']:+.3f}")
    print(f"gate active:           {row['gate_active']}")
    print(f"padding gate:          {row['padding_gate']}")
    print(f"contradiction gate:    {row['contradiction_gate']}")
    print(f"evidence word ratio:   {row['evidence_word_ratio']:.3f}")
    print(f"strict concepts:       {row['strict_supported_count']}/6")
    print(f"contradicted concepts: {row['contradicted_concepts']}")
    print(f"contradiction penalty: {row['contradiction_penalty']:.3f}")
    print(f"\nSTRICT EVIDENCE:\n{row['strict_evidence_text']}")


def main():
    human = pd.read_csv(HUMAN)
    weak = pd.read_csv(WEAK)
    adv = pd.read_csv(ADV)

    human_pair = paired(human)

    print("=" * 78)
    print("V2G FINAL-FAILURE DIAGNOSTIC")
    print("=" * 78)

    print("\n## Clean human answers where V2G gate activated")
    gated = human_pair[human_pair["gate_active"] == True]

    if gated.empty:
        print("None.")
    else:
        for _, row in gated.sort_values(
            "error_change",
            ascending=False,
        ).iterrows():
            print_clean_case(row)

    print("\n## Human answers most harmed vs V2B")
    harmed = human_pair.sort_values(
        "error_change",
        ascending=False,
    ).head(3)

    for _, row in harmed.iterrows():
        print_clean_case(row)

    v2g_adv = adv[
        (adv["model"] == "V2G")
        & (adv["evaluation"] == "WEAK_OOF")
    ].copy()

    print("\n## Remaining weak padding failures > +0.5")
    padding_failures = v2g_adv[
        (v2g_adv["attack"] != "contradiction_injection")
        & (v2g_adv["score_delta"] > 0.5)
    ].sort_values(
        "score_delta",
        ascending=False,
    )

    if padding_failures.empty:
        print("None.")
    else:
        for _, row in padding_failures.iterrows():
            print_attack(row)

    print("\n## Remaining contradiction increases > 0")
    contradiction_failures = v2g_adv[
        (v2g_adv["attack"] == "contradiction_injection")
        & (v2g_adv["score_delta"] > 0.0)
    ].sort_values(
        "score_delta",
        ascending=False,
    )

    if contradiction_failures.empty:
        print("None.")
    else:
        for _, row in contradiction_failures.iterrows():
            print_attack(row)

    print("\n## Failure counts")
    print(
        v2g_adv[
            v2g_adv["score_delta"] > 0.5
        ]
        .groupby("attack")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )


if __name__ == "__main__":
    main()
