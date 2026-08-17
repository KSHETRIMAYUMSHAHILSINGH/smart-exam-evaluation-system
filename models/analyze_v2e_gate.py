"""Analyze when the V2E evidence bottleneck should activate.

Reads existing V2E OOF outputs only. No models are loaded and nothing is retrained.

Goal
----
Find a selective gating threshold that:
1. rarely activates on clean human answers,
2. activates frequently on attacked weak answers,
3. especially catches combined_long_weak.

Inputs
------
data/research_grader_v2e_human_oof.csv
data/research_grader_v2e_weak_oof.csv
data/research_grader_v2e_adversarial.csv

Output
------
data/research_grader_v2e_gate_sweep.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

HUMAN = ROOT / "data" / "research_grader_v2e_human_oof.csv"
WEAK = ROOT / "data" / "research_grader_v2e_weak_oof.csv"
ADV = ROOT / "data" / "research_grader_v2e_adversarial.csv"
OUT = ROOT / "data" / "research_grader_v2e_gate_sweep.csv"


def quantiles(series):
    return series.quantile([0, .1, .25, .5, .75, .9, 1]).round(3)


def main():
    human = pd.read_csv(HUMAN)
    weak = pd.read_csv(WEAK)
    adv = pd.read_csv(ADV)

    human = human[human["model"] == "V2E"].copy()
    weak = weak[weak["model"] == "V2E"].copy()
    adv = adv[adv["model"] == "V2E"].copy()

    print("=" * 72)
    print("V2E SELECTIVE-GATE DIAGNOSTIC")
    print("=" * 72)

    print("\n## Clean evidence-word-ratio")
    print("\nHuman clean:")
    print(quantiles(human["evidence_word_ratio"]).to_string())

    print("\nWeak clean:")
    print(quantiles(weak["evidence_word_ratio"]).to_string())

    print("\n## Attacked evidence-word-ratio by attack")
    print(
        adv.groupby(["evaluation", "attack"])["attacked_evidence_word_ratio"]
        .agg(["mean", "median", "min", "max", "count"])
        .round(3)
        .to_string()
    )

    thresholds = np.arange(0.50, 0.981, 0.025)
    records = []

    weak_adv = adv[adv["evaluation"] == "WEAK_OOF"].copy()
    human_adv = adv[adv["evaluation"] == "HUMAN_OOF"].copy()

    combined = weak_adv[weak_adv["attack"] == "combined_long_weak"]
    padding = weak_adv[weak_adv["attack"] != "contradiction_injection"]

    for t in thresholds:
        records.append({
            "threshold": round(float(t), 3),
            "clean_human_trigger_rate":
                float((human["evidence_word_ratio"] < t).mean()),
            "clean_weak_trigger_rate":
                float((weak["evidence_word_ratio"] < t).mean()),
            "weak_padding_attack_trigger_rate":
                float((padding["attacked_evidence_word_ratio"] < t).mean()),
            "weak_combined_attack_trigger_rate":
                float((combined["attacked_evidence_word_ratio"] < t).mean()),
            "human_padding_attack_trigger_rate":
                float((
                    human_adv[
                        human_adv["attack"] != "contradiction_injection"
                    ]["attacked_evidence_word_ratio"] < t
                ).mean()),
        })

    sweep = pd.DataFrame(records)

    # Prefer thresholds that keep clean-human intervention <=10%.
    candidates = sweep[
        sweep["clean_human_trigger_rate"] <= 0.10
    ].copy()

    if not candidates.empty:
        candidates["utility"] = (
            0.60 * candidates["weak_combined_attack_trigger_rate"]
            + 0.40 * candidates["weak_padding_attack_trigger_rate"]
        )
        best = candidates.sort_values(
            ["utility", "threshold"],
            ascending=[False, True],
        ).iloc[0]

        print("\n## Best evidence-ratio gate with <=10% clean-human triggers")
        print(best.round(3).to_string())
    else:
        print("\nNo threshold keeps clean-human trigger rate <=10%.")

    print("\n## Full gate sweep")
    print(sweep.round(3).to_string(index=False))

    print("\n## Contradiction signal")
    contradiction = adv[adv["attack"] == "contradiction_injection"]
    print(
        contradiction.groupby("evaluation")["attacked_max_contradiction"]
        .agg(["mean", "median", "min", "max"])
        .round(3)
        .to_string()
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(OUT, index=False)

    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
