"""Attach the submitted frozen F1.5 decision policy to a retrained base bundle.

This is for method reproduction only. Frozen OOF thresholds were selected for the
original models; fresh retraining may change predictions and needs new validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "model/CN7_RG3_trained_model_retrained.joblib")
    parser.add_argument("--reference", type=Path, default=ROOT / "model/KAMP_CN7_RG3_RiskPriority_F15_Threshold.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "model/CN7_RG3_retrained_f15.joblib")
    args = parser.parse_args()
    base, reference = joblib.load(args.base), joblib.load(args.reference)
    for product, key in (("CN7", "cn7_quality"), ("RG3", "rg3_quality")):
        a, b = base[key], reference[key]
        features = a["base_features"] if product == "CN7" else a["feature_columns"]
        reference_features = b["base_features"] if product == "CN7" else b["feature_columns"]
        if list(features) != list(reference_features) or a["model_name"] != b["model_name"]:
            raise ValueError(f"{product} model definition changed; frozen policy cannot be attached.")
    if base["product_gate"]["selected_name"] != reference["product_gate"]["selected_name"]:
        raise ValueError("Product gate model changed.")
    base["decision_policy"] = reference["decision_policy"]
    base["policy_reused_from_reference_not_revalidated"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(base, args.output)
    print(f"saved={args.output}; frozen F1.5 thresholds reused, not revalidated")


if __name__ == "__main__":
    main()
