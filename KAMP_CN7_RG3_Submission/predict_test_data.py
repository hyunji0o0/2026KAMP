"""Score mixed CN7/RG3 official test rows and write readable product/combined CSVs.

The two input filenames are used only to restore original product file rows
AFTER the sensor-based product gate and product-specific models have run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source_code"))
from f15_model import score_dataframe  # noqa: E402

CN7_INPUT = ROOT / "test_data" / "CN7_test_data.csv"
RG3_INPUT = ROOT / "test_data" / "RG3_test_data.csv"
MODEL = ROOT / "model" / "KAMP_CN7_RG3_RiskPriority_F15_Threshold.joblib"
OUTPUT_DIR = ROOT / "test_prediction"

FEATURE_NAMES = {
    "Injection_Time", "Filling_Time", "Plasticizing_Time", "Cycle_Time",
    "Clamp_Close_Time", "Cushion_Position", "Plasticizing_Position",
    "Max_Injection_Speed", "Max_Screw_RPM", "Average_Screw_RPM",
    "Max_Injection_Pressure", "Max_Switch_Over_Pressure",
    "Max_Back_Pressure", "Average_Back_Pressure",
    *(f"Barrel_Temperature_{i}" for i in range(1, 7)),
    "Hopper_Temperature", "Mold_Temperature_3", "Mold_Temperature_4",
}
RESULT_COLUMNS = [
    "Predicted_Product", "Predicted_PassOrFail", "Predicted_Quality",
    "Inspection_Rank", "Inspection_Tier", "Model_Score", "Priority_Score",
]
TIER_LABELS = {
    "A_TOP5": "TOP_5_PCT", "B_5_TO_10": "TOP_5_TO_10_PCT",
    "C_10_TO_20": "TOP_10_TO_20_PCT", "D_AFTER_20": "AFTER_TOP_20_PCT",
}


def read_inputs(cn7_path: Path, rg3_path: Path):
    original = {"CN7": pd.read_csv(cn7_path), "RG3": pd.read_csv(rg3_path)}
    if list(original["CN7"].columns) != list(original["RG3"].columns):
        raise ValueError("Two test files must have identical input columns.")
    for product, df in original.items():
        if not len(df) or "PassOrFail" in df:
            raise ValueError(f"{product}: nonempty unlabeled CSV required.")
        if df.columns[0] != "Unnamed: 0":
            raise ValueError(f"{product}: original unnamed record ID column required.")
    return original


def mixed_indices(n_cn7: int, n_rg3: int) -> np.ndarray:
    """Interleave variable-sized blocks, preserving order within each product."""
    rng = np.random.default_rng(20261008)
    groups = [np.arange(n_cn7), n_cn7 + np.arange(n_rg3)]
    offsets = [0, 0]
    mixed: list[int] = []
    current = 0
    while offsets[0] < n_cn7 or offsets[1] < n_rg3:
        total = len(groups[current])
        size = int(rng.integers(35, 181))
        end = min(offsets[current] + size, total)
        mixed.extend(groups[current][offsets[current]:end].tolist())
        offsets[current] = end
        current = 1 - current
    order = np.asarray(mixed, dtype=int)
    if len(order) != n_cn7 + n_rg3 or len(np.unique(order)) != len(order):
        raise AssertionError("Mixed input omitted or duplicated rows.")
    if not np.all(np.diff(order[order < n_cn7]) > 0):
        raise AssertionError("CN7 order changed.")
    if not np.all(np.diff(order[order >= n_cn7]) > 0):
        raise AssertionError("RG3 order changed.")
    return order


def validate_model_output(pred: pd.DataFrame, bundle: dict) -> None:
    """Catch wrong model paths, constant labels and policy/rank inconsistencies."""
    if not pred["Routing_Source"].eq("AUTO_GATE").all():
        raise AssertionError("Product identity was injected instead of inferred by the gate.")
    if pred["Product_Review"].any():
        raise AssertionError("Official test batch unexpectedly needs product review.")
    policy = bundle["decision_policy"]
    if policy["score_is_calibrated_probability"] is not False:
        raise AssertionError("Priority scores must not be presented as probabilities.")
    for product in ("CN7", "RG3"):
        g = pred.loc[pred["Product_Type"].eq(product)]
        n = len(g)
        if not n or set(g["PriorityRank"].astype(int)) != set(range(1, n + 1)):
            raise AssertionError(f"{product}: rank is not a permutation of 1..N.")
        score = pd.to_numeric(g["PriorityScore_0_1"]).to_numpy(float)
        rank = g["PriorityRank"].to_numpy(int)
        np.testing.assert_allclose(score, 1 - (rank - 1) / max(n - 1, 1), atol=1e-12)
        threshold = float(policy["thresholds"][product]["threshold"])
        np.testing.assert_array_equal(g["Predicted_PassFail"].to_numpy(int), (score >= threshold).astype(int))
        if not np.isfinite(score).all() or not np.isfinite(threshold):
            raise AssertionError(f"{product}: nonfinite score or threshold.")
        if g["Predicted_PassFail"].nunique() != 2:
            raise AssertionError(f"{product}: official output collapsed to one Pass/Fail value.")
        for pct in (5, 10, 20):
            flag = g[f"Inspect_Top{pct}"].astype(bool).to_numpy()
            np.testing.assert_array_equal(flag, rank <= max(1, int(np.floor(n * pct / 100))))
        if g["InspectionTier"].isna().any():
            raise AssertionError(f"{product}: missing inspection tier.")


def read_raw_rows(path: Path, expected_rows: int):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    if len(rows) != expected_rows or not header or header[0] != "":
        raise ValueError(f"Unexpected raw CSV layout: {path}")
    return header, rows


def output_values(pred_row) -> list[str]:
    review = bool(pred_row.Product_Review)
    if review:
        return ["REVIEW", "", "PRODUCT_REVIEW_REQUIRED", "", "PRODUCT_REVIEW_REQUIRED", "", ""]
    value = int(pred_row.Predicted_PassFail)
    tier = TIER_LABELS[str(pred_row.InspectionTier)]
    if value == 1 and tier == "AFTER_TOP_20_PCT":
        # RG3 F1.5 flags more than the separate 20% inspection capacity.
        # Keep this disagreement visible instead of implying no review.
        tier = "AFTER_TOP_20_PCT_FAIL_SUSPECT"
    return [
        str(pred_row.Product_Type), str(value), "FAIL_SUSPECT" if value else "PASS_PREDICTED",
        str(int(pred_row.PriorityRank)), tier,
        str(float(pred_row.Native_Risk_Score)),
        str(float(pred_row.PriorityScore_0_1)),
    ]


def write_rows(path: Path, row_positions: np.ndarray, pred: pd.DataFrame,
               raw_cn7: list[list[str]], raw_rg3: list[list[str]],
               n_cn7: int, feature_names: list[str], raw_header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["Record_ID"] + feature_names + RESULT_COLUMNS
    column_indices = [0] + [raw_header.index(f) for f in feature_names]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        written = 0
        for pos in row_positions:
            i = int(pos)
            raw = raw_cn7[i] if i < n_cn7 else raw_rg3[i - n_cn7]
            # Predictions are indexed in original concatenated input order.
            p = pred.iloc[i]
            writer.writerow([raw[j] for j in column_indices] + output_values(p))
            written += 1
    if written != len(row_positions):
        raise AssertionError(f"{path}: incomplete write {written}/{len(row_positions)}.")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        if sum(1 for _ in csv.reader(f)) - 1 != written:
            raise AssertionError(f"{path}: CSV read-back row count mismatch.")
    if len(set(header)) != len(header):
        raise AssertionError("Output columns are not unique.")


def run(cn7_path: Path, rg3_path: Path, model_path: Path, output_dir: Path) -> dict:
    original = read_inputs(cn7_path, rg3_path)
    n_cn7 = len(original["CN7"])
    n_rg3 = len(original["RG3"])
    bundle = joblib.load(model_path)
    feature_names = list(bundle["product_gate"]["features"])
    if (feature_names != list(bundle["cn7_quality"]["base_features"])
            or feature_names != list(bundle["rg3_quality"]["feature_columns"])
            or set(feature_names) != FEATURE_NAMES):
        raise ValueError("Model feature lists do not agree with the readable output schema.")
    raw_header, raw_cn7 = read_raw_rows(cn7_path, n_cn7)
    header_rg3, raw_rg3 = read_raw_rows(rg3_path, n_rg3)
    if raw_header != header_rg3:
        raise ValueError("Raw test CSV columns differ.")
    normalized_header = ["Unnamed: 0" if i == 0 and h == "" else h for i, h in enumerate(raw_header)]
    if normalized_header != list(original["CN7"].columns):
        raise ValueError("Original header changed during CSV parsing.")
    original_join = pd.concat([original["CN7"], original["RG3"]], ignore_index=True)
    order = mixed_indices(n_cn7, n_rg3)
    mixed = original_join.iloc[order].reset_index(drop=True)
    scored_mixed = score_dataframe(mixed, bundle)  # NO Product_Type column: sensor gate runs.
    validate_model_output(scored_mixed, bundle)
    restored = scored_mixed.copy()
    restored.index = order
    restored = restored.sort_index()
    source_product = np.where(order < n_cn7, "CN7", "RG3")
    mismatches = int(np.sum(scored_mixed["Product_Type"].to_numpy() != source_product))
    if mismatches:
        raise AssertionError(f"Product router disagreed with the known official source on {mismatches} rows.")

    files = {
        "CN7": (np.arange(n_cn7), output_dir / "CN7_test_data_predictions.csv"),
        "RG3": (n_cn7 + np.arange(n_rg3), output_dir / "RG3_test_data_predictions.csv"),
        "COMBINED": (order, output_dir / "CN7_RG3_combined_predictions.csv"),
    }
    for positions, path in files.values():
        write_rows(path, positions, restored, raw_cn7, raw_rg3, n_cn7, feature_names, normalized_header)
    summary = {
        "rows": {"CN7": n_cn7, "RG3": n_rg3, "total": n_cn7 + n_rg3},
        "product_route_mismatches": mismatches,
        "passfail_counts": {
            p: {int(k): int(v) for k, v in
                restored.loc[restored.Product_Type.eq(p), "Predicted_PassFail"].value_counts().sort_index().items()}
            for p in ("CN7", "RG3")
        },
        "thresholds": {p: float(bundle["decision_policy"]["thresholds"][p]["threshold"]) for p in ("CN7", "RG3")},
        "output_files": {key: str(path) for key, (_, path) in files.items()},
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cn7-test", type=Path, default=CN7_INPUT)
    parser.add_argument("--rg3-test", type=Path, default=RG3_INPUT)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.cn7_test, args.rg3_test, args.model, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
