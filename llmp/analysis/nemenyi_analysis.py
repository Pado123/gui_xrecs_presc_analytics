"""
RQ2 Statistical Analysis: Nemenyi Post-Hoc Test

Compares LLM prediction quality on hashed vs non-hashed prompts for each dataset and KPI,
as described in Section 4.2 of the paper.

For each running trace across all experimental seeds, the per-trace error (AE for lead_time,
binary error for outcome_pred) is collected under both treatments (hashed / non-hashed).
The Nemenyi post-hoc test (implemented via a Wilcoxon signed-rank test for paired samples,
equivalent for two conditions) is then applied to quantify whether the difference is significant.

Usage:
    python nemenyi_analysis.py --output_dir ../output/pm --dataset bpi12 --kpi lead_time
    python nemenyi_analysis.py --output_dir ../output/pm  # runs all datasets and KPIs
"""

import argparse
import json
import glob
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from scipy.stats import wilcoxon

try:
    import scikit_posthocs as sp
    POSTHOCS_AVAILABLE = True
except ImportError:
    POSTHOCS_AVAILABLE = False
    print("Warning: scikit-posthocs not installed. Falling back to scipy.stats.wilcoxon "
          "(equivalent for two-treatment comparison).")


# ── helpers ──────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_per_trace_errors_lead_time(data: dict) -> List[float]:
    """Return a flat list of per-trace absolute errors from a lead_time output JSON."""
    errors = []
    for sample_size_key, sample_data in data.items():
        if sample_size_key == "model_config":
            continue
        for test_seed_data in sample_data.get("test_seeds", {}).values():
            for train_seed_data in test_seed_data.get("train_seeds", {}).values():
                ae_list = train_seed_data.get("metrics", {}).get("AE", [])
                errors.extend([abs(e) for e in ae_list])
    return errors


def collect_per_trace_errors_outcome(data: dict) -> Tuple[List[int], List[int]]:
    """
    Return flat lists of (predictions, true_labels) from an outcome_pred output JSON.
    Binary errors (|pred - true|) are then 0 or 1.
    """
    preds, trues = [], []
    for sample_size_key, sample_data in data.items():
        if sample_size_key == "model_config":
            continue
        for test_seed_data in sample_data.get("test_seeds", {}).values():
            true_values = test_seed_data.get("aggregated_metrics", {}).get("True_values", [])
            for train_seed_data in test_seed_data.get("train_seeds", {}).values():
                seed_preds = train_seed_data.get("predictions", [])
                preds.extend([int(round(p)) for p in seed_preds])
                trues.extend([int(t) for t in true_values])
    return preds, trues


def per_trace_binary_errors(preds: List[int], trues: List[int]) -> List[int]:
    return [abs(p - t) for p, t in zip(preds, trues)]


def _significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ── core analysis ─────────────────────────────────────────────────────────────

def nemenyi_two_treatment(errors_a: List[float], errors_b: List[float]) -> Tuple[float, float]:
    """
    For two treatments (A and B), run a paired Wilcoxon signed-rank test.
    Truncates to the shorter length so the two vectors are aligned.

    Returns (statistic, p_value).
    """
    n = min(len(errors_a), len(errors_b))
    if n == 0:
        raise ValueError("No paired observations found.")
    a = np.array(errors_a[:n])
    b = np.array(errors_b[:n])

    # Wilcoxon signed-rank test (equivalent to Nemenyi for 2 conditions)
    stat, p = wilcoxon(a, b, alternative="two-sided")
    return float(stat), float(p)


def run_analysis(
    output_dir: str,
    dataset: str,
    kpi: str,
    n_samples: int = 100,
) -> Dict:
    """
    Run the Nemenyi (Wilcoxon) analysis for a single dataset / KPI.

    Looks for JSON files in:
        output_dir / dataset / gemini_results_multiple_seeds_n{n_samples}seq_{kpi_tag}*.json
        output_dir / dataset / gemini_results_multiple_seeds_n{n_samples}seq_{kpi_tag}_hashed*.json

    Returns a dict with keys: dataset, kpi, mean_non_hashed, mean_hashed,
    difference_pct (or absolute), p_value, significance.
    """
    base = Path(output_dir) / dataset
    kpi_tag = "outcomepred" if kpi == "outcome_pred" else ""

    # Collect non-hashed files
    if kpi_tag:
        non_hashed_pattern = str(base / f"gemini_results_multiple_seeds_n{n_samples}seq_{kpi_tag}*.json")
    else:
        non_hashed_pattern = str(base / f"gemini_results_multiple_seeds_n{n_samples}seq*.json")

    all_files = glob.glob(non_hashed_pattern)
    non_hashed_files = [f for f in all_files if "hashed" not in os.path.basename(f)]
    hashed_files     = [f for f in all_files if "hashed"     in os.path.basename(f)]

    if not non_hashed_files:
        raise FileNotFoundError(f"No non-hashed files found for {dataset}/{kpi} "
                                f"(pattern: {non_hashed_pattern})")
    if not hashed_files:
        raise FileNotFoundError(f"No hashed files found for {dataset}/{kpi}")

    # Aggregate errors across all seeds/files
    if kpi == "lead_time":
        errors_non_hashed, errors_hashed = [], []
        for f in non_hashed_files:
            errors_non_hashed.extend(collect_per_trace_errors_lead_time(load_json(f)))
        for f in hashed_files:
            errors_hashed.extend(collect_per_trace_errors_lead_time(load_json(f)))

        mean_nh = np.mean(errors_non_hashed)
        mean_h  = np.mean(errors_hashed)
        diff_pct = ((mean_h - mean_nh) / mean_nh) * 100
        stat, p = nemenyi_two_treatment(errors_non_hashed, errors_hashed)

        return {
            "dataset": dataset,
            "kpi": "Total Time (MAE)",
            f"LLM (mean AE)": f"{mean_nh:.0f}",
            f"LLM anonymized (mean AE)": f"{mean_h:.0f}",
            "Difference": f"{diff_pct:.2f}%",
            "P-value": f"{p:.3f}",
            "Significance": _significance_stars(p),
        }

    else:  # outcome_pred
        preds_nh, trues_nh = [], []
        preds_h,  trues_h  = [], []
        for f in non_hashed_files:
            p_list, t_list = collect_per_trace_errors_outcome(load_json(f))
            preds_nh.extend(p_list); trues_nh.extend(t_list)
        for f in hashed_files:
            p_list, t_list = collect_per_trace_errors_outcome(load_json(f))
            preds_h.extend(p_list); trues_h.extend(t_list)

        errors_nh = per_trace_binary_errors(preds_nh, trues_nh)
        errors_h  = per_trace_binary_errors(preds_h,  trues_h)

        # F1 difference (non_hashed - hashed; higher F1 is better)
        from sklearn.metrics import f1_score
        f1_nh = f1_score(trues_nh, preds_nh, average="weighted", zero_division=0)
        f1_h  = f1_score(trues_h,  preds_h,  average="weighted", zero_division=0)
        diff  = f1_nh - f1_h

        stat, p = nemenyi_two_treatment(errors_nh, errors_h)

        return {
            "dataset": dataset,
            "kpi": "Activity Occurrence (F1)",
            "LLM (F1)": f"{f1_nh:.2f}",
            "LLM anonymized (F1)": f"{f1_h:.2f}",
            "Difference": f"{diff:.2f}",
            "P-value": f"{p:.3f}",
            "Significance": _significance_stars(p),
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_table(rows: List[Dict]) -> None:
    if not rows:
        print("No results.")
        return
    headers = list(rows[0].keys())
    col_widths = [max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    print(sep)
    print("|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, col_widths)) + "|")
    print(sep)
    for row in rows:
        print("|" + "|".join(f" {str(row.get(h,'')):<{w}} " for h, w in zip(headers, col_widths)) + "|")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Nemenyi post-hoc test for RQ2 (hashed vs non-hashed prompts)")
    parser.add_argument("--output_dir", type=str,
                        default=str(Path(__file__).parent.parent / "output" / "pm"),
                        help="Root directory containing per-dataset output folders")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["bpi12", "bac", "hospital"],
                        help="Run for a single dataset (default: all)")
    parser.add_argument("--kpi", type=str, default=None,
                        choices=["lead_time", "outcome_pred"],
                        help="Run for a single KPI (default: both)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of training samples used in the experiment (default: 100)")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["bpi12", "bac", "hospital"]
    kpis     = [args.kpi]     if args.kpi     else ["lead_time", "outcome_pred"]

    print(f"\nNemenyi Post-Hoc Test: Anonymized vs Non-Anonymized Prompts (n={args.n_samples})\n")

    rows = []
    for kpi in kpis:
        for ds in datasets:
            try:
                result = run_analysis(args.output_dir, ds, kpi, n_samples=args.n_samples)
                rows.append(result)
                print(f"  [{ds}/{kpi}] done")
            except FileNotFoundError as e:
                print(f"  [{ds}/{kpi}] SKIPPED: {e}")
            except Exception as e:
                print(f"  [{ds}/{kpi}] ERROR: {e}")

    if rows:
        print()
        print_table(rows)


if __name__ == "__main__":
    main()
