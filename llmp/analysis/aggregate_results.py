"""
Results Aggregation Script

Loads all Gemini experiment JSON output files for one or more datasets/KPIs and
produces summary tables matching the paper (Table 1 for MAE, Table 2 for F1-Score).

Each JSON file produced by ExperimentManager.run() corresponds to one experimental
run (one seed). This script aggregates across all seed files to compute mean ± std.

Usage:
    python aggregate_results.py                          # all datasets, both KPIs
    python aggregate_results.py --dataset bpi12 --kpi lead_time
    python aggregate_results.py --n_samples 100
"""

import argparse
import json
import glob
import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np


# ── JSON loading helpers ──────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _extract_mae_values(data: dict, n_samples: int) -> List[float]:
    """Collect all seed-level MAE values from a lead_time output JSON."""
    maes = []
    key = str(n_samples)
    if key not in data:
        return maes
    for test_seed_data in data[key].get("test_seeds", {}).values():
        for train_seed_data in test_seed_data.get("train_seeds", {}).values():
            mae = train_seed_data.get("metrics", {}).get("MAE")
            if mae is not None:
                maes.append(float(mae))
    return maes


def _extract_f1_values(data: dict, n_samples: int) -> List[float]:
    """Collect all seed-level F1 values from an outcome_pred output JSON."""
    f1s = []
    key = str(n_samples)
    if key not in data:
        return f1s
    for test_seed_data in data[key].get("test_seeds", {}).values():
        for train_seed_data in test_seed_data.get("train_seeds", {}).values():
            f1 = train_seed_data.get("metrics", {}).get("F1")
            if f1 is not None:
                f1s.append(float(f1))
    return f1s


# ── aggregation logic ─────────────────────────────────────────────────────────

def aggregate_dataset(
    output_dir: str,
    dataset: str,
    kpi: str,
    n_samples: int,
    hashed: bool = False,
) -> Optional[Dict]:
    """
    Aggregate results for one (dataset, kpi, n_samples, hashed) combination.
    Returns a dict with mean, std, and n_files, or None if no files found.
    """
    base = Path(output_dir) / dataset
    kpi_tag = "outcomepred" if kpi == "outcome_pred" else ""

    if kpi_tag:
        pattern = str(base / f"gemini_results_multiple_seeds_n{n_samples}seq_{kpi_tag}*.json")
    else:
        pattern = str(base / f"gemini_results_multiple_seeds_n{n_samples}seq*.json")

    all_files = glob.glob(pattern)
    if hashed:
        files = [f for f in all_files if "hashed" in os.path.basename(f)]
    else:
        files = [f for f in all_files if "hashed" not in os.path.basename(f)]

    if not files:
        return None

    all_values: List[float] = []
    for f in files:
        data = load_json(f)
        if kpi == "lead_time":
            all_values.extend(_extract_mae_values(data, n_samples))
        else:
            all_values.extend(_extract_f1_values(data, n_samples))

    if not all_values:
        return None

    return {
        "mean": float(np.mean(all_values)),
        "std":  float(np.std(all_values)),
        "n":    len(all_values),
        "files": len(files),
    }


# ── table printing ─────────────────────────────────────────────────────────────

def _fmt(mean: float, std: float, kpi: str) -> str:
    if kpi == "lead_time":
        return f"{mean:.0f} ± {std:.0f}"
    else:
        return f"{mean:.2f} ± {std:.2f}"


def print_summary_table(
    output_dir: str,
    datasets: List[str],
    kpis: List[str],
    n_samples: int,
) -> None:
    for kpi in kpis:
        metric_name = "MAE (minutes)" if kpi == "lead_time" else "F1-Score"
        print(f"\n{'='*60}")
        print(f"  KPI: {kpi}  |  Metric: {metric_name}  |  n_samples = {n_samples}")
        print(f"{'='*60}")
        print(f"{'Dataset':<12} {'LLM':>20} {'LLM (anonymized)':>22}")
        print("-" * 56)

        for ds in datasets:
            nh = aggregate_dataset(output_dir, ds, kpi, n_samples, hashed=False)
            h  = aggregate_dataset(output_dir, ds, kpi, n_samples, hashed=True)

            nh_str = _fmt(nh["mean"], nh["std"], kpi) if nh else "N/A"
            h_str  = _fmt(h["mean"],  h["std"],  kpi) if h  else "N/A"
            print(f"{ds:<12} {nh_str:>20} {h_str:>22}")

        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Gemini experiment results into summary tables."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(Path(__file__).parent.parent / "output" / "pm"),
        help="Root directory with per-dataset output folders",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["bpi12", "bac", "hospital"],
        help="Single dataset (default: all three)",
    )
    parser.add_argument(
        "--kpi", type=str, default=None,
        choices=["lead_time", "outcome_pred"],
        help="Single KPI (default: both)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=100,
        help="Number of training samples used (default: 100)",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["bpi12", "bac", "hospital"]
    kpis     = [args.kpi]     if args.kpi     else ["lead_time", "outcome_pred"]

    print_summary_table(args.output_dir, datasets, kpis, args.n_samples)


if __name__ == "__main__":
    main()
