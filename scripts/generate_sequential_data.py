"""
Generate sequential-encoding JSON train/test data for every (dataset, KPI, anonymized)
combination used in the paper, and write them into paper_artifacts/data/.

Mirrors the pipeline in main_pm4py.py but:
  - iterates over datasets, KPIs and the `hashed` flag,
  - forces sequential encoding regardless of what the hparams file specifies,
  - writes results into paper_artifacts/data/<dataset>/.

Outputs per dataset:
  - preprocessed_log_sequential_train_{kpi}.json
  - preprocessed_log_sequential_test_{kpi}.json
  - preprocessed_log_sequential_train_{kpi}_hashed.json
  - preprocessed_log_sequential_test_{kpi}_hashed.json
  - anonymization_dict.json     (activity name → 4-char code)
  - anonymization_attr_names.json (trace-attribute name → 4-char code)
"""

import json
import os
import sys
from pathlib import Path

# Add repo root to sys.path so `utils` is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import random
random.seed(1618)

from utils import add_activity_outcome
from utils import preprocessing_acts as pr_act
from utils import preprocessing_times as pr_time
from utils import log_parsing
from utils import IO


DATASETS = ["bpi12", "bac", "hospital"]
KPIS     = ["lead_time", "outcome_pred"]
HASHED   = [False, True]

OUTPUT_ROOT = ROOT / "paper_artifacts" / "data"


def run_one(dataset: str, kpi: str, hashed: bool) -> None:
    tag = f"{dataset}/{kpi}/hashed={hashed}"
    print(f"\n{'='*70}\n  Processing {tag}\n{'='*70}")

    with open(ROOT / "hparams" / f"{dataset}.json") as f:
        hparams = json.load(f)

    log_path    = hparams["log_path"]
    parse_dates = [hparams["start_date"], hparams["end_date"]]
    case_id_name, activity_column_name, trace_attr = log_parsing.parse_cf_caseid_traceatt(hparams)

    # Force sequential encoding for this artifact
    cf_preprocessing = "sequential"

    # ── encode log (adds activity count or sequential tagging) ────────────────
    log, attr_trace_dict = pr_act.encode_log(
        log_path,
        case_id_name=case_id_name,
        parse_dates=parse_dates,
        activity_column_name=activity_column_name,
        encoding=cf_preprocessing,
        last_act_num=3,
        trace_attr=trace_attr,
    )

    # ── time features ─────────────────────────────────────────────────────────
    log = pr_time.add_remaining_time_features(log)
    log = log.rename(columns={"remaining_time": "lead_time"})
    log = pr_time.add_daily_features(log)

    log = log_parsing.reorder_cols(log)
    log = log_parsing.drop_0s(log)
    log = log_parsing.add_attr(log, attr_trace_dict, cf_preprocessing)

    # ── outcome label ─────────────────────────────────────────────────────────
    if kpi == "outcome_pred":
        target_act = hparams["acts_not_freq"][0]
        log = add_activity_outcome.add_activity_outcome(
            log, act_to_encode=target_act, occurs_in_remaining=False
        )
        if "lead_time" in log.columns:
            del log["lead_time"]

    # ── anonymization ─────────────────────────────────────────────────────────
    hash_trace_attr_names = None
    hashing_dict_act, hashing_dict_attr = None, None
    if hashed:
        log, hashing_dict_act, hashing_dict_attr, hash_trace_attr_names = pr_act.hash_log(
            log,
            activity_column_name=activity_column_name,
            kpi=kpi,
            trace_attr=trace_attr,
        )

    # ── train/test split ──────────────────────────────────────────────────────
    train, test = pr_time.train_test_split(
        log,
        test_size=0.2,
        random_state=1618,
        temporal=True,
        encoding=cf_preprocessing,
        trace_attr=trace_attr,
        attr_trace_dict=attr_trace_dict,
        kpi=kpi,
        hash_trace_attr_names=hash_trace_attr_names,
    )

    # ── save ──────────────────────────────────────────────────────────────────
    out_folder = OUTPUT_ROOT / dataset
    out_folder.mkdir(parents=True, exist_ok=True)
    IO.save_log(
        experiment_folder=str(out_folder),
        log=train,
        encoding_cf=cf_preprocessing,
        type="train",
        kpi=kpi,
        hashed=hashed,
    )
    IO.save_log(
        experiment_folder=str(out_folder),
        log=test,
        encoding_cf=cf_preprocessing,
        type="test",
        kpi=kpi,
        hashed=hashed,
    )

    if hashed:
        with open(out_folder / "anonymization_dict.json", "w") as f:
            json.dump(hashing_dict_act, f, indent=2)
        if hash_trace_attr_names:
            with open(out_folder / "anonymization_attr_names.json", "w") as f:
                json.dump(hash_trace_attr_names, f, indent=2)

    print(f"  ✓ {tag} complete")


def main():
    for ds in DATASETS:
        for kpi in KPIS:
            for hashed in HASHED:
                try:
                    run_one(ds, kpi, hashed)
                except FileNotFoundError as e:
                    print(f"  ✗ Skipping {ds}/{kpi}/hashed={hashed}: {e}")
                except Exception as e:
                    print(f"  ✗ ERROR {ds}/{kpi}/hashed={hashed}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
