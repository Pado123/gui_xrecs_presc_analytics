"""
RQ3 Beta-Learner Evaluation Script

For each dataset and KPI, this script:
  1. Loads the preprocessed sequential train/test JSON data.
  2. Trains and evaluates all beta-learners (re-implementations of LLM reasoning patterns).
  3. Loads LLM per-trace predictions from the Gemini output JSON files.
  4. Runs a Wilcoxon signed-rank test comparing each beta-learner against the LLM.
  5. Computes Good-Turing frequency estimation to assess beta-learner catalog completeness.
  6. Optionally accepts a classification file (which beta-learner the LLM used per trace)
     to compute the "Δ LLM when occurring" column from the paper.

Beta-learners for lead_time (total time):
  - knn_act_{agg}   : KNN on activity-sequence features (mean/median/mode aggregation)
  - knn_att_{agg}   : KNN on trace-attribute features
  - time_seq_{agg}  : KNN on activity-duration sequence features
  - path_pred_{agg} : KNN on predicted next-activity path features

Beta-learners for outcome_pred (activity occurrence):
  - activity_based  : Decision-tree on activity-count (aggr_hist) features
  - state_based     : Decision-tree on last-event features
  - att_based       : Decision-tree on trace-attribute features
  - positive_evidence: Checks if the target activity already occurred

Usage:
    python beta_learner_eval.py --dataset bpi12 --kpi lead_time \\
        --data_dir ../../experiments/bpi12 \\
        --output_dir ../output/pm/bpi12

    python beta_learner_eval.py --dataset hospital --kpi outcome_pred \\
        --data_dir ../../experiments/hospital \\
        --output_dir ../output/pm/hospital \\
        --classification_file classifications_hospital.json
"""

import argparse
import ast
import json
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# Add parent directories to path so local utils are importable
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "llmp" / "gemini"))


# ── data loading ──────────────────────────────────────────────────────────────

def load_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_sequential_data(data_dir: str, kpi: str, hashed: bool = False) -> Tuple[dict, dict]:
    """Load train and test sequential JSON dicts."""
    suffix = "_hashed" if hashed else ""
    train_path = Path(data_dir) / f"preprocessed_log_sequential_train_{kpi}{suffix}.json"
    test_path  = Path(data_dir) / f"preprocessed_log_sequential_test_{kpi}{suffix}.json"

    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    return load_json(train_path), load_json(test_path)


def _parse_act_time_seq(raw) -> List[Tuple]:
    """Parse ActTimeSeq, which may be stored as a list or a string repr of a list."""
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except Exception:
            return []
    return raw or []


# ── feature extraction ────────────────────────────────────────────────────────

def extract_activity_features(trace_dict: dict) -> np.ndarray:
    """knn_act: encode activity sequence as bag-of-activities (occurrence counts)."""
    seq = _parse_act_time_seq(trace_dict.get("ActTimeSeq", []))
    activities = [s[0] for s in seq if isinstance(s, (list, tuple)) and len(s) >= 1
                  and s[0] != "Running"]
    return activities  # will be label-encoded after building vocabulary


def extract_attribute_features(trace_dict: dict, attr_keys: List[str]) -> List:
    """knn_att / att_based: trace-level attribute values."""
    return [trace_dict.get(k) for k in attr_keys]


def extract_time_seq_features(trace_dict: dict) -> List[float]:
    """time_seq: flat list of cumulative elapsed times from ActTimeSeq."""
    seq = _parse_act_time_seq(trace_dict.get("ActTimeSeq", []))
    return [float(s[1]) for s in seq if isinstance(s, (list, tuple)) and len(s) >= 2
            and s[0] != "Running"]


def build_activity_bag_matrix(
    train_traces: List[dict],
    test_traces: List[dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build bag-of-activities matrices for train and test."""
    vocab = set()
    for t in train_traces + test_traces:
        vocab.update(extract_activity_features(t))
    vocab = sorted(vocab)
    v_idx = {a: i for i, a in enumerate(vocab)}

    def to_vec(trace):
        acts = extract_activity_features(trace)
        vec = np.zeros(len(vocab))
        for a in acts:
            if a in v_idx:
                vec[v_idx[a]] += 1
        return vec

    X_train = np.vstack([to_vec(t) for t in train_traces])
    X_test  = np.vstack([to_vec(t) for t in test_traces])
    return X_train, X_test


def build_padded_time_matrix(
    train_traces: List[dict],
    test_traces: List[dict],
    max_len: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build zero-padded time-sequence matrices."""
    train_seqs = [extract_time_seq_features(t) for t in train_traces]
    test_seqs  = [extract_time_seq_features(t) for t in test_traces]
    if max_len is None:
        max_len = max((len(s) for s in train_seqs + test_seqs), default=1)

    def pad(seq):
        arr = np.zeros(max_len)
        arr[:min(len(seq), max_len)] = seq[:max_len]
        return arr

    X_train = np.vstack([pad(s) for s in train_seqs])
    X_test  = np.vstack([pad(s) for s in test_seqs])
    return X_train, X_test


def build_attribute_matrix(
    train_traces: List[dict],
    test_traces: List[dict],
    attr_keys: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build attribute matrices, label-encoding strings."""
    encoders: Dict[int, LabelEncoder] = {}

    def get_raw(traces):
        return [extract_attribute_features(t, attr_keys) for t in traces]

    raw_train = get_raw(train_traces)
    raw_test  = get_raw(test_traces)

    # Fit encoders on train, apply to both
    n_cols = len(attr_keys)
    mat_train = np.zeros((len(raw_train), n_cols))
    mat_test  = np.zeros((len(raw_test),  n_cols))

    for col in range(n_cols):
        col_vals_train = [row[col] for row in raw_train]
        col_vals_test  = [row[col] for row in raw_test]

        if any(isinstance(v, str) for v in col_vals_train):
            le = LabelEncoder()
            le.fit([str(v) for v in col_vals_train])
            encoders[col] = le
            mat_train[:, col] = le.transform([str(v) for v in col_vals_train])
            mat_test[:, col] = np.array([
                le.transform([str(v)])[0] if str(v) in le.classes_ else 0
                for v in col_vals_test
            ])
        else:
            mat_train[:, col] = np.array([v if v is not None else 0.0 for v in col_vals_train], dtype=float)
            mat_test[:, col]  = np.array([v if v is not None else 0.0 for v in col_vals_test],  dtype=float)

    return mat_train, mat_test


# ── KNN aggregation helpers ───────────────────────────────────────────────────

def knn_predict(X_train, y_train, X_test, k: int = 5, agg: str = "median") -> np.ndarray:
    """KNN regressor with custom aggregation (mean / median / mode)."""
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(k, len(X_train))).fit(X_train)
    distances, indices = nbrs.kneighbors(X_test)

    preds = []
    for idx_row in indices:
        neighbors = y_train[idx_row]
        if agg == "mean":
            preds.append(float(np.mean(neighbors)))
        elif agg == "median":
            preds.append(float(np.median(neighbors)))
        elif agg == "mode":
            vals, counts = np.unique(neighbors, return_counts=True)
            preds.append(float(vals[np.argmax(counts)]))
        else:
            raise ValueError(f"Unknown aggregation: {agg}")
    return np.array(preds)


# ── lead_time beta-learners ───────────────────────────────────────────────────

def run_lead_time_beta_learners(
    train_traces: List[dict],
    test_traces: List[dict],
    y_train: np.ndarray,
    y_test: np.ndarray,
    attr_keys: List[str],
    k: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Train and predict with all lead_time beta-learners.
    Returns a dict mapping learner name → per-trace predictions array.
    """
    results = {}

    # knn_act
    X_tr, X_te = build_activity_bag_matrix(train_traces, test_traces)
    for agg in ("mean", "median", "mode"):
        name = f"knn_act_{agg}"
        results[name] = knn_predict(X_tr, y_train, X_te, k=k, agg=agg)

    # knn_att
    if attr_keys:
        X_tr, X_te = build_attribute_matrix(train_traces, test_traces, attr_keys)
        for agg in ("mean", "median", "mode"):
            name = f"knn_att_{agg}"
            results[name] = knn_predict(X_tr, y_train, X_te, k=k, agg=agg)

    # time_seq
    X_tr, X_te = build_padded_time_matrix(train_traces, test_traces)
    for agg in ("mean", "median", "mode"):
        name = f"time_seq_{agg}"
        results[name] = knn_predict(X_tr, y_train, X_te, k=k, agg=agg)

    # path_pred: use activity bag as proxy for path-based prediction
    # (true path_pred requires iterative next-activity estimation; this approximates it)
    X_tr, X_te = build_activity_bag_matrix(train_traces, test_traces)
    for agg in ("mean", "median", "mode"):
        name = f"path_pred_{agg}"
        # Weight neighbors by inverse distance to approximate path similarity
        nbrs_model = KNeighborsRegressor(n_neighbors=min(k, len(X_tr)), weights="distance")
        nbrs_model.fit(X_tr, y_train)
        if agg == "median":
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(k, len(X_tr))).fit(X_tr)
            _, idxs = nn.kneighbors(X_te)
            preds = np.array([np.median(y_train[idx]) for idx in idxs])
        elif agg == "mode":
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(k, len(X_tr))).fit(X_tr)
            _, idxs = nn.kneighbors(X_te)
            preds = np.array([
                float(np.unique(y_train[idx])[np.argmax(np.unique(y_train[idx], return_counts=True)[1])])
                for idx in idxs
            ])
        else:
            preds = nbrs_model.predict(X_te)
        results[name] = preds

    return results


# ── outcome_pred beta-learners ────────────────────────────────────────────────

def _last_event_features(trace: dict) -> List:
    """state_based: features from the last activity in the sequence."""
    seq = _parse_act_time_seq(trace.get("ActTimeSeq", []))
    if not seq:
        return [None, None]
    last = seq[-1]
    act  = last[0] if len(last) > 0 else None
    time = float(last[1]) if len(last) > 1 else 0.0
    return [act, time]


def run_outcome_beta_learners(
    train_traces: List[dict],
    test_traces: List[dict],
    y_train: np.ndarray,
    y_test: np.ndarray,
    attr_keys: List[str],
    target_activity: str,
) -> Dict[str, np.ndarray]:
    """
    Train and predict with all outcome_pred beta-learners.
    Returns a dict mapping learner name → per-trace predictions array (0/1).
    """
    results = {}

    def _dt_predict(X_tr, X_te, y_tr):
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X_tr, y_tr)
        return clf.predict(X_te).astype(int)

    # activity_based: bag-of-activities
    X_tr, X_te = build_activity_bag_matrix(train_traces, test_traces)
    results["activity_based"] = _dt_predict(X_tr, X_te, y_train)

    # state_based: last-event (activity + elapsed time)
    raw_state_train = [_last_event_features(t) for t in train_traces]
    raw_state_test  = [_last_event_features(t) for t in test_traces]
    le_act = LabelEncoder()
    acts_all = [str(r[0]) for r in raw_state_train + raw_state_test]
    le_act.fit(acts_all)
    def to_state_vec(raw_list):
        mat = np.zeros((len(raw_list), 2))
        for i, r in enumerate(raw_list):
            mat[i, 0] = le_act.transform([str(r[0])])[0] if r[0] is not None else 0
            mat[i, 1] = r[1] if r[1] is not None else 0.0
        return mat
    X_tr_state = to_state_vec(raw_state_train)
    X_te_state = to_state_vec(raw_state_test)
    results["state_based"] = _dt_predict(X_tr_state, X_te_state, y_train)

    # att_based: trace attributes
    if attr_keys:
        X_tr, X_te = build_attribute_matrix(train_traces, test_traces, attr_keys)
        results["att_based"] = _dt_predict(X_tr, X_te, y_train)

    # positive_evidence: check if target activity already appears in the prefix
    pe_preds = []
    for trace in test_traces:
        seq = _parse_act_time_seq(trace.get("ActTimeSeq", []))
        already_occurred = any(
            s[0] == target_activity
            for s in seq
            if isinstance(s, (list, tuple)) and len(s) >= 1
        )
        if already_occurred:
            pe_preds.append(1)
        else:
            # Random guess when not yet seen (50/50 baseline)
            pe_preds.append(int(np.random.rand() > 0.5))
    results["positive_evidence"] = np.array(pe_preds)

    return results


# ── statistical tests ─────────────────────────────────────────────────────────

def wilcoxon_test(errors_llm: np.ndarray, errors_learner: np.ndarray) -> Tuple[float, str]:
    """Wilcoxon signed-rank test. Returns (p_value, significance_stars)."""
    try:
        _, p = wilcoxon(errors_llm, errors_learner, alternative="two-sided")
    except ValueError:
        p = 1.0
    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    return p, stars


def good_turing_estimation(occurrence_counts: Dict[str, int]) -> Dict:
    """
    Good-Turing frequency estimation to assess beta-learner catalog completeness.

    occurrence_counts: {learner_name: number_of_traces_classified_as_this_learner}

    Returns:
      N        - total observations
      N1       - learners observed exactly once ('singletons')
      P0       - estimated probability of a new unseen learner
      E_novel  - expected novel learners for m=1, 10, 100 new observations
    """
    counts = list(occurrence_counts.values())
    N  = sum(counts)
    N1 = sum(1 for c in counts if c == 1)
    P0 = N1 / N if N > 0 else 0.0
    return {
        "N":  N,
        "N1": N1,
        "P0": P0,
        "E_novel_1":   1   * P0,
        "E_novel_10":  10  * P0,
        "E_novel_100": 100 * P0,
    }


# ── LLM prediction loading ────────────────────────────────────────────────────

def load_llm_predictions(output_dir: str, dataset: str, kpi: str, n_samples: int) -> List[float]:
    """Load flat list of LLM per-trace predictions (non-hashed) from output JSONs."""
    base = Path(output_dir) / dataset
    kpi_tag = "outcomepred" if kpi == "outcome_pred" else ""

    if kpi_tag:
        pattern = str(base / f"gemini_results_multiple_seeds_n{n_samples}seq_{kpi_tag}*.json")
    else:
        pattern = str(base / f"gemini_results_multiple_seeds_n{n_samples}seq*.json")

    files = [f for f in glob.glob(pattern) if "hashed" not in os.path.basename(f)]
    preds = []
    for f in files:
        data = load_json(f)
        key = str(n_samples)
        if key not in data:
            continue
        for test_seed_data in data[key].get("test_seeds", {}).values():
            for train_seed_data in test_seed_data.get("train_seeds", {}).values():
                preds.extend(train_seed_data.get("predictions", []))
    return preds


def load_llm_true_labels(output_dir: str, dataset: str, kpi: str, n_samples: int) -> List[float]:
    """Load flat list of true labels paired with the LLM predictions."""
    base = Path(output_dir) / dataset
    kpi_tag = "outcomepred" if kpi == "outcome_pred" else ""

    if kpi_tag:
        pattern = str(base / f"gemini_results_multiple_seeds_n{n_samples}seq_{kpi_tag}*.json")
    else:
        pattern = str(base / f"gemini_results_multiple_seeds_n{n_samples}seq*.json")

    files = [f for f in glob.glob(pattern) if "hashed" not in os.path.basename(f)]
    labels = []
    for f in files:
        data = load_json(f)
        key = str(n_samples)
        if key not in data:
            continue
        for test_seed_data in data[key].get("test_seeds", {}).values():
            n_trains = len(test_seed_data.get("train_seeds", {}))
            true_vals = test_seed_data.get("aggregated_metrics", {}).get("True_values", [])
            # True values are repeated once per train seed
            labels.extend(true_vals * n_trains)
    return labels


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    data_dir: str,
    output_dir: str,
    dataset: str,
    kpi: str,
    n_samples: int,
    attr_keys: List[str],
    target_activity: str,
    classification_file: Optional[str],
    k_neighbors: int,
) -> None:
    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset}  |  KPI: {kpi}  |  n_samples: {n_samples}")
    print(f"{'='*70}")

    # Load sequential data
    train_dict, test_dict = load_sequential_data(data_dir, kpi)
    train_traces = list(train_dict.values())
    test_traces  = list(test_dict.values())

    # Sample n_samples training traces (fixed seed for reproducibility)
    rng = np.random.RandomState(1618)
    idx = rng.choice(len(train_traces), size=min(n_samples, len(train_traces)), replace=False)
    train_traces_sampled = [train_traces[i] for i in idx]

    # Extract labels
    if kpi == "lead_time":
        y_train = np.array([float(t.get("lead_time", 0)) for t in train_traces_sampled])
        y_test  = np.array([float(t.get("lead_time", 0)) for t in test_traces])
    else:
        last_key_train = [list(t.keys())[-1] for t in train_traces_sampled]
        last_key_test  = [list(t.keys())[-1] for t in test_traces]
        y_train = np.array([int(t.get(k, 0)) for t, k in zip(train_traces_sampled, last_key_train)])
        y_test  = np.array([int(t.get(k, 0)) for t, k in zip(test_traces, last_key_test)])

    # Load LLM predictions
    llm_preds  = np.array(load_llm_predictions(output_dir, dataset, kpi, n_samples))
    llm_labels = np.array(load_llm_true_labels(output_dir, dataset, kpi, n_samples))

    n_test = len(y_test)
    # Align LLM predictions to test set size
    if len(llm_preds) >= n_test:
        llm_preds_aligned  = llm_preds[:n_test]
        llm_labels_aligned = llm_labels[:n_test]
    else:
        llm_preds_aligned  = llm_preds
        llm_labels_aligned = llm_labels
        n_test = len(llm_preds_aligned)
        y_test = y_test[:n_test]
        test_traces = test_traces[:n_test]

    # Compute LLM errors
    if kpi == "lead_time":
        llm_errors = np.abs(llm_preds_aligned - llm_labels_aligned)
        llm_mae    = float(np.mean(llm_errors))
        print(f"\nLLM MAE: {llm_mae:.0f}  (n_preds={len(llm_preds_aligned)})")
    else:
        llm_errors = np.abs(
            np.array([int(round(p)) for p in llm_preds_aligned]) -
            np.array([int(l) for l in llm_labels_aligned])
        )
        llm_f1 = f1_score(
            [int(l) for l in llm_labels_aligned],
            [int(round(p)) for p in llm_preds_aligned],
            average="weighted", zero_division=0,
        )
        print(f"\nLLM F1: {llm_f1:.3f}  (n_preds={len(llm_preds_aligned)})")

    # Run beta-learners
    if kpi == "lead_time":
        learner_preds = run_lead_time_beta_learners(
            train_traces_sampled, test_traces[:n_test],
            y_train, y_test, attr_keys, k=k_neighbors,
        )
    else:
        np.random.seed(42)
        learner_preds = run_outcome_beta_learners(
            train_traces_sampled, test_traces[:n_test],
            y_train, y_test, attr_keys, target_activity,
        )

    # Load optional classification (which beta-learner the LLM used per trace)
    classification: Optional[Dict[int, str]] = None
    if classification_file and os.path.exists(classification_file):
        with open(classification_file) as f:
            raw = json.load(f)
        classification = {int(k): v for k, v in raw.items()}
        print(f"\nClassification file loaded: {len(classification)} trace labels")

    occurrence_counts: Dict[str, int] = {}

    # Print results table
    header = f"{'Model':<20} {'Metric':>12} {'p-value':>10} {'Sig':>6}"
    if classification:
        header += f" {'Occurrence':>12} {'Δ LLM occur.':>14}"
    print("\n" + header)
    print("-" * len(header))

    for name, preds in learner_preds.items():
        preds_aligned = preds[:n_test]

        if kpi == "lead_time":
            learner_errors = np.abs(preds_aligned - y_test[:n_test])
            metric_val = float(np.mean(learner_errors))
            metric_str = f"{metric_val:.0f}"
        else:
            learner_errors = np.abs(preds_aligned.astype(int) - y_test[:n_test].astype(int))
            metric_val = f1_score(
                y_test[:n_test].astype(int), preds_aligned.astype(int),
                average="weighted", zero_division=0,
            )
            metric_str = f"{metric_val:.3f}"

        p, stars = wilcoxon_test(llm_errors[:n_test], learner_errors[:n_test])
        row = f"{name:<20} {metric_str:>12} {p:>10.4f} {stars:>6}"

        if classification:
            # Traces classified as this learner
            classified_idxs = [i for i, lbl in classification.items() if lbl == name and i < n_test]
            occurrence = len(classified_idxs)
            occurrence_counts[name] = occurrence

            if classified_idxs:
                llm_sub    = llm_errors[classified_idxs]
                learn_sub  = learner_errors[classified_idxs]
                delta = float(np.mean(learn_sub) - np.mean(llm_sub))
                if kpi == "lead_time":
                    delta_str = f"{delta/max(np.mean(llm_sub), 1)*100:.1f}%"
                else:
                    delta_str = f"{delta:.3f}"
            else:
                delta_str = "N/A"
            row += f" {occurrence:>12} {delta_str:>14}"

        print(row)

    # Good-Turing estimation
    if occurrence_counts:
        print(f"\n--- Good-Turing Frequency Estimation ---")
        gt = good_turing_estimation(occurrence_counts)
        print(f"  Total observations (N) : {gt['N']}")
        print(f"  Singletons (N1)        : {gt['N1']}")
        print(f"  P(unseen learner)  P0  : {gt['P0']:.4f}")
        print(f"  E[novel] for m=1       : {gt['E_novel_1']:.4f}")
        print(f"  E[novel] for m=10      : {gt['E_novel_10']:.4f}")
        print(f"  E[novel] for m=100     : {gt['E_novel_100']:.4f}")
    else:
        print("\nNote: provide --classification_file to compute Good-Turing estimation "
              "and Δ LLM when occurring.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RQ3 beta-learner evaluation with Wilcoxon test and Good-Turing estimation"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["bpi12", "bac", "hospital"])
    parser.add_argument("--kpi", type=str, required=True,
                        choices=["lead_time", "outcome_pred"])
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to experiments/{dataset}/ with sequential JSON files")
    parser.add_argument("--output_dir", type=str,
                        default=str(Path(__file__).parent.parent / "output" / "pm"),
                        help="Root of Gemini output folders (for LLM predictions)")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--attr_keys", type=str, nargs="*", default=[],
                        help="Trace attribute keys used as features (e.g. AMOUNT_REQ)")
    parser.add_argument("--target_activity", type=str, default="",
                        help="Target activity name for outcome_pred (as it appears in the data)")
    parser.add_argument("--classification_file", type=str, default=None,
                        help="JSON file {trace_idx: learner_name} for Δ LLM and Good-Turing")
    parser.add_argument("--k_neighbors", type=int, default=5,
                        help="Number of neighbors for KNN-based beta-learners")
    args = parser.parse_args()

    evaluate(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        kpi=args.kpi,
        n_samples=args.n_samples,
        attr_keys=args.attr_keys,
        target_activity=args.target_activity,
        classification_file=args.classification_file,
        k_neighbors=args.k_neighbors,
    )


if __name__ == "__main__":
    main()
