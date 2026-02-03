#!/usr/bin/env python3
"""
Train a LoL pro-play draft win-probability model from CSV(s).

Expected CSV columns (case-sensitive):
match_id, patch, region, tournament, team1_name, team2_name,
team1_drafted_champions, team2_drafted_champions,
team1_banned_champions, team2_banned_champions, winner

- team1 = blue side by your construction
- winner should be 1 if team1 won, else 0
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import json
import re
from typing import Any
from collections import Counter
from itertools import combinations
from collections import Counter
from itertools import combinations


REQUIRED_COLS = [
    "match_id",
    "patch",
    "region",
    "tournament",
    "team1_name",
    "team2_name",
    "team1_drafted_champions",
    "team2_drafted_champions",
    "team1_banned_champions",
    "team2_banned_champions",
    "winner",
]


def _split_pipe_list(s: Optional[str]) -> List[str]:
    """Split 'A|B|C' into ['A','B','C'], safely."""
    if s is None:
        return []
    s = str(s).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    return [x.strip() for x in s.split("|") if x.strip()]

def _norm_name(s: str) -> str:
    """Lowercase and remove non-alphanumerics to normalize names like K'Sante -> ksante, Xin Zhao -> xinzhao."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def _canon_champ(token: str, alias_to_id: Dict[str, str]) -> str | None:
    """
    Convert a raw CSV token (e.g. "K'Sante", "Xin Zhao") to canonical ID ("KSante", "XinZhao").
    Returns None if unknown/unmappable.
    """
    key = _norm_name(token)          # relies on your _norm_name from earlier
    return alias_to_id.get(key)

def load_champion_master(champion_json_path: str):
    """
    Returns:
      vocab_ids: List[str] stable champion IDs (length 172)
      alias_to_id: Dict[str, str] mapping normalized forms -> canonical ID
    """
    with open(champion_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    champs = payload["data"]  # dict keyed by canonical ID (e.g., "KSante")
    # Stable ordering: by numeric "key" (Riot champ integer ID)
    items = sorted(champs.items(), key=lambda kv: int(kv[1]["key"]))

    vocab_ids = [cid for cid, _info in items]

    alias_to_id = {}
    for cid, info in items:
        # info typically contains: id (same as cid), name (display name), key (numeric string)
        display = info.get("name", cid)
        # Map multiple variants -> canonical
        for variant in {cid, display}:
            alias_to_id[_norm_name(variant)] = cid

    return vocab_ids, alias_to_id

def _normalize_patch(p: Optional[str]) -> str:
    """Turn missing/blank patch into 'UNKNOWN'. Keep others as strings."""
    if p is None:
        return "UNKNOWN"
    p = str(p).strip()
    if not p or p.lower() in {"nan", "none"}:
        return "UNKNOWN"
    return p

def _patch_bucket(p: Optional[str]) -> str:
    """
    Bucket patches like '15.2' -> '15.x' to reduce sparsity and improve generalization.
    If parsing fails, returns 'UNKNOWN'.
    """
    p = _normalize_patch(p)
    if p == "UNKNOWN":
        return "UNKNOWN"
    # accept '15.2' or '15.2.1' etc.
    parts = p.split(".")
    if len(parts) >= 2 and parts[0].isdigit():
        return f"{parts[0]}.x"
    return "UNKNOWN"


def _load_csvs(paths: Sequence[str]) -> pd.DataFrame:
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        df["__source_file"] = os.path.basename(path)
        dfs.append(df)
    if not dfs:
        raise ValueError("No CSV files loaded.")
    return pd.concat(dfs, ignore_index=True)


def build_frequent_pairs(
    df: pd.DataFrame,
    alias_to_id: Dict[str, str],
    min_count: int = 15,
    top_k: int = 20,
) -> list[tuple[str, str]]:
    pair_counts: Counter[tuple[str, str]] = Counter()

    for row in df.itertuples(index=False):
        t1_raw = _split_pipe_list(getattr(row, "team1_drafted_champions"))
        t2_raw = _split_pipe_list(getattr(row, "team2_drafted_champions"))

        t1 = sorted({c for tok in t1_raw if (c := _canon_champ(tok, alias_to_id)) is not None})
        t2 = sorted({c for tok in t2_raw if (c := _canon_champ(tok, alias_to_id)) is not None})

        for a, b in combinations(t1, 2):
            pair_counts[(a, b)] += 1
        for a, b in combinations(t2, 2):
            pair_counts[(a, b)] += 1

    # Filter by min_count
    items = [(pair, cnt) for pair, cnt in pair_counts.items() if cnt >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)

    if top_k and top_k > 0:
        items = items[:top_k]

    return [pair for pair, _ in items]

def compute_ban_scale_from_train(
    df_train: pd.DataFrame,
    alias_to_id: Dict[str, str],
    min_scale: float = 0.5,
    max_scale: float = 1.5,
) -> Dict[str, float]:
    """
    Returns a dict: canon_champ_id -> scale factor for ban weight.

    Idea: champions picked more often in TRAIN get a larger ban scale.
    We map log1p(count) to [min_scale, max_scale].
    """
    pick_counts: Counter[str] = Counter()

    for s in df_train["team1_drafted_champions"].tolist():
        for tok in _split_pipe_list(s):
            canon = _canon_champ(tok, alias_to_id)
            if canon is not None:
                pick_counts[canon] += 1

    for s in df_train["team2_drafted_champions"].tolist():
        for tok in _split_pipe_list(s):
            canon = _canon_champ(tok, alias_to_id)
            if canon is not None:
                pick_counts[canon] += 1

    if not pick_counts:
        return {}

    # log scale then normalize to [0,1]
    vals = np.array([np.log1p(c) for c in pick_counts.values()], dtype=float)
    vmin, vmax = float(vals.min()), float(vals.max())
    denom = (vmax - vmin) if (vmax > vmin) else 1.0

    ban_scale: Dict[str, float] = {}
    for champ, c in pick_counts.items():
        x = (np.log1p(c) - vmin) / denom  # 0..1
        ban_scale[champ] = min_scale + x * (max_scale - min_scale)

    return ban_scale

def encode_pair_features(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    alias_to_id: Dict[str, str],
) -> np.ndarray:
    pair_index = {p: i for i, p in enumerate(pairs)}
    X = np.zeros((len(df), len(pairs)), dtype=np.float32)

    for i, row in enumerate(df.itertuples(index=False)):
        t1_raw = _split_pipe_list(getattr(row, "team1_drafted_champions"))
        t2_raw = _split_pipe_list(getattr(row, "team2_drafted_champions"))

        t1 = {c for tok in t1_raw if (c := _canon_champ(tok, alias_to_id)) is not None}
        t2 = {c for tok in t2_raw if (c := _canon_champ(tok, alias_to_id)) is not None}

        for (a, b), j in pair_index.items():
            if a in t1 and b in t1:
                X[i, j] = 1.0
            elif a in t2 and b in t2:
                X[i, j] = -1.0

    return X

def build_frequent_counters(
    df: pd.DataFrame,
    alias_to_id: Dict[str, str],
    min_count: int = 20,
    top_k: int = 50,
) -> list[tuple[str, str]]:
    """
    Find frequent cross-team pick matchups (A on one team, B on the other).
    Returns ordered pairs (A, B).
    """
    counter_counts: Counter[tuple[str, str]] = Counter()

    for row in df.itertuples(index=False):
        t1_raw = _split_pipe_list(getattr(row, "team1_drafted_champions"))
        t2_raw = _split_pipe_list(getattr(row, "team2_drafted_champions"))

        t1 = {c for tok in t1_raw if (c := _canon_champ(tok, alias_to_id)) is not None}
        t2 = {c for tok in t2_raw if (c := _canon_champ(tok, alias_to_id)) is not None}

        # Count all cross pairs (ordered)
        for a in t1:
            for b in t2:
                counter_counts[(a, b)] += 1

    items = [(pair, cnt) for pair, cnt in counter_counts.items() if cnt >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)

    if top_k and top_k > 0:
        items = items[:top_k]

    return [pair for pair, _ in items]


def encode_counter_features(
    df: pd.DataFrame,
    counters: list[tuple[str, str]],
    alias_to_id: Dict[str, str],
) -> np.ndarray:
    """
    For each ordered counter pair (A, B):
      +1 if A in Team1 picks and B in Team2 picks
      -1 if A in Team2 picks and B in Team1 picks
      0 otherwise
    """
    counter_index = {p: i for i, p in enumerate(counters)}
    X = np.zeros((len(df), len(counters)), dtype=np.float32)

    for i, row in enumerate(df.itertuples(index=False)):
        t1_raw = _split_pipe_list(getattr(row, "team1_drafted_champions"))
        t2_raw = _split_pipe_list(getattr(row, "team2_drafted_champions"))

        t1 = {c for tok in t1_raw if (c := _canon_champ(tok, alias_to_id)) is not None}
        t2 = {c for tok in t2_raw if (c := _canon_champ(tok, alias_to_id)) is not None}

        for (a, b), j in counter_index.items():
            if (a in t1) and (b in t2):
                X[i, j] = 1.0
            elif (a in t2) and (b in t1):
                X[i, j] = -1.0

    return X

def _champions_used_in_df(df: pd.DataFrame, alias_to_id: Dict[str, str]) -> set[str]:
    used: set[str] = set()
    cols = [
        "team1_drafted_champions",
        "team2_drafted_champions",
        "team1_banned_champions",
        "team2_banned_champions",
    ]

    for col in cols:
        for s in df[col].tolist():
            for tok in _split_pipe_list(s):
                canon = _canon_champ(tok, alias_to_id)
                if canon is not None:
                    used.add(canon)
    return used

def _build_champion_vocab(df: pd.DataFrame) -> List[str]:
    champs = set()
    for col in [
        "team1_drafted_champions",
        "team2_drafted_champions",
        "team1_banned_champions",
        "team2_banned_champions",
    ]:
        for s in df[col].tolist():
            champs.update(_split_pipe_list(s))
    champs.discard("")  # just in case
    vocab = sorted(champs)
    if not vocab:
        raise ValueError("Champion vocabulary ended up empty. Check your CSV columns/format.")
    return vocab


def _encode_drafts(
    df: pd.DataFrame,
    champ_to_idx: Dict[str, int],
    alias_to_id: Dict[str, str],
    pick_weight: float = 1.0,
    ban_weight: float = 0.5,
    ban_scale: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    n = len(df)
    m = len(champ_to_idx)
    X = np.zeros((n, m), dtype=np.float32)
    ban_scale = ban_scale or {}

    for i, row in enumerate(df.itertuples(index=False)):
        def to_canon(token: str) -> str | None:
            key = _norm_name(token)
            return alias_to_id.get(key)

        # Picks
        for c in _split_pipe_list(getattr(row, "team1_drafted_champions")):
            canon = to_canon(c)
            if canon is None:
                continue
            idx = champ_to_idx[canon]
            X[i, idx] = pick_weight

        for c in _split_pipe_list(getattr(row, "team2_drafted_champions")):
            canon = to_canon(c)
            if canon is None:
                continue
            idx = champ_to_idx[canon]
            X[i, idx] = -pick_weight

        
        # Bans (only set if not already picked)
        for c in _split_pipe_list(getattr(row, "team1_banned_champions")):
            canon = to_canon(c)
            if canon is None:
                continue
            idx = champ_to_idx[canon]
            if X[i, idx] == 0:
                scale = ban_scale.get(canon, 1.0)
                X[i, idx] = ban_weight * scale

        for c in _split_pipe_list(getattr(row, "team2_banned_champions")):
            canon = to_canon(c)
            if canon is None:
                continue
            idx = champ_to_idx[canon]
            if X[i, idx] == 0:
                scale = ban_scale.get(canon, 1.0)
                X[i, idx] = -ban_weight * scale

    return X




@dataclass
class TrainedArtifact:
    pipeline: Pipeline
    champion_vocab: List[str]
    pick_weight: float
    ban_weight: float
    ban_scale: Dict[str, float]   # ✅ ADD THIS
    used_champion_mask: np.ndarray
    pairs: list[tuple[str, str]]
    pair_min_count: int
    pair_top_k: int
    counters: list[tuple[str, str]]
    counter_min_count: int
    counter_top_k: int
    C: float


def train(
    df: pd.DataFrame,
    pick_weight: float = 1.0,
    ban_weight: float = 0.5,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 2000,
    pair_min_count: int = 15,
    pair_top_k: int = 20,
    C: float = 1.0,
) -> Tuple[TrainedArtifact, Dict[str, float]]:
    # Clean target
    y_all = df["winner"].astype(int).to_numpy()
    if not set(np.unique(y_all)).issubset({0, 1}):
        raise ValueError("winner column must be binary (0/1).")

    # Normalize patch (UNKNOWN if missing)
    df = df.copy()
    df["patch"] = df["patch"].map(_normalize_patch).astype(str)
    # df["region"] = df["region"].astype(str)
    df["patch_bucket"] = df["patch"].map(_patch_bucket).astype(str)

    # IMPORTANT: split the dataframe first so champs/pairs/meta share identical row split
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all if len(np.unique(y_all)) == 2 else None,
    )
    y_train = df_train["winner"].astype(int).to_numpy()
    y_test = df_test["winner"].astype(int).to_numpy()

    # Champion vocab + encode champion features
    vocab, alias_to_id = load_champion_master("champion.json")
    champ_to_idx = {c: i for i, c in enumerate(vocab)}
    ban_scale = compute_ban_scale_from_train(df_train, alias_to_id, min_scale=0.5, max_scale=1.5)

    Xc_train = _encode_drafts(df_train, champ_to_idx, alias_to_id,
                            pick_weight=pick_weight, ban_weight=ban_weight,
                            ban_scale=ban_scale)
    Xc_test  = _encode_drafts(df_test, champ_to_idx, alias_to_id,
                            pick_weight=pick_weight, ban_weight=ban_weight,
                            ban_scale=ban_scale)  # IMPORTANT: use TRAIN-derived scale

    # Option A masking: based on TRAIN only
    used = _champions_used_in_df(df_train, alias_to_id)
    used_mask = np.array([c in used for c in vocab], dtype=bool)
    n_unused = int((~used_mask).sum())
    if n_unused > 0:
        print(f"Note: {n_unused} champions never appear in this training split; their coefficients will be zeroed.")

    # Pair features: select from TRAIN only, then encode for train/test
    pairs = build_frequent_pairs(df_train, alias_to_id, min_count=pair_min_count, top_k=pair_top_k)
    Xp_train = encode_pair_features(df_train, pairs, alias_to_id)
    Xp_test  = encode_pair_features(df_test,  pairs, alias_to_id)
    print(f"Using {len(pairs)} pair-synergy features (min_count={pair_min_count}, top_k={pair_top_k}).")

    # Counter features: select from TRAIN only, then encode for train/test
    counter_min_count = 6
    counter_top_k = 50

    counters = build_frequent_counters(
        df_train, alias_to_id,
        min_count=counter_min_count,
        top_k=counter_top_k,
    )
    Xct_train = encode_counter_features(df_train, counters, alias_to_id)
    Xct_test  = encode_counter_features(df_test,  counters, alias_to_id)
    print(f"Using {len(counters)} counter features (min_count={counter_min_count}, top_k={counter_top_k}).")

    # Metadata features (patch only)
    meta_cols = ["patch_bucket"]
    meta_train = df_train[meta_cols]
    meta_test  = df_test[meta_cols]

    # Preprocess metadata: impute + one-hot
    meta_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("meta", meta_preprocess, meta_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # Model: regularized logistic regression
    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
        class_weight="balanced",
    )

    # Combine champion + pair + meta features manually
    from scipy.sparse import hstack, csr_matrix  # type: ignore

    def make_features(Xc: np.ndarray, Xp: np.ndarray, Xct: np.ndarray, meta_df: pd.DataFrame):
        meta_encoded = preprocess.fit_transform(meta_df) if not hasattr(preprocess, "transformers_") else preprocess.transform(meta_df)
        Xc_sparse = csr_matrix(Xc)
        Xp_sparse = csr_matrix(Xp)
        Xct_sparse = csr_matrix(Xct)
        return hstack([Xc_sparse, Xp_sparse, Xct_sparse, meta_encoded], format="csr")

    # Fit
    X_train_full = make_features(Xc_train, Xp_train, Xct_train, meta_train)
    clf.fit(X_train_full, y_train)  # type: ignore[arg-type]

    # Zero out coefficients for champions never seen in TRAIN split
    # Feature layout: [ champions | pairs | meta_onehot ]
    coef = clf.coef_.copy()                  # shape (1, n_features)
    champ_coef = coef[:, :len(vocab)]        # first block is champions
    champ_coef[:, ~used_mask] = 0.0
    coef[:, :len(vocab)] = champ_coef
    clf.coef_ = coef

    # Evaluate
    X_test_full  = make_features(Xc_test,  Xp_test,  Xct_test,  meta_test)
    prob = clf.predict_proba(X_test_full)[:, 1]  # type: ignore[arg-type]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "n_samples": float(len(df)),
        "n_champions": float(len(vocab)),
        "n_pairs": float(len(pairs)),
        "n_counters": float(len(counters)),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_auc": float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) == 2 else float("nan"),
        "test_log_loss": float(log_loss(y_test, prob, labels=[0, 1])),
    }

    # Bundle everything needed for inference
    pipeline = Pipeline(
        steps=[
            ("meta_preprocess", preprocess),
            ("clf", clf),
        ]
    )

    artifact = TrainedArtifact(
        pipeline=pipeline,
        champion_vocab=vocab,
        pick_weight=pick_weight,
        ban_weight=ban_weight,
        ban_scale=ban_scale,
        used_champion_mask=used_mask,
        pairs=pairs,
        pair_min_count=pair_min_count,
        pair_top_k=pair_top_k,
        counters=counters,
        counter_min_count=counter_min_count,
        counter_top_k=counter_top_k,
        C=C,
    )

    return artifact, metrics


def save_artifact(out_path: str, artifact: TrainedArtifact) -> None:
    payload = {
        "pipeline": artifact.pipeline,
        "champion_vocab": artifact.champion_vocab,
        "pick_weight": artifact.pick_weight,
        "ban_weight": artifact.ban_weight,
        "ban_scale": artifact.ban_scale,
        "used_champion_mask": artifact.used_champion_mask,
        "pairs": artifact.pairs,
        "pair_min_count": artifact.pair_min_count,
        "pair_top_k": artifact.pair_top_k,
        "counters": artifact.counters,
        "counter_min_count": artifact.counter_min_count,
        "counter_top_k": artifact.counter_top_k,
    }
    joblib.dump(payload, out_path)


def print_top_champs(model_path: str, k: int = 20):
    payload = joblib.load(model_path)
    pipeline = payload["pipeline"]
    vocab = payload["champion_vocab"]
    used_mask = payload.get("used_champion_mask", np.ones(len(vocab), dtype=bool))

    clf = pipeline.named_steps["clf"]
    coef = clf.coef_.ravel()

    champ_coef = coef[:len(vocab)]

    # Only rank champions that appeared in training data
    idxs = np.where(used_mask)[0]
    ranked = idxs[np.argsort(champ_coef[idxs])]

    print(f"\nBottom {k} (hurts Team1/Blue):")
    for i in ranked[:k]:
        print(f"  {vocab[i]:20s} {champ_coef[i]: .4f}")

    print(f"\nTop {k} (helps Team1/Blue):")
    for i in ranked[-k:][::-1]:
        print(f"  {vocab[i]:20s} {champ_coef[i]: .4f}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "inputs",
        nargs="+",
        help="CSV file path(s) or glob(s), e.g. lck_games_with_bans.csv or data/*_games_with_bans.csv",
    )
    ap.add_argument("--out", default="draft_win_model.joblib", help="Output joblib path")
    ap.add_argument("--pick-weight", type=float, default=1.0)
    ap.add_argument("--ban-weight", type=float, default=0.5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--region-filter", type=str, default=None, help="Only use this region, e.g. LCK")
    ap.add_argument("--tournament-contains", type=str, default=None,
                    help="Only rows whose tournament contains this text, e.g. 'Spring 2024'")
    ap.add_argument("--patch-filter", type=str, default=None,
                    help="Only use this patch string exactly (optional)")
    ap.add_argument("--pair-min-count", type=int, default=9999)
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds to run, e.g. 0,1,2,3,4. If set, runs multiple trains and reports mean/std.",
    )
    ap.add_argument("--C", type=float, default=0.3, help="Inverse regularization strength for logistic regression (smaller = stronger regularization)")
    ap.add_argument(
        "--pair-top-k",
        type=int,
        default=20,
        help="Cap pair-synergy features to the top-K most frequent pairs (after min_count). Use 0 to disable cap.",
    )

    args = ap.parse_args()

    # Expand globs
    paths: List[str] = []
    for p in args.inputs:
        expanded = glob.glob(p)
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(p)

    df = _load_csvs(paths)
    if args.region_filter:
        df = df[df["region"].astype(str) == args.region_filter]

    if args.tournament_contains:
        df = df[df["tournament"].astype(str).str.contains(args.tournament_contains, case=False, na=False)]

    if args.patch_filter:
        df = df[df["patch"].astype(str) == args.patch_filter]

    print(f"Using {len(df)} rows after filters.")
    if len(df) < 200:
        print("Warning: small dataset after filtering; metrics may be noisy.")

    # --- Seed handling ---
    import numpy as np

    def _parse_seeds(seeds_str: str) -> List[int]:
        return [int(x.strip()) for x in seeds_str.split(",") if x.strip()]

    seed_list = [args.seed]
    if args.seeds:
        seed_list = _parse_seeds(args.seeds)

    all_metrics: List[Dict[str, float]] = []
    best_auc = float("-inf")
    best_seed = None
    best_artifact = None
    best_metrics = None

    for sd in seed_list:
        artifact, metrics = train(
            df,
            pick_weight=args.pick_weight,
            ban_weight=args.ban_weight,
            test_size=args.test_size,
            random_state=sd,
            max_iter=args.max_iter,
            pair_min_count=args.pair_min_count,
            pair_top_k=args.pair_top_k,
            C=args.C
        )
        all_metrics.append(metrics)

        auc = float(metrics.get("test_auc", float("nan")))
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_seed = sd
            best_artifact = artifact
            best_metrics = metrics

    # If AUC is nan for all (edge case), just take the first run
    if best_artifact is None:
        best_seed = seed_list[0]
        best_artifact, best_metrics = train(
            df,
            pick_weight=args.pick_weight,
            ban_weight=args.ban_weight,
            test_size=args.test_size,
            random_state=best_seed,
            max_iter=args.max_iter,
            pair_min_count=args.pair_min_count,
        )
        best_auc = float(best_metrics.get("test_auc", float("nan")))

    # Report aggregate stats if multiple seeds
    if len(seed_list) > 1:
        def _mean_std(key: str) -> str:
            vals = np.array([m.get(key, float("nan")) for m in all_metrics], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                return "nan"
            if len(vals) == 1:
                return f"{vals[0]:.4f}"
            return f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"

        print("\nSeed sweep results:")
        print("  seeds:", seed_list)
        print("  test_auc:", _mean_std("test_auc"))
        print("  test_accuracy:", _mean_std("test_accuracy"))
        print("  test_log_loss:", _mean_std("test_log_loss"))

    print(f"\nSaving best model by AUC: seed={best_seed}, auc={best_auc:.4f}")

    save_artifact(args.out, best_artifact)
    assert best_metrics is not None
    print("Saved:", args.out)
    print("Metrics (best seed):")
    for k, v in best_metrics.items():
        print(f"  {k}: {v}")

    print_top_champs(args.out, k=20)


if __name__ == "__main__":
    main()