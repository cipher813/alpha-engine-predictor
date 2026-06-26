"""scripts/momentum_ic_study_21d.py — per-feature IC study at 21d horizon.

One-shot diagnostic: for each candidate momentum feature, computes Pearson
+ Spearman IC against y_fwd[te] across the same 16 walk-forward folds the
training pipeline uses (`training/meta_trainer.py` line 1117-1171).

Sidesteps the ArcticDB dependency by recomputing momentum features directly
from the legacy OHLCV parquet cache at
`s3://alpha-engine-research/predictor/price_cache/`. Feature formulas mirror
`alpha-engine-data/features/feature_engineer.py`.

Output:
  - markdown table to stdout (median Pearson IC, median Spearman IC, IQR,
    % positive folds, n_folds, n_rows)
  - JSON dump to `~/Development/alpha-engine-docs/private/momentum_ic_study_21d.json`

Runtime: ~10-15 min on a Mac, dominated by 900 S3 downloads. Memory ~700MB
peak (one parquet at a time during compute, single concatenated float32
panel at the end). No model loading, no torch — safe to run locally.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BUCKET = "alpha-engine-research"
# Wave-3 PR4 cutover (DONE): price_cache parquets live at
# ``reference/price_cache/`` only — the legacy ``predictor/price_cache/`` tree
# is removed live via ``aws s3 rm``, so the fallback entry is dropped.
PRICE_CACHE_PREFIXES: tuple[str, ...] = (
    "reference/price_cache/",
)
# sector_map.json is co-located inside the price_cache prefix and arrives
# as part of the paginator-driven sync; the read in ``main()`` just opens
# the local cached copy.

FORWARD_DAYS = 21
LABEL_CLIP = 0.40

WF_TEST_WINDOW_DAYS = 126
WF_MIN_TRAIN_DAYS = 504
WF_PURGE_DAYS = 21

# Skip rows shorter than ~1y of history (matches _MIN_HISTORY_ROWS=265)
MIN_HISTORY_ROWS = 265

CANDIDATE_FEATURES = [
    "momentum_5d",      # 5d return — drop candidate (user explicit)
    "momentum_20d",     # 20d return — reversal-zone at 21d forecast
    "return_60d",       # 3m return — keep candidate
    "return_120d",      # 6m return — keep candidate
    "price_vs_ma50",    # ~2.5m trend
    "price_vs_ma200",   # ~10m trend
    "rsi_14",           # 14d momentum oscillator
    # Composite: classical Jegadeesh-Titman style with available windows
    # = 120d return minus most-recent 20d (skip-month variant)
    "ret120_minus_ret20",
]

LOCAL_CACHE = Path("/tmp/momentum_ic_study_parquets")
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute momentum features from OHLCV. Mirrors
    `alpha-engine-data/features/feature_engineer.py`."""
    close = df["Close"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["momentum_5d"] = (close / close.shift(5)) - 1.0
    out["momentum_20d"] = (close / close.shift(20)) - 1.0
    out["return_60d"] = (close / close.shift(60)) - 1.0
    out["return_120d"] = (close / close.shift(120)) - 1.0
    out["price_vs_ma50"] = (close / close.rolling(50).mean()) - 1.0
    out["price_vs_ma200"] = (close / close.rolling(200).mean()) - 1.0
    # Wilder RSI 14
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    # Composite: 6m return minus most-recent 20d — classical 12-1 momentum
    # surrogate with windows available today.
    out["ret120_minus_ret20"] = out["return_120d"] - out["momentum_20d"]
    return out


def compute_21d_forward_alpha(stock_close: pd.Series, bench_close: pd.Series) -> pd.Series:
    """Sector-neutral 21d forward SIMPLE return. Matches
    `data/label_generator.compute_labels` with `forward_days=21`."""
    stock_fwd = (stock_close.shift(-FORWARD_DAYS) / stock_close) - 1.0
    bench_aligned = bench_close.reindex(stock_close.index, method="ffill")
    bench_fwd = (bench_aligned.shift(-FORWARD_DAYS) / bench_aligned) - 1.0
    return (stock_fwd - bench_fwd).clip(-LABEL_CLIP, LABEL_CLIP)


def _download_one(s3, key: str) -> Path:
    """Download a parquet to LOCAL_CACHE if not already present."""
    fname = key.split("/")[-1]
    dest = LOCAL_CACHE / fname
    if not dest.exists():
        s3.download_file(BUCKET, key, str(dest))
    return dest


def download_price_cache() -> dict[str, Path]:
    """Sync all *.parquet under the price_cache read prefixes to LOCAL_CACHE.

    Iterates ``PRICE_CACHE_PREFIXES`` (new → legacy) and dedupes by
    ``{basename}.parquet`` so each ticker is fetched once. Returns
    ``{ticker: path}``.
    """
    s3 = boto3.client("s3", region_name="us-east-1")
    keys: list[str] = []
    seen_basenames: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for prefix in PRICE_CACHE_PREFIXES:
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if not k.endswith(".parquet"):
                    continue
                basename = k.rsplit("/", 1)[-1]
                if basename in seen_basenames:
                    continue
                seen_basenames.add(basename)
                keys.append(k)
    print(f"Found {len(keys)} parquets, syncing to {LOCAL_CACHE}...", file=sys.stderr)

    t0 = time.time()

    def _dl(k):
        try:
            return _download_one(s3, k), None
        except Exception as e:
            return k, str(e)

    paths: dict[str, Path] = {}
    n_done = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_dl, k): k for k in keys}
        for fut in as_completed(futs):
            result, err = fut.result()
            if err:
                print(f"  download error: {result}: {err}", file=sys.stderr)
                continue
            ticker = result.stem
            paths[ticker] = result
            n_done += 1
            if n_done % 100 == 0:
                print(f"  ...{n_done}/{len(keys)} ({time.time()-t0:.0f}s)", file=sys.stderr)
    print(f"Sync done: {n_done} parquets in {time.time()-t0:.0f}s", file=sys.stderr)
    return paths


def main():
    # Step 1: sync data
    paths = download_price_cache()
    sector_map = json.load(open(LOCAL_CACHE / "sector_map.json"))

    # Build benchmark cache (sector ETFs + SPY fallback)
    sector_etfs = sorted({v for v in sector_map.values()})
    bench_close: dict[str, pd.Series] = {}
    for etf in sector_etfs + ["SPY"]:
        p = paths.get(etf)
        if p is None or not p.exists():
            print(f"WARN: benchmark {etf} missing", file=sys.stderr)
            continue
        df = pd.read_parquet(p)
        if "Close" in df.columns:
            bench_close[etf] = df["Close"].astype(float)
    print(f"Loaded {len(bench_close)} benchmark series", file=sys.stderr)

    # Step 2: per-ticker labeled + featured panel — stream, accumulate float32 chunks
    date_chunks: list[np.ndarray] = []
    ticker_chunks: list[np.ndarray] = []
    feature_chunks: list[np.ndarray] = []
    fwd_chunks: list[np.ndarray] = []

    n_skipped_no_sector = 0
    n_skipped_short = 0
    n_skipped_missing_cols = 0
    n_accepted = 0

    universe_tickers = sorted(t for t in paths if t not in sector_etfs and t != "SPY")
    t0 = time.time()
    for i, ticker in enumerate(universe_tickers):
        p = paths[ticker]
        sector = sector_map.get(ticker)
        bench = bench_close.get(sector) if sector else bench_close.get("SPY")
        if bench is None:
            n_skipped_no_sector += 1
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if len(df) < MIN_HISTORY_ROWS:
            n_skipped_short += 1
            continue
        if "Close" not in df.columns:
            n_skipped_missing_cols += 1
            continue

        feats = compute_features(df)
        y_fwd = compute_21d_forward_alpha(df["Close"].astype(float), bench)

        # Align + drop NaN-on-label rows (~last 21 rows of every series)
        combined = feats.copy()
        combined["__y_fwd__"] = y_fwd
        # Need ALL candidate features present for a row to count.
        # (return_120d's 120-day warmup dominates — first ~120 rows dropped.)
        combined = combined.dropna(subset=CANDIDATE_FEATURES + ["__y_fwd__"])
        if combined.empty:
            continue

        n = len(combined)
        date_chunks.append(combined.index.to_numpy())
        ticker_chunks.append(np.full(n, ticker, dtype=object))
        feature_chunks.append(
            combined[CANDIDATE_FEATURES].to_numpy(dtype=np.float32)
        )
        fwd_chunks.append(combined["__y_fwd__"].to_numpy(dtype=np.float32))
        n_accepted += 1

        if (i + 1) % 100 == 0:
            print(
                f"  ticker {i+1}/{len(universe_tickers)} accepted={n_accepted} "
                f"({time.time()-t0:.0f}s)",
                file=sys.stderr,
            )

    print(
        f"Universe: accepted={n_accepted} skipped_short={n_skipped_short} "
        f"skipped_no_sector={n_skipped_no_sector} "
        f"skipped_missing_cols={n_skipped_missing_cols}",
        file=sys.stderr,
    )

    # Step 3: stitch + sort by date (matches meta_trainer.py line 813-825)
    all_dates = np.concatenate(date_chunks)
    all_tickers = np.concatenate(ticker_chunks)
    X = np.concatenate(feature_chunks, axis=0)
    y_fwd = np.concatenate(fwd_chunks)

    sort_idx = np.argsort(all_dates, kind="mergesort")
    all_dates = all_dates[sort_idx]
    all_tickers = all_tickers[sort_idx]
    X = X[sort_idx]
    y_fwd = y_fwd[sort_idx]

    print(f"Stitched panel: N={len(y_fwd):,} rows, F={X.shape[1]} features", file=sys.stderr)

    # Step 4: build folds — exactly matches meta_trainer.py 1056-1090
    # `np.unique` preserves datetime64 dtype (vs sorted(set(...)) which can
    # round-trip through Python datetime + lose dtype on np.array(...)).
    unique_dates = np.unique(all_dates)
    n_unique = len(unique_dates)
    folds = []
    fold_start_idx = WF_MIN_TRAIN_DAYS
    while fold_start_idx < n_unique:
        remaining = n_unique - fold_start_idx
        if remaining < WF_TEST_WINDOW_DAYS // 2:
            break
        test_start_date = unique_dates[fold_start_idx]
        test_end_idx = min(fold_start_idx + WF_TEST_WINDOW_DAYS - 1, n_unique - 1)
        test_end_date = unique_dates[test_end_idx]
        train_end_idx = fold_start_idx - WF_PURGE_DAYS
        if train_end_idx < WF_MIN_TRAIN_DAYS // 2:
            fold_start_idx += WF_TEST_WINDOW_DAYS
            continue
        # train_end_date = unique_dates[train_end_idx]  # unused for IC study
        test_mask = (all_dates >= test_start_date) & (all_dates <= test_end_date)
        test_idx = np.where(test_mask)[0]
        if len(test_idx) < 100:
            fold_start_idx += WF_TEST_WINDOW_DAYS
            continue
        folds.append({
            "test_idx": test_idx,
            "test_start": str(test_start_date)[:10],
            "test_end": str(test_end_date)[:10],
            "n_test": int(len(test_idx)),
        })
        fold_start_idx += WF_TEST_WINDOW_DAYS

    print(f"Built {len(folds)} walk-forward folds", file=sys.stderr)

    # Step 5: per-feature × per-fold IC
    per_feature_results = {}
    for fi, fname in enumerate(CANDIDATE_FEATURES):
        col = X[:, fi].astype(np.float64)
        fold_pearsons = []
        fold_spearmans = []
        for fold in folds:
            te = fold["test_idx"]
            xf = col[te]
            yf = y_fwd[te]
            mask = np.isfinite(xf) & np.isfinite(yf)
            if mask.sum() < 100:
                fold_pearsons.append(np.nan)
                fold_spearmans.append(np.nan)
                continue
            xf2, yf2 = xf[mask], yf[mask]
            if np.std(xf2) < 1e-10 or np.std(yf2) < 1e-10:
                fold_pearsons.append(0.0)
                fold_spearmans.append(0.0)
                continue
            fold_pearsons.append(float(np.corrcoef(xf2, yf2)[0, 1]))
            fold_spearmans.append(float(spearmanr(xf2, yf2)[0]))

        p_arr = np.array(fold_pearsons)
        s_arr = np.array(fold_spearmans)
        finite_p = p_arr[np.isfinite(p_arr)]
        finite_s = s_arr[np.isfinite(s_arr)]

        per_feature_results[fname] = {
            "median_pearson": float(np.median(finite_p)) if len(finite_p) else None,
            "median_spearman": float(np.median(finite_s)) if len(finite_s) else None,
            "pearson_q25": float(np.percentile(finite_p, 25)) if len(finite_p) else None,
            "pearson_q75": float(np.percentile(finite_p, 75)) if len(finite_p) else None,
            "spearman_q25": float(np.percentile(finite_s, 25)) if len(finite_s) else None,
            "spearman_q75": float(np.percentile(finite_s, 75)) if len(finite_s) else None,
            "pct_positive_pearson": float(np.mean(finite_p > 0) * 100) if len(finite_p) else None,
            "pct_positive_spearman": float(np.mean(finite_s > 0) * 100) if len(finite_s) else None,
            "fold_pearsons": [float(x) for x in p_arr.tolist()],
            "fold_spearmans": [float(x) for x in s_arr.tolist()],
        }

    # Step 6: feature-feature correlation matrix (training set only — pre-last-fold)
    if folds:
        # last fold's test_start is a string "YYYY-MM-DD"; coerce to ns dtype
        # consistent with all_dates (datetime64[ns]).
        pre_cutoff = np.datetime64(folds[-1]["test_start"]).astype("datetime64[ns]")
        pre_mask = all_dates < pre_cutoff
        Xpre = X[pre_mask]
        corr_matrix = pd.DataFrame(Xpre, columns=CANDIDATE_FEATURES).corr().round(3)
    else:
        corr_matrix = pd.DataFrame()

    # Step 7: print markdown table
    print()
    print("# Momentum IC Study — 21d horizon (sector-neutral simple return)")
    print(f"_N={len(y_fwd):,} rows, {len(folds)} walk-forward folds_")
    print(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_")
    print()
    print("## Per-feature IC")
    print()
    print("| Feature | Median Pearson | Median Spearman | Pearson IQR | %+folds (Pearson) |")
    print("|---|---:|---:|---|---:|")
    for fname, r in per_feature_results.items():
        mp = r["median_pearson"]
        ms = r["median_spearman"]
        q25 = r["pearson_q25"]
        q75 = r["pearson_q75"]
        pct = r["pct_positive_pearson"]
        print(
            f"| `{fname}` | {mp:+.4f} | {ms:+.4f} | "
            f"[{q25:+.4f}, {q75:+.4f}] | {pct:.0f}% |"
        )

    print()
    print("## Feature-feature correlation (training rows only)")
    print()
    print("```")
    print(corr_matrix.to_string())
    print("```")

    print()
    print("## Per-fold momentum_5d IC (sanity check vs production walk-forward)")
    print()
    print("| Fold | Test Start | Test End | momentum_5d Pearson IC |")
    print("|---|---|---|---:|")
    for i, fold in enumerate(folds):
        p = per_feature_results["momentum_5d"]["fold_pearsons"][i]
        print(f"| {i+1} | {fold['test_start']} | {fold['test_end']} | {p:+.4f} |")

    # Step 8: persist JSON
    out_dir = Path(os.path.expanduser("~/Development/alpha-engine-docs/private"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "momentum_ic_study_21d.json"
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "forward_days": FORWARD_DAYS,
        "label_clip": LABEL_CLIP,
        "n_rows": int(len(y_fwd)),
        "n_folds": len(folds),
        "n_tickers_accepted": int(n_accepted),
        "wf_geometry": {
            "test_window_days": WF_TEST_WINDOW_DAYS,
            "min_train_days": WF_MIN_TRAIN_DAYS,
            "purge_days": WF_PURGE_DAYS,
        },
        "per_feature": per_feature_results,
        "fold_meta": folds,
        "correlation_matrix": corr_matrix.to_dict() if not corr_matrix.empty else {},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults written to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
