"""
analysis/compare_modes.py — Compare MSE vs Lambdarank vs Ensemble model modes.

Reads predictor/metrics/mode_history.json from S3 and produces a summary
report showing which mode wins most often and IC trends over time.

Usage:
    python -m analysis.compare_modes
    python -m analysis.compare_modes --csv mode_comparison.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

MODE_HISTORY_KEY = "predictor/metrics/mode_history.json"


def load_mode_history() -> list[dict]:
    """Download mode_history.json from S3."""
    import boto3

    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=cfg.S3_BUCKET, Key=MODE_HISTORY_KEY)
        data = json.loads(obj["Body"].read())
        if isinstance(data, list):
            return data
    except Exception as e:
        log.error("Failed to load mode history from S3: %s", e)
    return []


def summarize(history: list[dict]) -> str:
    """Produce a markdown summary report."""
    if not history:
        return "# Mode Comparison Report\n\nNo mode history data available.\n"

    lines = ["# Mode Comparison Report", ""]
    lines.append(f"**Entries:** {len(history)}")
    dates = [e["date"] for e in history]
    lines.append(f"**Range:** {dates[0]} to {dates[-1]}")
    lines.append("")

    # Mode win counts
    wins = Counter(e["best_mode"] for e in history)
    lines.append("## Mode Selection Frequency")
    lines.append("")
    for mode, count in wins.most_common():
        pct = count / len(history) * 100
        lines.append(f"- **{mode}**: {count} ({pct:.0f}%)")
    lines.append("")

    # Average IC by mode (across all weeks, not just when that mode won)
    lines.append("## Average IC Across All Weeks")
    lines.append("")
    mse_ics = [e["mse_ic"] for e in history if e.get("mse_ic") is not None]
    rank_ics = [e["rank_ic"] for e in history if e.get("rank_ic") is not None]
    ens_ics = [e["ensemble_ic"] for e in history if e.get("ensemble_ic") is not None]

    if mse_ics:
        lines.append(f"- **MSE**: {sum(mse_ics)/len(mse_ics):.4f} (n={len(mse_ics)})")
    if rank_ics:
        lines.append(f"- **Rank**: {sum(rank_ics)/len(rank_ics):.4f} (n={len(rank_ics)})")
    if ens_ics:
        lines.append(f"- **Ensemble**: {sum(ens_ics)/len(ens_ics):.4f} (n={len(ens_ics)})")
    lines.append("")

    # IC delta trends
    deltas_rank = [e["ic_delta_rank_vs_mse"] for e in history if e.get("ic_delta_rank_vs_mse") is not None]
    deltas_ens = [e["ic_delta_ens_vs_mse"] for e in history if e.get("ic_delta_ens_vs_mse") is not None]

    if deltas_rank:
        avg_delta_rank = sum(deltas_rank) / len(deltas_rank)
        lines.append("## IC Deltas vs MSE Baseline")
        lines.append("")
        lines.append(f"- **Rank vs MSE**: avg delta = {avg_delta_rank:+.4f}")
    if deltas_ens:
        avg_delta_ens = sum(deltas_ens) / len(deltas_ens)
        lines.append(f"- **Ensemble vs MSE**: avg delta = {avg_delta_ens:+.4f}")
    lines.append("")

    # Weekly detail table
    lines.append("## Weekly Detail")
    lines.append("")
    lines.append("| Date | MSE IC | Rank IC | Ensemble IC | Best Mode | Promoted | IC IR |")
    lines.append("|------|--------|---------|-------------|-----------|----------|-------|")
    for e in history:
        rank_str = f"{e['rank_ic']:.4f}" if e.get("rank_ic") is not None else "—"
        ens_str = f"{e['ensemble_ic']:.4f}" if e.get("ensemble_ic") is not None else "—"
        promo = "yes" if e.get("promoted") else "no"
        lines.append(
            f"| {e['date']} | {e['mse_ic']:.4f} | {rank_str} | {ens_str} "
            f"| {e['best_mode']} | {promo} | {e.get('ic_ir', 0):.2f} |"
        )
    lines.append("")

    return "\n".join(lines)


def export_csv(history: list[dict], path: str):
    """Write mode history to CSV."""
    import csv

    if not history:
        log.warning("No data to export")
        return

    fields = ["date", "mse_ic", "rank_ic", "ensemble_ic", "best_mode",
              "promoted", "ic_delta_rank_vs_mse", "ic_delta_ens_vs_mse",
              "n_train", "ic_ir"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    log.info("Exported %d rows to %s", len(history), path)


def main():
    parser = argparse.ArgumentParser(description="Compare GBM mode selection history")
    parser.add_argument("--csv", type=str, help="Export to CSV file")
    args = parser.parse_args()

    history = load_mode_history()
    if not history:
        print("No mode history found on S3. Run at least one training cycle first.")
        return

    report = summarize(history)
    print(report)

    if args.csv:
        export_csv(history, args.csv)


if __name__ == "__main__":
    main()
