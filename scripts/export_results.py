"""Export readable CSV summaries from the archived sample-level result files."""

from __future__ import annotations

import argparse
import sys
import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from statistics import fmean

from analysis.statistics import calculate, write_csv
from sk_editing.io import load_json
from sk_editing.results import (
    FLUX_METHOD_FIELDS,
    IP2P_METHOD_FIELDS,
    METRICS,
    flux_sample_means,
    flux_seed_summary,
    ip2p_values,
)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def overall_rows(flux: list[dict], ip2p: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for label, prefix in FLUX_METHOD_FIELDS.items():
        row = {"backbone": "FLUX Kontext", "method": label}
        for metric in METRICS:
            mean, std, _ = flux_seed_summary(flux, prefix, metric)
            row[metric] = f"{mean:.6f}"
            row[f"{metric}_seed_std"] = f"{std:.6f}"
        rows.append(row)
    for label, prefix in IP2P_METHOD_FIELDS.items():
        row = {"backbone": "InstructPix2Pix", "method": label}
        for metric in METRICS:
            values = list(ip2p_values(ip2p, prefix, metric).values())
            row[metric] = f"{fmean(values):.6f}"
            row[f"{metric}_seed_std"] = ""
        rows.append(row)
    return rows


def condition_rows(flux: list[dict], ip2p: list[dict]) -> list[dict]:
    rows: list[dict] = []
    conditions = ("rainy", "snowy", "foggy", "night", "clear")
    for condition in conditions:
        f_subset = [row for row in flux if row["kg_condition"] == condition]
        i_subset = [row for row in ip2p if row["kg_condition"] == condition]
        for label, prefix in FLUX_METHOD_FIELDS.items():
            values = flux_sample_means(f_subset, f"{prefix}_clip_dir_common")
            rows.append(
                {
                    "backbone": "FLUX Kontext",
                    "condition": condition,
                    "method": label,
                    "n": len(values),
                    "clip_dir_common": f"{fmean(values.values()):.6f}" if values else "",
                }
            )
        for label, prefix in IP2P_METHOD_FIELDS.items():
            values = ip2p_values(i_subset, prefix, "clip_dir_common")
            rows.append(
                {
                    "backbone": "InstructPix2Pix",
                    "condition": condition,
                    "method": label,
                    "n": len(values),
                    "clip_dir_common": f"{fmean(values.values()):.6f}" if values else "",
                }
            )
    return rows


def matched_rows(flux: list[dict], ip2p: list[dict]) -> list[dict]:
    shared = sorted({row["hash"] for row in flux} & {row["hash"] for row in ip2p})
    valid_flux = flux_sample_means(
        [row for row in flux if row["hash"] in shared], "simple_clip_dir_common"
    )
    valid_ip2p = {
        key: value
        for key, value in ip2p_values(ip2p, "simple", "clip_dir_common").items()
        if key in shared
    }
    return [
        {
            "shared_samples": len(shared),
            "flux_clipdir_valid": len(valid_flux),
            "ip2p_clipdir_valid": len(valid_ip2p),
            "note": (
                "Controls sample composition only; editor-specific VLM and prompt "
                "configurations differ between backbone evaluations."
            ),
        }
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux", type=Path, default=Path("results/raw/flux_results.json"))
    parser.add_argument("--ip2p", type=Path, default=Path("results/raw/ip2p_results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flux = load_json(args.flux)
    ip2p = load_json(args.ip2p)
    write_rows(args.output_dir / "overall_metrics.csv", overall_rows(flux, ip2p))
    write_rows(args.output_dir / "condition_clipdir.csv", condition_rows(flux, ip2p))
    write_rows(args.output_dir / "matched_subset.csv", matched_rows(flux, ip2p))
    write_csv(args.output_dir / "statistical_tests.csv", calculate(flux, ip2p))
    print(f"Wrote result summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
