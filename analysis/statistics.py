"""Recompute the paired statistics reported for FLUX and IP2P.

FLUX scores are first averaged across seeds for each sample, then paired tests
are performed over samples. IP2P uses one seed and is tested directly over
samples. The matched-subset analysis controls sample composition only; it does
not isolate the editing backbone because the two evaluations used editor-
specific scene-description and instruction-generation configurations.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from sk_editing.io import load_json
from sk_editing.results import flux_sample_means, ip2p_values

N_BOOT = 10_000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class PairedTest:
    section: str
    metric: str
    comparison: str
    n: int
    mean_difference: float
    ci_low: float
    ci_high: float
    wilcoxon_p: float


def bootstrap_ci(differences: np.ndarray, *, n_boot: int = N_BOOT) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Keep the original resampling order so the released confidence intervals
    # reproduce the archived analysis exactly.
    bootstrap_means = [
        np.mean(rng.choice(differences, len(differences), replace=True))
        for _ in range(n_boot)
    ]
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(low), float(high)


def paired_test(
    section: str,
    metric: str,
    comparison: str,
    left: dict[str, float],
    right: dict[str, float],
) -> PairedTest:
    keys = [key for key in left if key in right]
    if not keys:
        raise ValueError(f"No paired samples for {section}: {comparison} ({metric})")
    differences = np.asarray([left[key] - right[key] for key in keys], dtype=float)
    low, high = bootstrap_ci(differences)
    if np.allclose(differences, 0):
        p_value = 1.0
    else:
        p_value = float(stats.wilcoxon(differences).pvalue)
    return PairedTest(
        section=section,
        metric=metric,
        comparison=comparison,
        n=len(keys),
        mean_difference=float(differences.mean()),
        ci_low=low,
        ci_high=high,
        wilcoxon_p=p_value,
    )


def calculate(flux_rows: list[dict], ip2p_rows: list[dict]) -> list[PairedTest]:
    output: list[PairedTest] = []

    for metric in ("clip_dir_common", "clip_out"):
        values = {
            key: flux_sample_means(flux_rows, f"{key}_{metric}")
            for key in ("simple", "llm", "kg", "kg_nofilter")
        }
        output.extend(
            [
                paired_test(
                    "FLUX", metric, "SK+LLM vs LLM-only",
                    values["kg_nofilter"], values["llm"],
                ),
                paired_test(
                    "FLUX", metric, "SK+LLM vs Simple",
                    values["kg_nofilter"], values["simple"],
                ),
                paired_test(
                    "FLUX", metric, "SK+LLM vs SK+Filter",
                    values["kg_nofilter"], values["kg"],
                ),
            ]
        )

    labels = {
        "simple": "Simple",
        "mgie_style": "MGIE",
        "llm_only": "LLM-only",
        "kg_llm": "SK+Filter",
        "kg_llm_nofilter": "SK+LLM",
    }
    for metric in ("clip_dir_common", "clip_out"):
        cached = {key: ip2p_values(ip2p_rows, key, metric) for key in labels}
        for first, second in itertools.combinations(labels, 2):
            # Orient comparisons involving the proposed method as SK+LLM minus baseline,
            # matching the paper text; retain the natural order for all other pairs.
            if second == "kg_llm_nofilter":
                left, right = second, first
            else:
                left, right = first, second
            output.append(
                paired_test(
                    "IP2P",
                    metric,
                    f"{labels[left]} vs {labels[right]}",
                    cached[left],
                    cached[right],
                )
            )

    shared = {row["hash"] for row in flux_rows} & {row["hash"] for row in ip2p_rows}
    flux_shared = [row for row in flux_rows if row["hash"] in shared]
    flux_sk = flux_sample_means(flux_shared, "kg_nofilter_clip_dir_common")
    flux_llm = flux_sample_means(flux_shared, "llm_clip_dir_common")
    ip2p_sk = {
        key: value
        for key, value in ip2p_values(ip2p_rows, "kg_llm_nofilter", "clip_dir_common").items()
        if key in shared
    }
    ip2p_llm = {
        key: value
        for key, value in ip2p_values(ip2p_rows, "llm_only", "clip_dir_common").items()
        if key in shared
    }
    output.extend(
        [
            paired_test(
                "Matched subset", "clip_dir_common", "FLUX: SK+LLM vs LLM-only",
                flux_sk, flux_llm,
            ),
            paired_test(
                "Matched subset", "clip_dir_common", "IP2P: SK+LLM vs LLM-only",
                ip2p_sk, ip2p_llm,
            ),
        ]
    )
    return output


def print_report(tests: list[PairedTest]) -> None:
    current: tuple[str, str] | None = None
    for test in tests:
        heading = (test.section, test.metric)
        if heading != current:
            print("\n" + "=" * 98)
            print(f"{test.section} — {test.metric}")
            print("=" * 98)
            current = heading
        p_text = "<0.000001" if test.wilcoxon_p < 0.000001 else f"={test.wilcoxon_p:.6f}"
        print(
            f"{test.comparison:<42} N={test.n:>4}  "
            f"diff={test.mean_difference:+.4f}  "
            f"95% CI [{test.ci_low:+.4f}, {test.ci_high:+.4f}]  p{p_text}"
        )


def write_csv(path: Path, tests: list[PairedTest]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(tests[0])))
        writer.writeheader()
        writer.writerows(asdict(test) for test in tests)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flux", type=Path, default=Path("results/raw/flux_results.json")
    )
    parser.add_argument(
        "--ip2p", type=Path, default=Path("results/raw/ip2p_results.json")
    )
    parser.add_argument("--csv", type=Path, help="Optional CSV output path")
    parser.add_argument("--json", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tests = calculate(load_json(args.flux), load_json(args.ip2p))
    print_report(tests)
    if args.csv:
        write_csv(args.csv, tests)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([asdict(test) for test in tests], indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
