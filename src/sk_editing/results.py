"""Mappings and aggregation helpers for the archived result schemas."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

FLUX_METHOD_FIELDS = {
    "Simple": "simple",
    "LLM-only": "llm",
    "SK+Filter": "kg",
    "SK+LLM": "kg_nofilter",
}

IP2P_METHOD_FIELDS = {
    "Simple": "simple",
    "MGIE": "mgie_style",
    "LLM-only": "llm_only",
    "SK+Filter": "kg_llm",
    "SK+LLM": "kg_llm_nofilter",
}

METRICS = ("clip_dir_common", "clip_out", "clip_im", "l1", "dino")


def _mean(values):
    values = list(values)
    if not values:
        raise ValueError("Cannot compute the mean of an empty sequence")
    return sum(values) / len(values)


def _population_std(values):
    values = list(values)
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def valid(value: Any) -> bool:
    return value is not None and not (
        isinstance(value, float) and math.isnan(value)
    )


def flux_sample_means(rows: Iterable[dict[str, Any]], field: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if valid(value):
            values[row["hash"]].append(float(value))
    return {sample_hash: _mean(sample_values) for sample_hash, sample_values in values.items()}


def flux_seed_summary(rows: list[dict[str, Any]], prefix: str, metric: str) -> tuple[float, float, int]:
    field = f"{prefix}_{metric}"
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if valid(value):
            by_seed[int(row["seed"])].append(float(value))
    seed_means = [_mean(values) for _, values in sorted(by_seed.items()) if values]
    return _mean(seed_means), _population_std(seed_means), sum(len(values) for values in by_seed.values())


def ip2p_values(rows: Iterable[dict[str, Any]], prefix: str, metric: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for row in rows:
        if metric == "clip_dir_common":
            value = row.get(f"{prefix}_clip_dir_common")
        else:
            value = row.get(prefix, {}).get(metric)
        if valid(value):
            output[row["hash"]] = float(value)
    return output
