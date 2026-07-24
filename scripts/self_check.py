"""Run deterministic CPU-side integrity checks for the public repository."""

from __future__ import annotations

import compileall
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistics import calculate  # noqa: E402
from sk_editing.io import load_json  # noqa: E402
from sk_editing.prompts import FLUX_PROMPTS, IP2P_PROMPTS  # noqa: E402
from sk_editing.results import (  # noqa: E402
    FLUX_METHOD_FIELDS,
    IP2P_METHOD_FIELDS,
    flux_sample_means,
    flux_seed_summary,
    ip2p_values,
)
from sk_editing.retrieval import StructuredRepository  # noqa: E402


class CheckFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def close(actual: float, expected: float, tolerance: float = 5e-4) -> bool:
    return math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0.0)


def check_required_files() -> None:
    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "pyproject.toml",
        ".gitignore",
        ".gitattributes",
        "assets/framework.png",
        "assets/clipdir_results.png",
        "data/structured_repository.json",
        "data/clipout_captions.json",
        "data/emu_edit_weather_final.json",
        "data/flux_sample_ids.json",
        "data/ip2p_sample_ids.json",
        "docs/experiment_protocol.md",
        "docs/prompts.md",
        "docs/reproducibility.md",
        "results/raw/flux_results.json",
        "results/raw/ip2p_results.json",
        "results/overall_metrics.csv",
        "results/condition_clipdir.csv",
        "results/statistical_tests.csv",
        "results/matched_subset.csv",
        "analysis/statistics.py",
        "scripts/generate_flux.py",
        "scripts/generate_ip2p.py",
        "scripts/recompute_metrics.py",
        "scripts/export_results.py",
        "scripts/self_check.py",
        "src/sk_editing/retrieval.py",
        "src/sk_editing/prompts.py",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    require(not missing, f"Missing required files: {missing}")

    forbidden_names = {
        "run_gen_data1_v6.py",
        "run_comparison_ip2p_v4.py",
        "_make_fig3_original.py",
        "migrate_from_workspace.py",
    }
    present = [path.name for path in ROOT.rglob("*") if path.is_file() and path.name in forbidden_names]
    require(not present, f"Legacy workspace files remain: {present}")


def check_repository_data() -> None:
    repository = StructuredRepository.from_path(ROOT / "data/structured_repository.json")
    stats = repository.statistics()
    require(stats["groups"] == 5, f"Expected 5 repository groups, found {stats['groups']}")
    require(stats["entries"] == 21, f"Expected 21 entries, found {stats['entries']}")
    require(stats["cues"] == 498, f"Expected 498 cues, found {stats['cues']}")
    expected_slots = {
        "global",
        "lighting",
        "surfaces",
        "atmospheric_effects",
        "objects_details",
    }
    missing_slots = expected_slots - set(stats["slots"])
    require(expected_slots.issubset(stats["slots"]), f"Missing typed slots: {missing_slots}")

    example_requests = (
        "make it rainy",
        "make it snowy",
        "make it foggy",
        "turn day into night",
        "make the weather clear",
    )
    for request in example_requests:
        result = repository.retrieve(request)
        require(result.matched_entries, f"No repository match for request: {request}")
        require(result.cues, f"No cues retrieved for request: {request}")
        require(len(result.cues) <= 50, f"Cue cap exceeded for request: {request}")

    captions = load_json(ROOT / "data/clipout_captions.json")["captions"]
    require(
        set(captions) == {"rainy", "snowy", "foggy", "night", "clear"},
        "Unexpected CLIPout caption keys",
    )
    require(
        all(isinstance(text, str) and text.strip() for text in captions.values()),
        "Blank CLIPout caption",
    )


def check_sample_selections() -> None:
    annotations = load_json(ROOT / "data/emu_edit_weather_final.json")["samples"]
    lookup = {sample["hash"]: sample for sample in annotations}
    require(
        len(annotations) == 136,
        f"Expected 136 environmental-condition annotations, found {len(annotations)}",
    )
    flux_ids = load_json(ROOT / "data/flux_sample_ids.json")["hashes"]
    ip2p_ids = load_json(ROOT / "data/ip2p_sample_ids.json")["hashes"]
    require(
        len(flux_ids) == len(set(flux_ids)) == 110,
        "FLUX sample IDs must contain 110 unique hashes",
    )
    require(
        len(ip2p_ids) == len(set(ip2p_ids)) == 128,
        "IP2P sample IDs must contain 128 unique hashes",
    )
    require(set(flux_ids).issubset(lookup), "FLUX selection references unknown annotations")
    require(set(ip2p_ids).issubset(lookup), "IP2P selection references unknown annotations")
    require(len(set(flux_ids) & set(ip2p_ids)) == 102, "Expected 102 shared samples")

    flux_counts = Counter(lookup[sample_hash]["kg_condition"] for sample_hash in flux_ids)
    expected_flux_counts = Counter(
        {"night": 41, "rainy": 29, "snowy": 20, "clear": 15, "foggy": 5}
    )
    require(
        flux_counts == expected_flux_counts,
        f"Unexpected FLUX condition counts: {flux_counts}",
    )
    ip2p_counts = Counter(
        lookup[sample_hash]["kg_condition"] for sample_hash in ip2p_ids
    )
    expected_ip2p_counts = Counter(
        {"night": 45, "rainy": 32, "snowy": 25, "clear": 21, "foggy": 5}
    )
    require(
        ip2p_counts == expected_ip2p_counts,
        f"Unexpected IP2P condition counts: {ip2p_counts}",
    )


def check_result_archives() -> tuple[list[dict], list[dict]]:
    flux = load_json(ROOT / "results/raw/flux_results.json")
    ip2p = load_json(ROOT / "results/raw/ip2p_results.json")
    require(isinstance(flux, list) and len(flux) == 330, "FLUX archive must contain 330 records")
    require(isinstance(ip2p, list) and len(ip2p) == 128, "IP2P archive must contain 128 records")
    require(
        Counter(row["seed"] for row in flux) == Counter({42: 110, 123: 110, 777: 110}),
        "FLUX seed counts are incorrect",
    )
    require(len({(row["hash"], row["seed"]) for row in flux}) == 330, "Duplicate FLUX hash/seed records")
    require(len({row["hash"] for row in ip2p}) == 128, "Duplicate IP2P hashes")

    for method, prefix in FLUX_METHOD_FIELDS.items():
        values = flux_sample_means(flux, f"{prefix}_clip_dir_common")
        require(len(values) == 107, f"FLUX {method} CLIPdir should have 107 valid samples")
    for method, prefix in IP2P_METHOD_FIELDS.items():
        values = ip2p_values(ip2p, prefix, "clip_dir_common")
        require(len(values) == 99, f"IP2P {method} CLIPdir should have 99 valid samples")
    return flux, ip2p


def check_reported_means(flux: list[dict], ip2p: list[dict]) -> None:
    expected_flux = {
        "Simple": (0.158, 0.180, 0.854, 0.255, 0.855),
        "LLM-only": (0.158, 0.177, 0.867, 0.240, 0.881),
        "SK+Filter": (0.181, 0.192, 0.831, 0.292, 0.817),
        "SK+LLM": (0.183, 0.199, 0.804, 0.301, 0.775),
    }
    metrics = ("clip_dir_common", "clip_out", "clip_im", "l1", "dino")
    for method, prefix in FLUX_METHOD_FIELDS.items():
        actual = [flux_seed_summary(flux, prefix, metric)[0] for metric in metrics]
        require(
            all(close(value, target, 6e-4) for value, target in zip(actual, expected_flux[method])),
            f"FLUX means differ for {method}: {actual}",
        )

    expected_ip2p = {
        "Simple": (0.088, 0.208, 0.896, 0.172, 0.870),
        "MGIE": (0.095, 0.211, 0.881, 0.162, 0.826),
        "LLM-only": (0.099, 0.212, 0.865, 0.185, 0.809),
        "SK+Filter": (0.091, 0.212, 0.864, 0.184, 0.804),
        "SK+LLM": (0.095, 0.213, 0.870, 0.184, 0.820),
    }
    for method, prefix in IP2P_METHOD_FIELDS.items():
        actual = [fmean(ip2p_values(ip2p, prefix, metric).values()) for metric in metrics]
        require(
            all(close(value, target, 6e-4) for value, target in zip(actual, expected_ip2p[method])),
            f"IP2P means differ for {method}: {actual}",
        )

    archived_sk_filter_l1 = fmean(ip2p_values(ip2p, "kg_llm", "l1").values())
    require(
        close(archived_sk_filter_l1, 0.183484375, 1e-9),
        "Unexpected archived SK+Filter L1 mean",
    )
    precision_note = (ROOT / "results/README.md").read_text(encoding="utf-8")
    require(
        "0.183484" in precision_note and "0.184" in precision_note,
        "Missing archived-metric precision disclosure",
    )


def check_statistics(flux: list[dict], ip2p: list[dict]) -> None:
    tests = calculate(flux, ip2p)
    lookup = {(test.section, test.metric, test.comparison): test for test in tests}
    required = {
        ("FLUX", "clip_dir_common", "SK+LLM vs LLM-only"): (107, 0.0250, 0.002158),
        ("FLUX", "clip_out", "SK+LLM vs LLM-only"): (110, 0.0219, 0.000001),
        ("IP2P", "clip_dir_common", "SK+LLM vs LLM-only"): (99, -0.0033, 0.552814),
        ("Matched subset", "clip_dir_common", "FLUX: SK+LLM vs LLM-only"): (99, 0.0274, 0.000961),
        ("Matched subset", "clip_dir_common", "IP2P: SK+LLM vs LLM-only"): (99, -0.0033, 0.552814),
    }
    for key, (n, difference, p_value) in required.items():
        test = lookup[key]
        require(test.n == n, f"Unexpected paired N for {key}: {test.n}")
        require(
            close(test.mean_difference, difference, 6e-4),
            f"Unexpected difference for {key}: {test.mean_difference}",
        )
        if p_value == 0.000001:
            require(test.wilcoxon_p < p_value, f"Expected p < {p_value} for {key}")
        else:
            require(
                close(test.wilcoxon_p, p_value, 2e-6),
                f"Unexpected p-value for {key}: {test.wilcoxon_p}",
            )

    ip2p_pairwise = [test for test in tests if test.section == "IP2P"]
    require(
        len(ip2p_pairwise) == 20,
        f"Expected 20 IP2P pairwise tests, found {len(ip2p_pairwise)}",
    )
    require(
        all(test.wilcoxon_p >= 0.05 for test in ip2p_pairwise),
        "An IP2P pairwise test is unexpectedly significant",
    )


def check_prompt_protocol() -> None:
    require(FLUX_PROMPTS.scene_prompt != IP2P_PROMPTS.scene_prompt, "Editor-specific scene prompts should differ")
    require(FLUX_PROMPTS.system_prompt != IP2P_PROMPTS.system_prompt, "Editor-specific system prompts should differ")
    require(FLUX_PROMPTS.reasoning_mode == "postprocess", "FLUX reasoning handling is incorrectly documented")
    require(IP2P_PROMPTS.reasoning_mode == "closed_think_prefill", "IP2P reasoning handling is incorrectly documented")

    combined = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in ("README.md", "docs/experiment_protocol.md", "docs/prompts.md", "docs/reproducibility.md")
    ).lower()
    for phrase in (
        "independently regenerated for ip2p",
        "descriptive rather than a causal",
        "controls sample composition only",
    ):
        require(phrase in combined, f"Missing cross-backbone disclosure: {phrase}")


def check_markdown_links_and_placeholders() -> None:
    markdown_files = list(ROOT.rglob("*.md"))
    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    html_image_pattern = re.compile(r'<img[^>]+src="([^"]+)"', flags=re.IGNORECASE)
    broken: list[str] = []
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        links = link_pattern.findall(text) + html_image_pattern.findall(text)
        for link in links:
            if link.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = link.split("#", 1)[0]
            if not target:
                continue
            if not (path.parent / target).resolve().exists():
                broken.append(f"{path.relative_to(ROOT)} -> {link}")
    require(not broken, f"Broken local Markdown links: {broken}")

    forbidden_patterns = {
        "<repo-url>": "repository URL placeholder",
        "@article{todo": "citation placeholder",
        "<authors>": "author placeholder",
        "author action required": "internal author note",
        "/workspace/": "workspace-specific absolute path",
        "instructions reused unchanged": "incorrect cross-backbone claim",
    }
    searchable_extensions = {".py", ".md", ".toml", ".txt", ".yml", ".yaml", ".cff"}
    hits: list[str] = []
    for path in ROOT.rglob("*"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.is_file() and path.suffix.lower() in searchable_extensions:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for pattern, label in forbidden_patterns.items():
                if pattern in text:
                    hits.append(f"{label}: {path.relative_to(ROOT)}")
    require(not hits, f"Placeholders or stale claims remain: {hits}")

    secret_patterns = [
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"hf_[A-Za-z0-9]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
    ]
    secret_hits = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.stat().st_size < 2_000_000:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(pattern.search(text) for pattern in secret_patterns):
                secret_hits.append(str(path.relative_to(ROOT)))
    require(not secret_hits, f"Potential secrets found: {secret_hits}")


def check_citation_and_exports() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    require(citation["cff-version"] == "1.2.0", "Unexpected CFF version")
    require(citation["title"].startswith("Training-Free Structured"), "CITATION title mismatch")
    require(citation["authors"][0]["family-names"] == "Park", "CITATION author missing")

    expected_rows = {
        "overall_metrics.csv": 9,
        "condition_clipdir.csv": 45,
        "statistical_tests.csv": 28,
        "matched_subset.csv": 1,
    }
    for filename, count in expected_rows.items():
        with (ROOT / "results" / filename).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == count, f"{filename} should have {count} rows, found {len(rows)}")


def check_assets_and_syntax() -> None:
    from PIL import Image

    expected_hashes = {
        "framework.png": "9e67985302fe6b290553f0a1d8203a66fde41ae006eafd0e1a28e766a58a0b75",
        "clipdir_results.png": "257a8c4e6ce485bef50db131067c1a0797d55e827899423d47a0ce5eca46933c",
    }
    for path in (ROOT / "assets/framework.png", ROOT / "assets/clipdir_results.png"):
        with Image.open(path) as image:
            require(image.width >= 1200 and image.height >= 400, f"Asset resolution too low: {path.name} {image.size}")
            image.verify()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == expected_hashes[path.name], f"Canonical paper asset changed: {path.name}")

    success = compileall.compile_dir(ROOT / "src", quiet=1)
    success &= compileall.compile_dir(ROOT / "scripts", quiet=1)
    success &= compileall.compile_dir(ROOT / "analysis", quiet=1)
    success &= compileall.compile_dir(ROOT / "figures", quiet=1)
    require(bool(success), "Python syntax compilation failed")

    bytecode = list(ROOT.rglob("*.pyc")) + [path for path in ROOT.rglob("__pycache__") if path.is_dir()]
    for path in bytecode:
        if path.is_file():
            path.unlink()
    for path in sorted((item for item in ROOT.rglob("__pycache__") if item.is_dir()), reverse=True):
        path.rmdir()


def main() -> None:
    checks = [
        ("required files and cleanup", check_required_files),
        ("structured repository and evaluation captions", check_repository_data),
        ("sample selections", check_sample_selections),
    ]
    flux: list[dict] = []
    ip2p: list[dict] = []

    for label, function in checks:
        function()
        print(f"[PASS] {label}")

    flux, ip2p = check_result_archives()
    print("[PASS] archived result schemas and valid-sample counts")
    check_reported_means(flux, ip2p)
    print("[PASS] reported overall metrics")
    check_statistics(flux, ip2p)
    print("[PASS] paired statistics and matched subset")
    check_prompt_protocol()
    print("[PASS] editor-specific prompt protocol and disclosure")
    check_markdown_links_and_placeholders()
    print("[PASS] README links, placeholders, stale claims, and secrets")
    check_citation_and_exports()
    print("[PASS] citation metadata and exported result tables")
    check_assets_and_syntax()
    print("[PASS] figure assets and Python syntax")
    print("\nAll 10 repository check groups passed.")


if __name__ == "__main__":
    try:
        main()
    except CheckFailure as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        raise SystemExit(1) from error
