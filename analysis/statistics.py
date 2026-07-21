"""
Paired statistics reported in the paper.

FLUX: the three seed-level scores of each sample are averaged first, then
sample-level paired tests are run (bootstrap CI + Wilcoxon signed-rank).
IP2P: single seed, so sample-level values are used directly.

Usage:
    python statistics.py flux_results.json ip2p_results.json
"""
import json, sys
import numpy as np
from scipy import stats

N_BOOT = 10_000
SEED   = 42


def bootstrap_ci(diffs, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    boot = [np.mean(rng.choice(diffs, len(diffs), replace=True)) for _ in range(n_boot)]
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def _valid(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def flux_sample_means(rows, field):
    """Average the seed-level scores per sample."""
    acc = {}
    for r in rows:
        v = r.get(field)
        if _valid(v):
            acc.setdefault(r["hash"], []).append(v)
    return {h: float(np.mean(vs)) for h, vs in acc.items()}


def paired_report(label, a, b):
    """a, b: dict hash -> value. Reports mean(a-b), 95% CI, Wilcoxon p."""
    keys = [h for h in a if h in b]
    d = np.array([a[h] - b[h] for h in keys])
    lo, hi = bootstrap_ci(d)
    p = stats.wilcoxon(d).pvalue
    print(f"{label:<42} N={len(d):>4}  diff={np.mean(d):+.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.6f}")
    return np.mean(d), lo, hi, p


def main():
    flux_path = sys.argv[1] if len(sys.argv) > 1 else "flux_results.json"
    ip2p_path = sys.argv[2] if len(sys.argv) > 2 else "ip2p_results.json"

    flux = json.load(open(flux_path))
    print("=" * 96)
    print("FLUX Kontext  (seed-averaged per sample, then sample-level paired tests)")
    print("=" * 96)
    for metric in ["clip_dir_common", "clip_out"]:
        print(f"\n--- {metric} ---")
        m = {k: flux_sample_means(flux, f"{k}_{metric}")
             for k in ["simple", "llm", "kg", "kg_nofilter"]}
        paired_report("SK+LLM vs LLM-only",  m["kg_nofilter"], m["llm"])
        paired_report("SK+LLM vs Simple",    m["kg_nofilter"], m["simple"])
        paired_report("SK+LLM vs SK+Filter", m["kg_nofilter"], m["kg"])

    ip2p = json.load(open(ip2p_path))
    print("\n" + "=" * 96)
    print("InstructPix2Pix  (single seed)")
    print("=" * 96)

    def ip2p_vals(method, metric):
        out = {}
        for r in ip2p:
            if metric == "clip_dir_common":
                v = r.get(f"{method}_clip_dir_common")
            else:
                v = r.get(method, {}).get(metric)
            if _valid(v):
                out[r["hash"]] = v
        return out

    for metric in ["clip_dir_common", "clip_out"]:
        print(f"\n--- {metric} ---")
        paired_report("SK+LLM vs LLM-only",
                      ip2p_vals("kg_llm_nofilter", metric), ip2p_vals("llm_only", metric))
        paired_report("SK+LLM vs MGIE",
                      ip2p_vals("kg_llm_nofilter", metric), ip2p_vals("mgie_style", metric))

    # all pairwise comparisons under IP2P (supports the claim in Section 4.3)
    import itertools
    LABEL = {"simple": "Simple", "mgie_style": "MGIE", "llm_only": "LLM-only",
             "kg_llm": "SK+Filter", "kg_llm_nofilter": "SK+LLM"}
    for metric in ["clip_dir_common", "clip_out"]:
        print(f"\n--- IP2P all pairwise, {metric} ---")
        n_sig = 0
        for a, b in itertools.combinations(LABEL, 2):
            _, _, _, p = paired_report(f"{LABEL[a]} vs {LABEL[b]}",
                                       ip2p_vals(a, metric), ip2p_vals(b, metric))
            n_sig += int(p < 0.05)
        print(f"  significant pairs at alpha=0.05: {n_sig}/10")

    # matched subset
    print("\n" + "=" * 96)
    print("Matched subset shared by both backbones")
    print("=" * 96)
    shared = {r["hash"] for r in flux} & {r["hash"] for r in ip2p}
    print(f"shared samples: {len(shared)}")
    f = {k: flux_sample_means([r for r in flux if r["hash"] in shared],
                              f"{k}_clip_dir_common") for k in ["llm", "kg_nofilter"]}
    paired_report("FLUX  SK+LLM vs LLM-only (CLIPdir)", f["kg_nofilter"], f["llm"])
    i_n = {h: v for h, v in ip2p_vals("kg_llm_nofilter", "clip_dir_common").items() if h in shared}
    i_l = {h: v for h, v in ip2p_vals("llm_only", "clip_dir_common").items() if h in shared}
    paired_report("IP2P  SK+LLM vs LLM-only (CLIPdir)", i_n, i_l)


if __name__ == "__main__":
    main()
