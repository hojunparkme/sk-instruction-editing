# Results

## `raw/`

- `flux_results.json`: 330 records = 110 samples × 3 seeds.
- `ip2p_results.json`: 128 single-seed records.

These files retain the original workspace field names. In particular:

| Archived field | Paper label |
|---|---|
| FLUX `kg` | SK+Filter |
| FLUX `kg_nofilter` | SK+LLM |
| IP2P `kg_llm` | SK+Filter |
| IP2P `kg_llm_nofilter` | SK+LLM |

## Readable exports

Run:

```bash
python scripts/export_results.py
```

This regenerates:

- `overall_metrics.csv`
- `condition_clipdir.csv`
- `statistical_tests.csv`
- `matched_subset.csv`

## Precision note

The archived per-sample metrics are stored to four decimal places. Their mean for IP2P SK+Filter L1 is `0.183484`, which rounds to `0.183` with standard three-decimal formatting; the manuscript table reports `0.184`. The original unrounded per-sample values are not available. The release therefore keeps the archived records unchanged and reports the discrepancy explicitly. All other manuscript table means match the archived records at three decimals.
