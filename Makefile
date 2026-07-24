.PHONY: verify export figures

verify:
	python scripts/self_check.py
	python analysis/statistics.py

export:
	python scripts/export_results.py

figures:
	python figures/make_framework.py
	python figures/make_clipdir_figure.py
