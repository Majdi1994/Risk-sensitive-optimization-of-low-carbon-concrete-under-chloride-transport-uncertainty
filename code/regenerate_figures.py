#!/usr/bin/env python3
"""Regenerate figures from the deposited exports. Does not run NSGA-II.

If a figure disagrees with an export, correct the export it reads, then rerun
this script. Do not draw a new search to refresh a figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_specification import EXPORTS  # noqa: E402
from plot_style import apply_plot_style  # noqa: E402
from matched_search import FIGS, make_figures  # noqa: E402
from workflow_carbon_figures import (  # noqa: E402
    make_workflow,
    make_carbon_figure,
)


def main() -> int:
    apply_plot_style()
    saa = pd.read_csv(EXPORTS / "rev2_saa_convergence.csv")
    pools = pd.read_csv(EXPORTS / "rev2_matched_nsga2_pools.csv")
    audits = pd.read_csv(EXPORTS / "rev2_identical_S3_audits.csv")
    lambdas = pd.read_csv(EXPORTS / "rev2_lambda_sensitivity.csv")
    indicators = pd.read_csv(EXPORTS / "rev2_generation_indicators.csv")
    make_figures(saa, pools, audits, lambdas, indicators)
    make_workflow()
    make_carbon_figure()
    print(f"regenerated figures under {FIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
