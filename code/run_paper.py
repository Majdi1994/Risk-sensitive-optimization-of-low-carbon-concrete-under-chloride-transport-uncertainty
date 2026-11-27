#!/usr/bin/env python3
"""Run every analysis in this package from this one file, in order.

Library modules (chloride_pipeline.py, initiation_physics.py, plot_style.py)
are imported by the steps below. They are not separate commands.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent


def run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(CODE / script), *(extra or [])]
    print(f"\n=== {script} ===", flush=True)
    subprocess.run(cmd, cwd=CODE, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="rescore outputs/rev2_matched_nsga2_pools.csv instead of running NSGA-II",
    )
    parser.add_argument(
        "--from",
        dest="start_at",
        default=None,
        help="resume at this script name, using files already in outputs/",
    )
    args = parser.parse_args()

    missing = [
        name for name in (
            "chloride.csv",
            "astm_with_source.csv",
            "ASTM_C1556_n540.csv",
            "mk_aging.xlsx",
            "Compressive_Strength.xlsx",
            "ngb_checkpoint.pkl",
        )
        if not (ROOT / "data" / name).exists()
    ]
    if missing:
        raise SystemExit("Missing data files: " + ", ".join(missing))

    archived = [
    ]
    absent = [str(path.relative_to(ROOT)) for path in archived if not path.exists()]
    if absent:
        raise SystemExit("Missing archived outputs: " + ", ".join(absent))

    steps: list[tuple[str, list[str]]] = [
        ("analysis_specification.py", []),
        ("uci_age28_strength.py", []),
        ("grouped_loso_validation.py", []),
        ("loso_exclude_dominant.py", []),
        ("within_domain_split.py", []),
        ("strength_model.py", []),
        ("export_inventory_and_features.py", []),
        (
            "matched_search.py",
            ["--reuse-pools", "--skip-saa"] if args.skip_search else [],
        ),
        ("posthoc_rerank_control.py", []),
        ("saa_multistream_audit.py", []),
        ("ood_threshold_sensitivity.py", []),
        ("ood_support_audit.py", []),
        ("strength_retention.py", []),
        ("weighted_loso_metrics.py", []),
        ("transfer_blockers.py", []),
        ("dependence_sensitivity.py", []),
        ("opc_reference_benchmark.py", []),
        ("loso_dependence_summary.py", []),
        ("cover_cap_example.py", []),
        ("cross_law_rescore.py", []),
        ("workflow_carbon_figures.py", []),
        ("make_figure6.py", []),
        ("graphical_abstract.py", []),
        ("regenerate_figures.py", []),
    ]
    names = [name for name, _ in steps]
    if args.start_at is not None and args.start_at not in names:
        raise SystemExit("Unknown script: " + args.start_at)
    started = args.start_at is None
    for name, extra in steps:
        if not started:
            started = name == args.start_at
            if not started:
                continue
        run(name, extra)
    print(f"\nFinished. Tables are in {ROOT / 'outputs'} and figures in {ROOT / 'figures'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
