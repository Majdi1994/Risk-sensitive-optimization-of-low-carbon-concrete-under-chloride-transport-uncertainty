#!/usr/bin/env python3
"""Export the live inventory constants and the production retained-feature set.

Both are used by the search but were not printed in the original submission.
Values are read from the executing code and the serialized production artifact so
the SI tables cannot drift from what the search actually evaluated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chloride_pipeline import AP, COST, EP, GWP, RHO  # noqa: E402
from grouped_loso_validation import EXPORTS  # noqa: E402

OUT = EXPORTS / "rev2_inventory_and_features.json"

# impact_from_mix() maps decision-variable names onto inventory keys.
MIX_TO_INVENTORY = {
    "OPC": "OPC", "FA": "FA", "GGBFS": "GGBFS", "SF": "SF", "MK": "MK",
    "Water": "Water", "NFA": "Fine_Aggregate", "NCA": "Coarse_Aggregate",
    "SP": "Admixture",
}


def retained_features() -> dict:
    art = joblib.load(EXPORTS / "rev2_production_ngb.joblib")
    pre = art["preprocessing"]
    names_in = list(art["all_feature_names"])
    sel = pre.named_steps["select"]
    mask = sel.get_support()
    if len(mask) != len(names_in):
        raise RuntimeError(
            f"selector support ({len(mask)}) does not match all_feature_names ({len(names_in)})"
        )
    scores = getattr(sel, "scores_", None)
    kept = [
        {"feature": n, "mutual_information": float(scores[i]) if scores is not None else None}
        for i, n in enumerate(names_in)
        if mask[i]
    ]
    kept.sort(key=lambda r: (r["mutual_information"] is None, -(r["mutual_information"] or 0.0)))
    for rank, row in enumerate(kept, 1):
        row["rank"] = rank
    dropped = [n for i, n in enumerate(names_in) if not mask[i]]
    return {
        "pipeline_steps": [n for n, _ in pre.steps],
        "selector": type(sel).__name__,
        "score_function": getattr(getattr(sel, "score_func", None), "__name__", None),
        "k": int(getattr(sel, "k", len(kept))),
        "n_candidate_features": len(names_in),
        "n_selected": len(kept),
        "selection_scope": (
            "Refit inside every LOSO fold on training sources only; the list below is "
            "the production fit on all 540 records, which is the model used in the search. "
            "Fold-local retained sets can differ."
        ),
        "selected_features": kept,
        "dropped_features": dropped,
    }


def main() -> int:
    payload = {
        "note": (
            "Inventory factors are cradle-to-gate, per kg of constituent, except "
            "'Production', which is a per-cubic-metre batching constant added once "
            "per mixture by impact_from_mix()."
        ),
        "source_of_truth": "chloride_pipeline.py (imported by matched_search.py)",
        "objectives_using_these": {
            "f1_cost": "impact_from_mix(mix, COST)",
            "f4_gwp": "impact_from_mix(mix, GWP)",
            "post_hoc_only": ["AP", "EP"],
        },
        "mix_key_to_inventory_key": MIX_TO_INVENTORY,
        "gwp_kg_co2eq_per_kg": GWP,
        "ap_kg_so2eq_per_kg": AP,
        "ep_kg_po4eq_per_kg": EP,
        "cost_usd_per_kg": COST,
        "density_kg_per_m3": RHO,
        "features": retained_features(),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "features"}, indent=2))
    f = payload["features"]
    print(f"\nselector={f['selector']}({f['score_function']}, k={f['k']}) "
          f"{f['n_candidate_features']} -> {f['n_selected']}")
    for row in f["selected_features"]:
        print(f"  {row['rank']:2d}. {row['feature']:<28} MI={row['mutual_information']:.4f}")
    print("\ndropped:", ", ".join(f["dropped_features"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
