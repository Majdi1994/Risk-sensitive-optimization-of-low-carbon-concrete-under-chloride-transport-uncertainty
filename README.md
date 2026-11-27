# Reproducible Analyses

Companion code and data for *Risk-sensitive optimization of low-carbon concrete under chloride transport uncertainty*. Released under the MIT License (`LICENSE`).

The reported numbers are produced by `code/run_paper.py`. Every script reads and writes inside this folder. Public repository: https://github.com/Majdi1994/Risk-sensitive-optimization-of-low-carbon-concrete-under-chloride-transport-uncertainty

## Data

| File | Role |
|---|---|
| `ASTM_C1556_n540.csv` | Deposited modelling corpus (n = 540) with `source_id` / `source_label`, `Da_units`, and `cs_provenance` (source-reported vs UCI-filled compressive strength) |
| `ASTM C1556 - Data Collected.xlsx` | Human-readable workbook of the same deposited corpus |
| `chloride.csv` | Cleaned ASTM C1556 bulk-diffusion table used by the search |
| `astm_with_source.csv` | Same experiments with a source label, used by leave-one-source-out |
| `mk_aging.xlsx` | Aging-exponent workbook (MK mean \(m=0.283\) from 48 MK-only ASTM C1556 rows); includes `Source_ID` / `DOI` / `Reference` per row |
| `Compressive_Strength.xlsx` | UCI concrete compressive-strength set used to train the auxiliary fill model |
| `ngb_checkpoint.pkl` | Fitted NGBRegressor loaded by the validation |
| `ngb_model.joblib` | Saved production model from the archived run. The scripts do not read this file. |

The production scripts read the CSV deposit (`ASTM_C1556_n540.csv` / `chloride.csv` / `astm_with_source.csv`). The `.xlsx` workbook is a human-readable companion copy of the curated corpus; it is not required to run `code/run_paper.py`.

Missing compressive strengths for fly-ash and/or slag mixtures were filled with an auxiliary UCI Age-28 model before analysis; silica-fume and metakaolin rows with missing strength were not filled that way. See `data/README_DATA_DEPOSIT.txt`.

The 24-source companion diagnostic that removes the dominant source **before** retraining is `code/loso_exclude_dominant.py` → `outputs/rev2_loso24_*` (pooled R² ≈ 0.220).


## Notebooks

`notebooks/Final_project_notebook_majdi.ipynb` is the course notebook used to develop the UCI Age-28 compressive-strength fill workflow. It reads `data/Compressive_Strength.xlsx` (n = 1030; identical records to the original `Concrete_Data.xlsx`). The deposited entry point remains `code/run_paper.py` / `code/uci_age28_strength.py`.
## Run

```bash
pip install -r requirements.txt
python code/run_paper.py
```

That single command is the only entry point. It runs the validation, the search, the audits, the cross-law rescore, Figure 6, the workflow and carbon figures, the graphical abstract, and the remaining figures. Tables are written to `outputs/`. Figures are written to `figures/`. The deposited `figures/` folder contains only a placeholder. The figures are written there when `code/run_paper.py` or `code/regenerate_figures.py` is run; they are not pre-rendered in this ZIP. The validation loads `data/ngb_checkpoint.pkl` and writes the scoring model to `outputs/rev2_production_ngb.joblib`.

`chloride_pipeline.py`, `initiation_physics.py`, and `plot_style.py` are imported by those steps. They are not launched on their own.

The command prints the archived pooled leave-one-source-out diffusivity R² on n = 540 (~0.316). It also rescores the deposited fronts under production mu/sigma and constant sigma. Summary: `outputs/_cross_law_rescore/cross_law_rescore_summary.json`. Residual-dependence arms in `dependence_sensitivity.py` draw one residual array over each seed's 160-candidate union.
