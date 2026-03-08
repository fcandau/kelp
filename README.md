# Kelp Forests and Maritime Shipping Regulation

Impact of Emission Control Areas (ECA) on kelp forest recovery along the California coast.

**Authors:** Fabien Candau (UPPA E2S, TREE) & Florian Lafferrere (UPPA E2S, TREE)

## Project Structure

```
data/raw/          Raw source data (satellite kelp, AIS shipping, rasters, covariates)
data/generated/    Script-generated intermediate datasets
code/              Python scripts, Stata do files
paper/             LaTeX manuscript and bibliography
output/            Regression tables, event study figures, coefficient CSVs
```

## Pipeline

Run scripts in this order:

1. `code/1_build_data.py` — Build regression dataset → `data/generated/new_model_kelp_exposure_3.csv`
2. `code/2_a_estim.py` — Main PPML tables, event study figure, placebo test (Python/pyfixest)
3. `code/2_b_robustness.py` — Robustness event studies (Python/pyfixest)
4. `code/2_c_anticipation.py` — Anticipation robustness analysis (Python/pyfixest)
5. `code/3_estim2011.do` — 2011 boundary modification PPML tables (Stata/ppmlhdfe)
6. `code/2_d_dechaisemartin.do` — de Chaisemartin & D'Haultfoeuille event study for 2011 (Stata)

## Setup

```bash
pip install -r data/raw/requirements.txt
```

Python 3.10+ with GDAL/PROJ required for the data pipeline (Step 1).
Stata 17+ with `ppmlhdfe`, `reghdfe`, `did_multiplegt_dyn` required for Steps 5–6.
