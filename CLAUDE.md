# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project analyzing the impact of maritime shipping regulation (Emission Control Areas) on kelp forests along the California coast. Uses satellite kelp biomass data, AIS vessel tracking, and environmental covariates in a continuous Difference-in-Differences (DiD) econometric framework.

Paper title: "The Hidden Cost of Globalization in the Underwater World: Evidence from California's Kelp Forests"
Authors: Fabien Candau (UPPA E2S, TREE, France) & Florian Lafferrere (UPPA E2S, TREE)

## Languages & Tools

- **Python** (primary): `pyfixest` for PPML regressions and event studies; data pipeline
- **Stata**: `ppmlhdfe` for 2011 tables; `did_multiplegt_dyn` for heterogeneity-robust event study
- **LaTeX**: Paper manuscript in `paper/Kelp.tex`, bibliography in `paper/ECA.bib`

## Python Environment

Python executable: `C:\Users\fcanda01\AppData\Local\anaconda3\envs\r-env\python.exe`
Always use this Python path when running scripts.

## Setup

```bash
pip install -r data/raw/requirements.txt
```

Requires Python 3.10+ with GDAL/PROJ for geospatial operations. Key Python packages: `pyfixest`, `geopandas`, `rasterio`, `rasterstats`, `scipy`.

## Project Structure

```
kelp/
├── data/
│   ├── raw/          Raw source data (satellite kelp, AIS shipping, rasters, covariates)
│   └── generated/    Script-generated intermediate datasets
├── code/             Python scripts, Stata do files
├── paper/            LaTeX manuscript and bibliography
└── output/           Regression tables, figures, coefficient CSVs
    ├── tables/       LaTeX regression tables
    ├── figures/      Final event study & comparison figures
    └── csv/          Coefficient CSVs from event studies
```

## Running the Pipeline

A Stata master script runs the full pipeline: `code/master.do`. It calls Python via `shell` and Stata do-files via `do`. The individual steps are:

1. **`code/1_build_data.py`** — Build regression dataset from raw data. Produces `data/generated/new_model_kelp_exposure_3.csv`.
2. **`code/2_a_estim.py`** — Main PPML tables, event study figures, placebo test. Produces `output/tables/ship_exposure_kelp_2009_main.tex`, `output/tables/cdid_weighted.tex`, `output/tables/placebo_test.tex`, `output/figures/event_study_main.png`.
3. **`code/2_b_robustness.py`** — Robustness event studies (no SWH, cell trends, restricted sample 2005-2012, model_2008 with/without vessel control). Produces robustness figures.
4. **`code/2_c_anticipation.py`** — Anticipation robustness: temporal stability figure, alternative treatment timing table, treatment correlation 2009 vs 2011. Produces `output/tables/anticipation_robustness.tex`, `output/figures/temporal_stability_shipping.png`, `output/figures/treatment_correlation_2009_2011.png`.
5. **`code/3_estim2011.do`** — 2011 boundary modification PPML tables (unweighted + weighted). Produces `output/tables/ship_exposure_kelp_2011_b.tex` (unweighted, main) and `output/tables/ship_exposure_kelp_2011.tex` (weighted, appendix). No vessel count threshold (2011 boundary redistribution requires low-exposure cells for identification). SEs clustered at cell level.
6. **`code/2_d_dechaisemartin.do`** — de Chaisemartin & D'Haultfoeuille (2024) event study for 2011. Uses `did_multiplegt_dyn` with `continuous(1)`, annual 2008-2016, 999 bootstrap reps. Produces `output/figures/event_study_2011_dCdH.png`.
7. **`code/4_balance_table.py`** — Balance table by treatment-intensity quartile. Produces `output/tables/balance_table.tex`.
8. **`code/2_e_weighting_robustness.py`** — Weighting robustness: percentile sweep figure, restricted-sample unweighted PPML, pre-period kelp biomass weights. Produces `output/figures/weighting_diagnostic.png`, `output/tables/weighting_robustness.tex`.
**Other scripts** (not in numbered pipeline):
- `code/archive/6_first_stage.py` — Traffic redistribution analysis (Lloyd's + AIS). Not in pipeline; kept for referee requests.
- `code/do_file_continuous_did.do` — **Deprecated.** Original Stata PPML; superseded by `2_a_estim.py` + `3_estim2011.do`.
- `code/install_ppmlhdfe.do` — One-time Stata package installer.
- `code/replication/` — Cross-language replication scripts (Stata, R) for Table 1.
- Co-author's notebooks (`code/code_continuous.ipynb`, `code/stat_continuous.ipynb`) use Dropbox paths and are not updated.

There are no tests or linting configurations.

## Key Data Files

- `data/generated/new_model_kelp_exposure_3.csv` — Final regression dataset (14,820 rows x 262 cols)
- `data/raw/full_model.tif` — Halpern cumulative human impact index
- `data/raw/ECA_10km.gpkg` — 10km hexagonal grid (1,719 hexagons)

## Known Data Issues

- **`ln_swh`** (wave height): ERA5 data extended to 2016 via `download_swh_2015_2016.py`. 77 cells with NaN (vs 58 original) due to 0.5° resolution at coastline. Sample now 2005-2016.
- **Safe controls**: `ln_sst` and `ln_precipitation` have zero NaN across all years
- **PPML sample**: 3,880 obs (81 cells x 48 quarters, 2005-2016) after count >= 2 threshold
- **AIS data**: Starts January 2009 — no pre-2009 vessel tracking available
- **Marine heat wave**: The 2013-2016 "Blob" is absorbed by year-quarter FE (coast-wide shock)

## Main Specification: Unweighted PPML on Restricted Sample

**Sample restriction**: cells with >= 2 vessels/month (`ln_mean_count >= ln(2)`), yielding 81 cells x 48 quarters = 3,880 obs (2005-2016).

Main results use **unweighted PPML** (`fepois` without offset trick). Weighted PPML results (using offset trick) are in the appendix as robustness.

**Offset trick** (for appendix weighted PPML): `pyfixest.fepois()` does NOT support user-specified weights. To replicate Stata's `ppmlhdfe y [pw=w]`:
```python
df["log_w"] = np.log(df["w"])
df["biomass_w"] = df["w"] * df["biomass_sum"]
est = pf.fepois("biomass_w ~ treatment:post + controls + log_w | FE_i + FE_t", data=df, vcov={"CRV1": "cell_id"})
```

On the count >= 2 sample, weighted and unweighted give nearly identical coefficients (no sign reversal).

## Main Results (Table 1) — Unweighted PPML, 81 cells, 2005-2016

| Treatment | Biomass coef | Area coef | p-value |
|-----------|-------------|-----------|---------|
| Vessel Distance | +0.136*** | +0.132*** | <0.001 |
| Vessel Count | +0.155*** | +0.149*** | <0.01 |

## Paper Structure

- **Main text**: Intro → Background → The 2009 Ocean-Going Vessel Fuel Rule (Data, Specification, Results, Dynamic effects, Threats to identification) → Boundary Modification of the Regulated Zone in 2011 → Conclusion
- **Main tables**: `ship_exposure_kelp_2009_main.tex` (Table 1), `ship_exposure_kelp_2011_b.tex` (Table 2)
- **Main figures**: `event_study_main.png`, `event_study_2011_dCdH.png`
- **Appendix order** (by first main-body reference): Weighted PPML 2009 → Placebo → Balance Table → Financial Crisis → Anticipation → Synthetic DiD → After 2011 → 2010 Treatment → Treatment Correlation → Weighted PPML 2011
- **Automated tables** (via `\input{}`): all main tables, placebo, anticipation, balance table, unweighted specs
- **Co-author's files** (never delete): `output/tables/resultats_sc.tex`, map figures (`SC.png`, `map_SC.png`, `weigth_sc.png`)

## Important Notes

- Co-author's notebooks have Dropbox paths — NOT updated
- pyfixest API: use `.coef()` and `.se()` (NOT `.params()` which doesn't exist)
- **Windows NTFS is case-insensitive**: never create directories differing only in case
- All LaTeX table/figure paths use `../output/` relative to `paper/` directory
- **Never export figures as PDF** — only `.png`
- Standard errors: clustered at cell level throughout (Python: `{"CRV1": "cell_id"}`, Stata: `vce(cluster FE_i)`)
