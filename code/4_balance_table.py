"""
Balance table: covariate means by treatment-intensity quartile.

Generates:
  - output/tables/balance_table.tex  (LaTeX table for appendix)

Computes cell-level pre-treatment (2005-2008) means of key variables,
split by quartile of the main treatment variable (vessel distance,
January-June 2009). Reports means and a Kruskal-Wallis test for
differences across quartiles.

Sample restricted to cells with >= 2 vessels/month (count threshold).
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kruskal
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED = os.path.join(BASE, "data", "generated")
TABLE_DIR = os.path.join(BASE, "output", "tables")
os.makedirs(TABLE_DIR, exist_ok=True)

# Load data — same sample filters as 2_a_estim.py
df = pd.read_csv(os.path.join(GENERATED, "new_model_kelp_exposure_3.csv"))
df = df[(df["year"] <= 2016) & (df["latitude"] >= 32) & (df["latitude"] <= 36)].copy()
df["yq"] = df["year"].astype(str) + "_" + df["quarter"].astype(str)
df = df.drop_duplicates(subset=["eca_index_official", "year_eca", "yq"])

key_cols = ["biomass_sum", "area_sum",
            "ln_mean_length_2009_jan_jun_20km", "ln_mean_count_2009_jan_jun_20km",
            "ln_sst_mean", "ln_precipitation_mean", "ln_swh_mean",
            "ln_sum_count_2009_jan_jun_20km"]
df = df.dropna(subset=key_cols)

# Alias column names
df["ln_sst"] = df["ln_sst_mean"]
df["ln_precipitation"] = df["ln_precipitation_mean"]
df["ln_swh"] = df["ln_swh_mean"]

# Cell-level identifiers
df["cell_id"] = df["eca_index_official"].astype(str) + "_" + df["year_eca"].astype(str)

# -- Sample restriction: cells with >= 2 vessels/month ---------------------
VESSEL_THRESHOLD = np.log(2)
treat_count_var = "ln_mean_count_2009_jan_jun_20km"
cell_count = df.groupby("cell_id")[treat_count_var].first()
cells_above = cell_count[cell_count >= VESSEL_THRESHOLD].index
df = df[df["cell_id"].isin(cells_above)].copy()

print(f"Full sample: {len(df)} obs, {df['cell_id'].nunique()} cells")
print(f"  (Restricted to cells with >= 2 vessels/month)")

# Treatment variable: cell-level (time-invariant)
treat_var = "ln_mean_length_2009_jan_jun_20km"

# Get cell-level treatment values (one per cell)
cell_treat = df.groupby("cell_id")[treat_var].first()

# Create quartiles
quartile_labels = {1: "Q1 (low)", 2: "Q2", 3: "Q3", 4: "Q4 (high)"}
cell_quartile = pd.qcut(cell_treat, 4, labels=[1, 2, 3, 4]).astype(int)
cell_quartile.name = "quartile"
df = df.merge(cell_quartile.reset_index(), on="cell_id", suffixes=("", "_q"))

# Restrict to pre-treatment period (2005-2008) for balance
df_pre = df[df["year"] <= 2008].copy()
print(f"Pre-treatment sample: {len(df_pre)} obs, {df_pre['cell_id'].nunique()} cells")

# Variables to report
balance_vars = [
    ("biomass_sum", "Kelp biomass (kg)"),
    ("area_sum", "Kelp canopy area (m$^{2}$)"),
    ("ln_sst", "Log SST"),
    ("ln_precipitation", "Log precipitation"),
    ("ln_swh", "Log significant wave height"),
    ("ln_mean_count_2009_jan_jun_20km", "Log vessel count (Jan-Jun 2009)"),
]

# Compute means by quartile and overall, plus Kruskal-Wallis test
print(f"\n{'Variable':<35} {'Q1 (low)':>10} {'Q2':>10} {'Q3':>10} {'Q4 (high)':>10} {'KW p':>8}")
print("-" * 85)

results = []
for var, label in balance_vars:
    row = {"label": label}
    groups = []
    for q in [1, 2, 3, 4]:
        vals = df_pre.loc[df_pre["quartile"] == q, var].dropna()
        row[f"q{q}_mean"] = vals.mean()
        row[f"q{q}_sd"] = vals.std()
        row[f"q{q}_n"] = len(vals)
        groups.append(vals.values)

    try:
        stat, p = kruskal(*groups)
        row["kw_stat"] = stat
        row["kw_p"] = p
    except Exception:
        row["kw_stat"] = np.nan
        row["kw_p"] = np.nan

    results.append(row)
    print(f"{label:<35} {row['q1_mean']:>10.3f} {row['q2_mean']:>10.3f} "
          f"{row['q3_mean']:>10.3f} {row['q4_mean']:>10.3f} {row.get('kw_p', np.nan):>8.3f}")

# Also report the treatment variable itself
treat_row = {"label": "Log vessel distance (Jan-Jun 2009)"}
groups = []
for q in [1, 2, 3, 4]:
    vals = df_pre.loc[df_pre["quartile"] == q, treat_var].dropna()
    treat_row[f"q{q}_mean"] = vals.mean()
    treat_row[f"q{q}_sd"] = vals.std()
    groups.append(vals.values)
stat, p = kruskal(*groups)
treat_row["kw_stat"] = stat
treat_row["kw_p"] = p
results.insert(0, treat_row)
print(f"{'Log vessel distance (Jan-Jun 2009)':<35} {treat_row['q1_mean']:>10.3f} "
      f"{treat_row['q2_mean']:>10.3f} {treat_row['q3_mean']:>10.3f} "
      f"{treat_row['q4_mean']:>10.3f} {treat_row['kw_p']:>8.3f}")

# Number of cells per quartile
n_cells = df_pre.groupby("quartile")["cell_id"].nunique()
print(f"\nCells per quartile: {dict(n_cells)}")

# --- Build LaTeX table ---
lines = []
lines.append(r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
lines.append(r"\begin{tabular}{l*{4}{c}c}")
lines.append(r"\hline\hline")
lines.append(r"                    &\multicolumn{1}{c}{Q1 (low)}&\multicolumn{1}{c}{Q2}"
             r"&\multicolumn{1}{c}{Q3}&\multicolumn{1}{c}{Q4 (high)}&\multicolumn{1}{c}{KW $p$}\\")
lines.append(r"\hline")

for i, row in enumerate(results):
    label = row["label"]
    mean_line = f"{label:<40}"
    sd_line = "                                        "
    for q in [1, 2, 3, 4]:
        m = row[f"q{q}_mean"]
        s = row[f"q{q}_sd"]
        if abs(m) >= 100:
            mean_line += f"&{m:>12.1f}"
            sd_line += f"&{'[' + f'{s:.1f}' + ']':>12}"
        elif abs(m) >= 1:
            mean_line += f"&{m:>12.3f}"
            sd_line += f"&{'[' + f'{s:.3f}' + ']':>12}"
        else:
            mean_line += f"&{m:>12.4f}"
            sd_line += f"&{'[' + f'{s:.4f}' + ']':>12}"

    p = row.get("kw_p", np.nan)
    if not np.isnan(p):
        mean_line += f"&{p:>8.3f}"
    else:
        mean_line += f"&{'':>8}"
    sd_line += f"&{'':>8}"

    lines.append(mean_line + r"\\")
    lines.append(sd_line + r"\\")
    if i == 0:
        lines.append("[0.5em]")
    else:
        lines.append("[0.3em]")

# Footer: N cells
lines.append(r"\hline")
n_line = "Cells                                   "
for q in [1, 2, 3, 4]:
    n_line += f"&{n_cells.get(q, 0):>12d}"
n_line += f"&{'':>8}"
lines.append(n_line + r"\\")
lines.append(r"\hline\hline")
lines.append(r"\end{tabular}")

tex_path = os.path.join(TABLE_DIR, "balance_table.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nBalance table saved to {tex_path}")
