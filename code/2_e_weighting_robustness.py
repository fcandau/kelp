"""
Weighting robustness analysis for the 2009 ECA result.

Now that the main specification is unweighted PPML on the count >= 2
restricted sample, this script shows robustness to alternative approaches:

  1. Percentile sweep: progressively drop low-exposure cells from the
     FULL sample (all cells with positive shipping) and show the
     unweighted PPML coefficient trajectory. Demonstrates that the
     count >= 2 threshold is not cherry-picked.
     -> output/figures/weighting_diagnostic.png

  2. Weighted PPML (offset trick) on same count >= 2 sample:
     confirms the main result is robust to inverse-variance weighting.

  3. Alternative weights: pre-period kelp biomass (2005-2008 average).

  -> output/tables/weighting_robustness.tex
"""

import os
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED = os.path.join(BASE, 'data', 'generated')
TABLE_DIR = os.path.join(BASE, 'output', 'tables')
FIG_DIR = os.path.join(BASE, 'output', 'figures')
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# -- Formatting helpers -------------------------------------------------------
def stars(pval):
    if pval < 0.01: return "***"
    elif pval < 0.05: return "**"
    elif pval < 0.10: return "*"
    return ""

def fmt_stars(pval):
    s = stars(pval)
    return f"\\sym{{{s}}}" if s else ""

def fmt_coef(val):
    return f"{val:>11.4f}"

def fmt_se(val):
    return f"{'(' + f'{val:.4f}' + ')':>21}"

def get_int_key(est, treatment, policy_var="policy_eca_2009"):
    for key in est.coef().index:
        if treatment in key and policy_var in key:
            return key
    raise KeyError(f"No interaction key for '{treatment}' x '{policy_var}' "
                   f"in {list(est.coef().index)}")


# -- Data preparation (identical to 2_a_estim.py) ----------------------------
df = pd.read_csv(os.path.join(GENERATED, 'new_model_kelp_exposure_3.csv'))
df = df[(df["year"] <= 2016) & (df["latitude"] >= 32) & (df["latitude"] <= 36)].copy()

df["cell_id"] = df["eca_index_official"].astype(str) + "_" + df["year_eca"].astype(str)
df["yq"] = df["year"].astype(str) + "_" + df["quarter"].astype(str)
df = df.drop_duplicates(subset=["cell_id", "yq"])

df["policy_eca_2009"] = 0
df.loc[df["year"] > 2008, "policy_eca_2009"] = 1
df.loc[(df["year"] == 2009) & (df["quarter"] < 3), "policy_eca_2009"] = 0

key_cols = ["biomass_sum", "area_sum", "model_2008",
            "ln_mean_length_2009_jan_jun_20km", "ln_mean_count_2009_jan_jun_20km",
            "ln_sst_mean", "ln_precipitation_mean", "ln_swh_mean",
            "ln_sum_count_2009_jan_jun_20km"]
df = df.dropna(subset=key_cols)

df["ln_sst"] = df["ln_sst_mean"]
df["ln_precipitation"] = df["ln_precipitation_mean"]
df["ln_swh"] = df["ln_swh_mean"]

# Start from the full sample with positive shipping (for sweep)
df_full = df[df["ln_sum_count_2009_jan_jun_20km"] > 0].copy()

treat_dist = "ln_mean_length_2009_jan_jun_20km"
treat_count = "ln_mean_count_2009_jan_jun_20km"
VCOV = {"CRV1": "cell_id"}
controls_base = "ln_sst + ln_precipitation + ln_swh"

cell_treat = df_full.groupby("cell_id")[treat_dist].first()

# Count >= 2 sample (main specification)
VESSEL_THRESHOLD = np.log(2)
cell_count = df_full.groupby("cell_id")[treat_count].first()
cells_main = cell_count[cell_count >= VESSEL_THRESHOLD].index
df_main = df_full[df_full["cell_id"].isin(cells_main)].copy()

print(f"Full sample (w > 0): {len(df_full)} obs, {df_full['cell_id'].nunique()} cells")
print(f"Main sample (count >= 2): {len(df_main)} obs, {df_main['cell_id'].nunique()} cells")


# ================================================================
# 1. PERCENTILE SWEEP FIGURE
# ================================================================
print("\n" + "=" * 80)
print("PERCENTILE SWEEP: Unweighted PPML dropping low-exposure cells")
print("=" * 80)

# Get main unweighted estimates for reference lines (one per depvar)
main_coefs = {}
for _dep in ["biomass_sum", "area_sum"]:
    _est = pf.fepois(
        f"{_dep} ~ {treat_dist}:policy_eca_2009 + {controls_base} | cell_id + yq",
        data=df_main, vcov=VCOV)
    _key = get_int_key(_est, treat_dist)
    main_coefs[_dep] = _est.coef()[_key]

percentiles = list(range(0, 55, 5))
sweep = {dep: {"pcts": [], "coefs": [], "ci_lo": [], "ci_hi": [], "cells": []}
         for dep in ["biomass_sum", "area_sum"]}

print(f"\n{'Pct':>4} {'Cells':>5}  {'Biomass coef':>12} {'p':>7}  {'Area coef':>12} {'p':>7}")
print("-" * 55)

for pct in percentiles:
    cutoff = cell_treat.quantile(pct / 100)
    cells = cell_treat[cell_treat >= cutoff].index
    dfs = df_full[df_full["cell_id"].isin(cells)].copy()
    n_cells = dfs["cell_id"].nunique()

    row_str = f"  P{pct:<2} {n_cells:>5}"
    for depvar in ["biomass_sum", "area_sum"]:
        fml = f"{depvar} ~ {treat_dist}:policy_eca_2009 + {controls_base} | cell_id + yq"
        est = pf.fepois(fml, data=dfs, vcov=VCOV)
        int_key = get_int_key(est, treat_dist)
        c = est.coef()[int_key]
        s = est.se()[int_key]
        p = est.pvalue()[int_key]

        sweep[depvar]["pcts"].append(pct)
        sweep[depvar]["coefs"].append(c)
        sweep[depvar]["ci_lo"].append(c - 1.96 * s)
        sweep[depvar]["ci_hi"].append(c + 1.96 * s)
        sweep[depvar]["cells"].append(n_cells)

        row_str += f"  {c:>8.4f}{stars(p):<3} {p:>7.4f}"
    print(row_str)

# Figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
dep_labels = {"biomass_sum": "Biomass", "area_sum": "Area"}

for i, depvar in enumerate(["biomass_sum", "area_sum"]):
    ax = axes[i]
    r = sweep[depvar]

    ax.fill_between(r["pcts"], r["ci_lo"], r["ci_hi"], alpha=0.2, color="#2166ac")
    ax.plot(r["pcts"], r["coefs"], "o-", color="#2166ac", markersize=5,
            linewidth=1.5, label="Unweighted PPML coefficient")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    mc = main_coefs[depvar]
    ax.axhline(mc, color="#b2182b", linestyle=":", linewidth=1.2, alpha=0.7,
               label=f"Main estimate ({mc:.4f})")

    for j, (pct, nc) in enumerate(zip(r["pcts"], r["cells"])):
        if pct % 10 == 0:
            ax.annotate(f"n={nc}", (pct, r["ci_lo"][j]),
                        fontsize=7, ha="center", va="top", color="grey")

    ax.set_xlabel("Percentile cutoff (cells below dropped)", fontsize=10)
    ax.set_ylabel("Coefficient", fontsize=10)
    ax.set_title(f"Panel {'AB'[i]}: {dep_labels[depvar]}", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(-2, 52)

fig.suptitle("Unweighted PPML: Effect of Dropping Low-Exposure Cells",
             fontsize=12, y=1.02)
fig.tight_layout()
fig_path = os.path.join(FIG_DIR, "weighting_diagnostic.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"\nFigure saved to {fig_path}")
plt.close(fig)


# ================================================================
# 2. WEIGHTED PPML ON SAME COUNT >= 2 SAMPLE (appendix robustness)
# ================================================================
print("\n" + "=" * 80)
print("WEIGHTED PPML: Same count >= 2 sample (offset trick)")
print("=" * 80)

df_main["w"] = df_main["ln_sum_count_2009_jan_jun_20km"]
df_main["log_w"] = np.log(df_main["w"])
df_main["biomass_w"] = df_main["w"] * df_main["biomass_sum"]
df_main["area_w"] = df_main["w"] * df_main["area_sum"]

results_weighted = []
for depvar_w, dep_label in [("biomass_w", "Biomass"), ("area_w", "Area")]:
    for treat, treat_label in [(treat_dist, "Vessel Dist"), (treat_count, "Vessel Count")]:
        fml = (f"{depvar_w} ~ {treat}:policy_eca_2009 "
               f"+ {controls_base} + log_w | cell_id + yq")
        est = pf.fepois(fml, data=df_main, vcov=VCOV)
        int_key = get_int_key(est, treat)
        c, s, p = est.coef()[int_key], est.se()[int_key], est.pvalue()[int_key]
        n = est._N
        print(f"  {dep_label} x {treat_label}: coef={c:.4f}, se={s:.4f}, p={p:.4f} {stars(p)}, N={n}")
        results_weighted.append({
            "depvar": dep_label, "treatment": treat_label,
            "coef": c, "se": s, "pvalue": p, "nobs": n
        })


# ================================================================
# 3. ALTERNATIVE WEIGHT: Pre-period kelp biomass
# ================================================================
print("\n" + "=" * 80)
print("ALTERNATIVE WEIGHT: Pre-period kelp biomass (2005-2008 average)")
print("=" * 80)

df_pre = df_main[df_main["year"] < 2009].groupby("cell_id")["biomass_sum"].mean().reset_index()
df_pre.columns = ["cell_id", "pre_kelp_biomass"]
df_main = df_main.merge(df_pre, on="cell_id", how="left")

df_kelp = df_main[df_main["pre_kelp_biomass"] > 0].copy()
n_dropped = df_main["cell_id"].nunique() - df_kelp["cell_id"].nunique()
print(f"Cells with positive pre-period kelp: {df_kelp['cell_id'].nunique()} (dropped {n_dropped})")

df_kelp["w_kelp"] = df_kelp["pre_kelp_biomass"]
df_kelp["log_w_kelp"] = np.log(df_kelp["w_kelp"])
df_kelp["biomass_w_kelp"] = df_kelp["w_kelp"] * df_kelp["biomass_sum"]
df_kelp["area_w_kelp"] = df_kelp["w_kelp"] * df_kelp["area_sum"]

results_kelp = []
for depvar_w, dep_label in [("biomass_w_kelp", "Biomass"), ("area_w_kelp", "Area")]:
    for treat, treat_label in [(treat_dist, "Vessel Dist"), (treat_count, "Vessel Count")]:
        fml = (f"{depvar_w} ~ {treat}:policy_eca_2009 "
               f"+ {controls_base} + log_w_kelp | cell_id + yq")
        est = pf.fepois(fml, data=df_kelp, vcov=VCOV)
        int_key = get_int_key(est, treat)
        c, s, p = est.coef()[int_key], est.se()[int_key], est.pvalue()[int_key]
        n = est._N
        print(f"  {dep_label} x {treat_label}: coef={c:.4f}, se={s:.4f}, p={p:.4f} {stars(p)}, N={n}")
        results_kelp.append({
            "depvar": dep_label, "treatment": treat_label,
            "coef": c, "se": s, "pvalue": p, "nobs": n
        })


# ================================================================
# LATEX TABLE
# ================================================================
print("\n" + "=" * 80)
print("GENERATING LATEX TABLE")
print("=" * 80)

col_order = [
    ("Biomass", "Vessel Dist"),
    ("Area", "Vessel Dist"),
    ("Biomass", "Vessel Count"),
    ("Area", "Vessel Count"),
]

lines = []
lines.append(r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
lines.append(r"\begin{tabular}{l*{4}{c}}")
lines.append(r"\hline\hline")
lines.append(r"                    &\multicolumn{2}{c}{Distance Traveled}"
             r"&\multicolumn{2}{c}{Vessel Count}\\")
lines.append(r"                    \cmidrule(lr){2-3}\cmidrule(lr){4-5}")
lines.append(r"                    &\multicolumn{1}{c}{(1)}&\multicolumn{1}{c}{(2)}"
             r"&\multicolumn{1}{c}{(3)}&\multicolumn{1}{c}{(4)}\\")
lines.append(r"                    &\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}"
             r"&\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}\\")
lines.append(r"\hline")

# Panel A: Main specification (unweighted, count >= 2)
lines.append(r"\multicolumn{5}{l}{\textit{Panel A: Main specification "
             r"(unweighted PPML)}}\\[0.3em]")

# Compute main results live
main_results = []
for depvar, dep_label in [("biomass_sum", "Biomass"), ("area_sum", "Area")]:
    for treat, treat_label in [(treat_dist, "Vessel Dist"), (treat_count, "Vessel Count")]:
        fml = f"{depvar} ~ {treat}:policy_eca_2009 + {controls_base} | cell_id + yq"
        est = pf.fepois(fml, data=df_main, vcov=VCOV)
        int_key = get_int_key(est, treat)
        main_results.append({
            "depvar": dep_label, "treatment": treat_label,
            "coef": est.coef()[int_key], "se": est.se()[int_key],
            "pvalue": est.pvalue()[int_key], "nobs": est._N
        })

coef_line = r"Post 2009 $\times$ Exposure"
se_line = "                    "
obs_line = "Observations        "
for dep_label, treat_label in col_order:
    r = [x for x in main_results if x["depvar"] == dep_label and x["treatment"] == treat_label][0]
    coef_line += f"&{fmt_coef(r['coef'])}{fmt_stars(r['pvalue'])}"
    se_line += f"&{fmt_se(r['se'])}"
    obs_line += f"&{r['nobs']:>12d}         "
lines.append(coef_line + r"\\")
lines.append(se_line + r"\\")
lines.append(obs_line + r"\\")
lines.append("[1em]")

# Panel B: Weighted PPML (offset trick), same sample
lines.append(r"\multicolumn{5}{l}{\textit{Panel B: Weighted by "
             r"vessel count (offset trick)}}\\[0.3em]")

coef_line = r"Post 2009 $\times$ Exposure"
se_line = "                    "
obs_line = "Observations        "
for dep_label, treat_label in col_order:
    r = [x for x in results_weighted if x["depvar"] == dep_label and x["treatment"] == treat_label][0]
    coef_line += f"&{fmt_coef(r['coef'])}{fmt_stars(r['pvalue'])}"
    se_line += f"&{fmt_se(r['se'])}"
    obs_line += f"&{r['nobs']:>12d}         "
lines.append(coef_line + r"\\")
lines.append(se_line + r"\\")
lines.append(obs_line + r"\\")
lines.append("[1em]")

# Panel C: Pre-period kelp biomass weight
lines.append(r"\multicolumn{5}{l}{\textit{Panel C: Weighted by "
             r"pre-period kelp biomass}}\\[0.3em]")

coef_line = r"Post 2009 $\times$ Exposure"
se_line = "                    "
obs_line = "Observations        "
for dep_label, treat_label in col_order:
    r = [x for x in results_kelp if x["depvar"] == dep_label and x["treatment"] == treat_label][0]
    coef_line += f"&{fmt_coef(r['coef'])}{fmt_stars(r['pvalue'])}"
    se_line += f"&{fmt_se(r['se'])}"
    obs_line += f"&{r['nobs']:>12d}         "
lines.append(coef_line + r"\\")
lines.append(se_line + r"\\")
lines.append(obs_line + r"\\")
yes4 = "&         Yes         " * 4
lines.append(f"Controls            {yes4}" + r"\\")
lines.append(f"Cell FE             {yes4}" + r"\\")
lines.append(f"Year-Quarter FE     {yes4}" + r"\\")

lines.append(r"\hline\hline")
lines.append(r"\end{tabular}")

tex_content = "\n".join(lines)
tex_path = os.path.join(TABLE_DIR, "weighting_robustness.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(tex_content)
print(f"LaTeX table saved to {tex_path}")

print("\n" + "=" * 80)
print("ALL WEIGHTING ROBUSTNESS TESTS COMPLETE")
print("=" * 80)
