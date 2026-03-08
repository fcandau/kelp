"""
Anticipation and treatment timing robustness (Appendix).

Generates:
  - output/tables/anticipation_robustness.tex  (LaTeX table: baseline vs 2012)
  - output/figures/temporal_stability_shipping.png  (Spearman bar chart)
  - output/csv/anticipation_*.csv  (CDID + event study coefficients)

All use unweighted PPML on the count >= 2 restricted sample.
"""

import os
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED = os.path.join(BASE, "data", "generated")
FIG_DIR = os.path.join(BASE, "output", "figures")
CSV_DIR = os.path.join(BASE, "output", "csv")
TABLE_DIR = os.path.join(BASE, "output", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(GENERATED, "new_model_kelp_exposure_3.csv"))
df = df[(df["year"] <= 2016) & (df["latitude"] >= 32) & (df["latitude"] <= 36)].copy()

df["cell_id"] = df["eca_index_official"].astype(str) + "_" + df["year_eca"].astype(str)
df["yq"] = df["year"].astype(str) + "_" + df["quarter"].astype(str)
df = df.drop_duplicates(subset=["cell_id", "yq"])

key_cols = ["biomass_sum", "area_sum",
            "ln_mean_length_2009_jan_jun_20km", "ln_mean_count_2009_jan_jun_20km",
            "ln_mean_length_2009_20km", "ln_mean_count_2009_20km",
            "ln_sst_mean", "ln_precipitation_mean", "ln_swh_mean",
            "ln_sum_count_2009_jan_jun_20km"]
df = df.dropna(subset=key_cols)

# Alias column names for cleaner formulas
df["ln_sst"] = df["ln_sst_mean"]
df["ln_precipitation"] = df["ln_precipitation_mean"]
df["ln_swh"] = df["ln_swh_mean"]

# -- Sample restriction: cells with >= 2 vessels/month ---------------------
VESSEL_THRESHOLD = np.log(2)
treat_count_var = "ln_mean_count_2009_jan_jun_20km"
cell_count = df.groupby("cell_id")[treat_count_var].first()
cells_above = cell_count[cell_count >= VESSEL_THRESHOLD].index
df = df[df["cell_id"].isin(cells_above)].copy()


def assign_period(row):
    if row["year"] == 2009 and row["quarter"] <= 2:
        return 2009.1
    elif row["year"] == 2009 and row["quarter"] >= 3:
        return 2009.2
    else:
        return float(row["year"])


df["period"] = df.apply(assign_period, axis=1)
df["FE_i"] = df["cell_id"]
df["FE_t"] = df["yq"]

VCOV = {"CRV1": "cell_id"}
TICK_LABELS = ["2005", "2006", "2007", "09H1", "09H2",
               "2010", "2011", "2012", "2013", "2014", "2015", "2016"]
TREAT_XINTERCEPT = 2.5


def stars(pval):
    if pval < 0.01:
        return "***"
    elif pval < 0.05:
        return "**"
    elif pval < 0.10:
        return "*"
    return ""


def fmt_stars(pval):
    s = stars(pval)
    return f"\\sym{{{s}}}" if s else ""


def fmt_coef(val):
    return f"{val:>11.4f}"


def fmt_se(val):
    if abs(val) >= 1:
        return f"{'(' + f'{val:.3f}' + ')':>21}"
    else:
        return f"{'(' + f'{val:.4f}' + ')':>21}"


def pretrend_ftest(est):
    params = est.coef()
    all_names = list(params.index)
    pre_idx = [i for i, k in enumerate(all_names)
               if any(p in k for p in ["2005", "2006", "2007"])]
    if not pre_idx:
        return np.nan, np.nan
    beta = params.values[pre_idx]
    V = est._vcov[np.ix_(pre_idx, pre_idx)]
    chi2 = float(beta @ np.linalg.inv(V) @ beta)
    p_val = 1 - scipy_stats.chi2.cdf(chi2, len(pre_idx))
    return chi2, p_val


# ================================================================
# PPML CDID: Baseline vs alternative treatment timing
# ================================================================
print("=" * 90)
print("PPML ROBUSTNESS: ALTERNATIVE TREATMENT TIMING (unweighted, count >= 2)")
print("=" * 90)
print(f"Sample: {len(df)} obs, {df['cell_id'].nunique()} cells")

df["policy_eca_2009"] = 0
df.loc[df["year"] > 2008, "policy_eca_2009"] = 1
df.loc[(df["year"] == 2009) & (df["quarter"] < 3), "policy_eca_2009"] = 0

controls = "ln_sst + ln_precipitation + ln_swh"
fe = "cell_id + yq"


def get_int_key(est, treatment):
    for key in est.coef().index:
        if treatment in key and "policy_eca_2009" in key:
            return key
    return None


# Collect results for LaTeX table
cdid_results = []

print("\n--- VESSEL DISTANCE (mean_length) ---")
print(f"{'Treatment timing':<40} {'Biomass coef':>12} {'p':>8} {'Area coef':>12} {'p':>8}")
print("-" * 90)

dist_specs = [
    ("ln_mean_length_2009_jan_jun_20km", "Jan-Jun 2009 (baseline)"),
    ("ln_mean_length_2009_20km", "Full-year 2009"),
    ("ln_mean_length_2010_20km", "2010"),
    ("ln_mean_length_2012_20km", "2012"),
    ("ln_mean_length_2014_20km", "2014"),
]

for treat_var, label in dist_specs:
    row_res = {"label": label, "type": "distance"}
    row_str = f"{label:<40}"
    for depvar in ["biomass_sum", "area_sum"]:
        dep_key = "biomass" if "biomass" in depvar else "area"
        fml = f"{depvar} ~ {treat_var}:policy_eca_2009 + {controls} | {fe}"
        try:
            est = pf.fepois(fml, data=df, vcov=VCOV)
            int_key = get_int_key(est, treat_var)
            c = est.coef()[int_key]
            s = est.se()[int_key]
            p = est.pvalue()[int_key]
            n = est._N
            row_str += f" {c:>10.4f}{stars(p):>3} {p:>8.4f}"
            row_res[f"{dep_key}_coef"] = c
            row_res[f"{dep_key}_se"] = s
            row_res[f"{dep_key}_pval"] = p
            row_res[f"{dep_key}_n"] = n
        except Exception as e:
            row_str += f" {'FAIL':>10}    {'':>8}"
    cdid_results.append(row_res)
    print(row_str)

print("\n--- VESSEL COUNT (mean_count) ---")
print(f"{'Treatment timing':<40} {'Biomass coef':>12} {'p':>8} {'Area coef':>12} {'p':>8}")
print("-" * 90)

count_specs = [
    ("ln_mean_count_2009_jan_jun_20km", "Jan-Jun 2009 (baseline)"),
    ("ln_mean_count_2009_20km", "Full-year 2009"),
    ("ln_mean_count_2010_20km", "2010"),
    ("ln_mean_count_2012_20km", "2012"),
    ("ln_mean_count_2014_20km", "2014"),
]

for treat_var, label in count_specs:
    row_res = {"label": label, "type": "count"}
    row_str = f"{label:<40}"
    for depvar in ["biomass_sum", "area_sum"]:
        dep_key = "biomass" if "biomass" in depvar else "area"
        fml = f"{depvar} ~ {treat_var}:policy_eca_2009 + {controls} | {fe}"
        try:
            est = pf.fepois(fml, data=df, vcov=VCOV)
            int_key = get_int_key(est, treat_var)
            c = est.coef()[int_key]
            s = est.se()[int_key]
            p = est.pvalue()[int_key]
            n = est._N
            row_str += f" {c:>10.4f}{stars(p):>3} {p:>8.4f}"
            row_res[f"{dep_key}_coef"] = c
            row_res[f"{dep_key}_se"] = s
            row_res[f"{dep_key}_pval"] = p
            row_res[f"{dep_key}_n"] = n
        except Exception as e:
            row_str += f" {'FAIL':>10}    {'':>8}"
    cdid_results.append(row_res)
    print(row_str)

# --- Export anticipation table as LaTeX (Panel A: baseline, Panel B: 2012) ---
baseline_dist = next(r for r in cdid_results if r["label"] == "Jan-Jun 2009 (baseline)" and r["type"] == "distance")
baseline_count = next(r for r in cdid_results if r["label"] == "Jan-Jun 2009 (baseline)" and r["type"] == "count")
alt_dist = next(r for r in cdid_results if r["label"] == "2012" and r["type"] == "distance")
alt_count = next(r for r in cdid_results if r["label"] == "2012" and r["type"] == "count")

alines = []
alines.append(r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
alines.append(r"\begin{tabular}{l*{4}{c}}")
alines.append(r"\hline\hline")
alines.append(r"                    &\multicolumn{2}{c}{Distance Traveled (km)}&\multicolumn{2}{c}{Vessel Count}\\")
alines.append(r"                    \cmidrule(lr){2-3}\cmidrule(lr){4-5}")
alines.append(r"                    &\multicolumn{1}{c}{(1)}&\multicolumn{1}{c}{(2)}&\multicolumn{1}{c}{(3)}&\multicolumn{1}{c}{(4)}\\")
alines.append(r"                    &\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}&\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}\\")
alines.append(r"\hline")

# Panel A: Baseline
alines.append(r"\multicolumn{5}{l}{\textit{Panel A: Baseline (January--June 2009)}}\\[0.3em]")
treat_label = r"Post 2009 $\times$ Exposure"
coef_line = f"{treat_label:<20}"
se_line = "                    "
for r, dep in [(baseline_dist, "biomass"), (baseline_dist, "area"),
               (baseline_count, "biomass"), (baseline_count, "area")]:
    c, s, p = r[f"{dep}_coef"], r[f"{dep}_se"], r[f"{dep}_pval"]
    coef_line += f"&{fmt_coef(c)}{fmt_stars(p)}"
    se_line += f"&{fmt_se(s)}"
alines.append(coef_line + r"\\")
alines.append(se_line + r"\\")
alines.append("[0.8em]")

# Panel B: 2012
alines.append(r"\multicolumn{5}{l}{\textit{Panel B: 2012 (post-adjustment)}}\\[0.3em]")
coef_line = f"{treat_label:<20}"
se_line = "                    "
for r, dep in [(alt_dist, "biomass"), (alt_dist, "area"),
               (alt_count, "biomass"), (alt_count, "area")]:
    c, s, p = r[f"{dep}_coef"], r[f"{dep}_se"], r[f"{dep}_pval"]
    coef_line += f"&{fmt_coef(c)}{fmt_stars(p)}"
    se_line += f"&{fmt_se(s)}"
alines.append(coef_line + r"\\")
alines.append(se_line + r"\\")
alines.append("[0.5em]")

# Footer
alines.append(r"\hline")
obs_line = "Observations        "
for r, dep in [(baseline_dist, "biomass"), (baseline_dist, "area"),
               (baseline_count, "biomass"), (baseline_count, "area")]:
    obs_line += f"&{r[f'{dep}_n']:>12d}         "
alines.append(obs_line + r"\\")
yes4 = "&         Yes         " * 4
alines.append(f"Controls            {yes4}" + r"\\")
alines.append(f"Cell FE             {yes4}" + r"\\")
alines.append(f"Year-Quarter FE     {yes4}" + r"\\")
alines.append(r"\hline\hline")
alines.append(r"\end{tabular}")

antic_path = os.path.join(TABLE_DIR, "anticipation_robustness.tex")
with open(antic_path, "w", encoding="utf-8") as f:
    f.write("\n".join(alines))
print(f"\nAnticipation table saved to {antic_path}")


# ================================================================
# EVENT STUDY: Baseline vs Full-year 2009
# ================================================================
print("\n" + "=" * 80)
print("EVENT STUDY: BASELINE vs FULL-YEAR 2009 TREATMENT")
print("=" * 80)

es_controls = "ln_sst + ln_precipitation + ln_swh"

est_dict = {}
for treat_var, treat_label in [("ln_mean_length_2009_jan_jun_20km", "Jan-Jun 2009"),
                                ("ln_mean_length_2009_20km", "Full-year 2009")]:
    for depvar, dep_label in [("biomass_sum", "Biomass"), ("area_sum", "Area")]:
        fml = (f"{depvar} ~ i(period, {treat_var}, ref=2008.0) "
               f"+ {es_controls} | FE_i + FE_t")
        est = pf.fepois(fml, data=df, vcov=VCOV)
        key = f"{treat_label}_{dep_label}"
        est_dict[key] = est
        chi2, p = pretrend_ftest(est)
        print(f"\n  {dep_label} x {treat_label}:")
        for k in est.coef().index:
            if "period" in k.lower():
                c = est.coef()[k]
                pv = est.pvalue()[k]
                print(f"    {c:>8.4f} ({est.se()[k]:.4f}) {stars(pv)}")
        print(f"    Pre-trend F-test: chi2={chi2:.2f}, p={p:.4f}")

# Also vessel count
for treat_var, treat_label in [("ln_mean_count_2009_jan_jun_20km", "Count Jan-Jun 2009"),
                                ("ln_mean_count_2009_20km", "Count Full-year 2009")]:
    fml = (f"biomass_sum ~ i(period, {treat_var}, ref=2008.0) "
           f"+ {es_controls} | FE_i + FE_t")
    est = pf.fepois(fml, data=df, vcov=VCOV)
    key = f"{treat_label}_Biomass"
    est_dict[key] = est
    chi2, p = pretrend_ftest(est)
    print(f"\n  Biomass x {treat_label}:")
    for k in est.coef().index:
        if "period" in k.lower():
            c = est.coef()[k]
            pv = est.pvalue()[k]
            print(f"    {c:>8.4f} ({est.se()[k]:.4f}) {stars(pv)}")
    print(f"    Pre-trend F-test: chi2={chi2:.2f}, p={p:.4f}")


# ================================================================
# FIGURE 2: Temporal stability of shipping patterns (bar chart)
# ================================================================
print("\nGenerating temporal stability figure...")

cell_df = df[["eca_index_official", "year_eca",
              "ln_mean_length_2009_jan_jun_20km",
              "ln_mean_length_2009_20km",
              "ln_mean_length_2010_20km", "ln_mean_length_2011_20km",
              "ln_mean_length_2012_20km", "ln_mean_length_2013_20km",
              "ln_mean_length_2014_20km",
              "ln_mean_count_2009_jan_jun_20km",
              "ln_mean_count_2009_20km",
              "ln_mean_count_2010_20km", "ln_mean_count_2011_20km",
              "ln_mean_count_2012_20km", "ln_mean_count_2013_20km",
              "ln_mean_count_2014_20km"]].drop_duplicates(
    subset=["eca_index_official", "year_eca"])

years = ["2009\n(full)", "2010", "2011", "2012", "2013", "2014"]
dist_vars = ["ln_mean_length_2009_20km", "ln_mean_length_2010_20km",
             "ln_mean_length_2011_20km", "ln_mean_length_2012_20km",
             "ln_mean_length_2013_20km", "ln_mean_length_2014_20km"]
count_vars = ["ln_mean_count_2009_20km", "ln_mean_count_2010_20km",
              "ln_mean_count_2011_20km", "ln_mean_count_2012_20km",
              "ln_mean_count_2013_20km", "ln_mean_count_2014_20km"]

spearman_dist = []
spearman_count = []
for dv, cv in zip(dist_vars, count_vars):
    valid_d = cell_df[["ln_mean_length_2009_jan_jun_20km", dv]].dropna()
    sr_d, _ = spearmanr(valid_d["ln_mean_length_2009_jan_jun_20km"], valid_d[dv])
    spearman_dist.append(sr_d)

    valid_c = cell_df[["ln_mean_count_2009_jan_jun_20km", cv]].dropna()
    sr_c, _ = spearmanr(valid_c["ln_mean_count_2009_jan_jun_20km"], valid_c[cv])
    spearman_count.append(sr_c)

fig2, ax2 = plt.subplots(figsize=(8, 5))
x2 = np.arange(len(years))
width = 0.35
bars1 = ax2.bar(x2 - width / 2, spearman_dist, width,
                label="Distance Traveled", color="#1f77b4", alpha=0.8)
bars2 = ax2.bar(x2 + width / 2, spearman_count, width,
                label="Vessel Count", color="#ff7f0e", alpha=0.8)

ax2.axhline(y=0.9, color="green", linewidth=1, linestyle="--", alpha=0.5,
            label="Threshold (0.9)")
ax2.set_ylabel("Spearman Rank Correlation", fontsize=11)
ax2.set_xlabel("Year of Treatment Measurement", fontsize=11)
ax2.set_title("Temporal Stability of Cell-Level Shipping Rankings\n"
              "(Reference: Jan-Jun 2009)", fontsize=12)
ax2.set_xticks(x2)
ax2.set_xticklabels(years, fontsize=9)
ax2.set_ylim(0.75, 1.02)
ax2.legend(fontsize=9, loc="lower left")

for bar in bars1:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
             f"{height:.2f}", ha="center", va="bottom", fontsize=7)
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
             f"{height:.2f}", ha="center", va="bottom", fontsize=7)

fig2.tight_layout()
path2 = os.path.join(FIG_DIR, "temporal_stability_shipping.png")
fig2.savefig(path2, dpi=300, bbox_inches="tight")
print(f"Saved to {path2}")
plt.close(fig2)


# ================================================================
# Save coefficient CSVs
# ================================================================
print("\nSaving coefficient CSVs...")

for key, est in est_dict.items():
    safe_key = key.replace(" ", "_").replace("-", "_").lower()
    params = est.coef()
    se_vals = est.se()
    pvals = est.pvalue()
    pd.DataFrame({
        "coef_name": params.index,
        "coef": params.values,
        "se": se_vals.values,
        "pval": pvals.values,
    }).to_csv(os.path.join(CSV_DIR, f"anticipation_{safe_key}.csv"), index=False)
    print(f"  Saved anticipation_{safe_key}.csv")

# Save CDID results
cdid_df = pd.DataFrame(cdid_results)
cdid_df.to_csv(os.path.join(CSV_DIR, "anticipation_cdid_comparison.csv"), index=False)
print(f"  Saved anticipation_cdid_comparison.csv")

# Save temporal stability results
stab_df = pd.DataFrame({
    "year": years,
    "spearman_distance": spearman_dist,
    "spearman_count": spearman_count,
})
stab_df.to_csv(os.path.join(CSV_DIR, "temporal_stability_shipping.csv"), index=False)
print(f"  Saved temporal_stability_shipping.csv")

# ================================================================
# TREATMENT CORRELATION: 2009 vs 2011 exposure variables
# ================================================================
print("\n" + "=" * 80)
print("TREATMENT CORRELATION: 2009 vs 2011 EXPOSURE VARIABLES")
print("=" * 80)

corr_pairs = [
    ("ln_mean_length_2009_jan_jun_20km", "ln_mean_length_2011_20km", "Distance Traveled"),
    ("ln_mean_count_2009_jan_jun_20km", "ln_mean_count_2011_20km", "Vessel Count"),
]

corr_results = []
print(f"\n{'Measure':<25} {'Pearson r':>12} {'p':>10} {'Spearman rho':>14} {'p':>10} {'N':>6}")
print("-" * 80)

for var09, var11, label in corr_pairs:
    valid = cell_df[[var09, var11]].dropna()
    valid = valid[(valid[var09] > 0) & (valid[var11] > 0)]
    n = len(valid)
    pr, pp = pearsonr(valid[var09], valid[var11])
    sr, sp = spearmanr(valid[var09], valid[var11])
    corr_results.append({
        "measure": label, "pearson_r": pr, "pearson_p": pp,
        "spearman_rho": sr, "spearman_p": sp, "n": n,
    })
    print(f"{label:<25} {pr:>12.4f} {pp:>10.4f} {sr:>14.4f} {sp:>10.4f} {n:>6d}")

# Save CSV
corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(os.path.join(CSV_DIR, "treatment_correlation_2009_2011.csv"), index=False)
print(f"\nSaved treatment_correlation_2009_2011.csv")

# Generate scatter plot
fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (var09, var11, label), res in zip(axes3, corr_pairs, corr_results):
    valid = cell_df[[var09, var11]].dropna()
    valid = valid[(valid[var09] > 0) & (valid[var11] > 0)]
    ax.scatter(valid[var09], valid[var11], alpha=0.5, s=20, color="#1f77b4")
    lims = [min(valid[var09].min(), valid[var11].min()),
            max(valid[var09].max(), valid[var11].max())]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel(f"{label}\n(Jan-Jun 2009)", fontsize=10)
    ax.set_ylabel(f"{label}\n(Full-year 2011)", fontsize=10)
    ax.set_title(f"{label}", fontsize=11)
    ax.text(0.05, 0.92,
            f"Pearson r = {res['pearson_r']:.3f}\nSpearman $\\rho$ = {res['spearman_rho']:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

fig3.suptitle("Treatment Variable Correlation: 2009 vs 2011", fontsize=12, y=1.02)
fig3.tight_layout()
path3 = os.path.join(FIG_DIR, "treatment_correlation_2009_2011.png")
fig3.savefig(path3, dpi=300, bbox_inches="tight")
print(f"Saved to {path3}")
plt.close(fig3)

print("\n" + "=" * 80)
print("DONE: Anticipation robustness analysis complete.")
print("=" * 80)
