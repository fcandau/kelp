"""
PPML Event Study Robustness Checks

Uses pf.iplot() for standard economics event study plots.

Additional specifications to strengthen the vessel traffic event study:
  1. Without ln_swh (extends sample to 2005-2016 instead of 2005-2014)
  2. With cell-specific linear time trends (absorb differential trajectories)
  3. Restricted sample: 2005-2012 (avoid warm blob contamination)
  4. model_2008 with and without vessel traffic control

All use unweighted PPML on the count >= 2 restricted sample.
"""

import os
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings("ignore")

# -- Paths ------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED = os.path.join(BASE, 'data', 'generated')
FIG_DIR = os.path.join(BASE, 'output', 'figures')
CSV_DIR = os.path.join(BASE, 'output', 'csv')
TABLE_DIR = os.path.join(BASE, 'output', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


# -- Load data --------------------------------------------------------------
df = pd.read_csv(os.path.join(GENERATED, 'new_model_kelp_exposure_3.csv'))
df = df[(df["year"] <= 2016) & (df["latitude"] >= 32) & (df["latitude"] <= 36)].copy()

df["cell_id"] = df["eca_index_official"].astype(str) + "_" + df["year_eca"].astype(str)
df["yq"] = df["year"].astype(str) + "_" + df["quarter"].astype(str)
df = df.drop_duplicates(subset=["cell_id", "yq"])

df["FE_i"] = df["cell_id"]
df["FE_t"] = df["yq"]

# Alias column names for cleaner formulas
for col_short, col_long in [("ln_sst", "ln_sst_mean"),
                             ("ln_precipitation", "ln_precipitation_mean"),
                             ("ln_swh", "ln_swh_mean")]:
    if col_long in df.columns and col_short not in df.columns:
        df[col_short] = df[col_long]

# -- Sample restriction: cells with >= 2 vessels/month ---------------------
VESSEL_THRESHOLD = np.log(2)
treat_count_var = "ln_mean_count_2009_jan_jun_20km"
cell_count = df.groupby("cell_id")[treat_count_var].first()
cells_above = cell_count[cell_count >= VESSEL_THRESHOLD].index
df = df[df["cell_id"].isin(cells_above)].copy()

VCOV = {"CRV1": "FE_i"}


def assign_period(row):
    if row["year"] == 2009 and row["quarter"] <= 2:
        return 2009.1
    elif row["year"] == 2009 and row["quarter"] >= 3:
        return 2009.2
    else:
        return float(row["year"])


df["period"] = df.apply(assign_period, axis=1)

# Cell-specific linear time trend variable
df["time_trend"] = df["year"] + (df["quarter"] - 1) / 4.0


# -- Tick labels per max year -----------------------------------------------
TICK_2014 = ["2005", "2006", "2007", "09H1", "09H2",
             "2010", "2011", "2012", "2013", "2014"]
TICK_2016 = ["2005", "2006", "2007", "09H1", "09H2",
             "2010", "2011", "2012", "2013", "2014", "2015", "2016"]
TICK_2012 = ["2005", "2006", "2007", "09H1", "09H2",
             "2010", "2011", "2012"]
TREAT_XINTERCEPT = 2.5  # between 2007 (idx 2) and 09H1 (idx 3)


# -- Helper functions -------------------------------------------------------
def pretrend_ftest(est):
    """Joint Wald test that pre-period coefficients = 0 (proper, using full VCV)."""
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


def stars(pval):
    if pval < 0.01: return "***"
    elif pval < 0.05: return "**"
    elif pval < 0.10: return "*"
    return ""


def annotate_and_style(ax, est, title, tick_labels, fontsize=8):
    """Add F-test annotation and clean labels."""
    chi2, p = pretrend_ftest(est)
    if not np.isnan(chi2):
        ax.text(0.02, 0.98, f"F-test p={p:.3f}",
                transform=ax.transAxes, fontsize=fontsize, va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.5))
    ax.set_xticklabels(tick_labels, fontsize=fontsize, rotation=45)
    ax.set_title(title, fontsize=fontsize + 1)
    return chi2, p


def print_est(est, label):
    """Print coefficients from a pyfixest estimation."""
    print(f"\n  --- {label} ---")
    params = est.coef()
    se_vals = est.se()
    pvals = est.pvalue()
    for k in params.index:
        if "period" in k.lower() or any(str(y) in k for y in range(2005, 2017)):
            sig = stars(pvals[k])
            print(f"  {params[k]:>10.4f} ({se_vals[k]:.4f}) {sig}")
    chi2, p = pretrend_ftest(est)
    print(f"  Pre-trend F-test: chi2={chi2:.2f}, p={p:.4f}")


def save_coef_csv(est, filepath):
    """Save coefficient table as CSV."""
    params = est.coef()
    se_vals = est.se()
    pvals = est.pvalue()
    pd.DataFrame({
        "coef_name": params.index,
        "coef": params.values,
        "se": se_vals.values,
        "pval": pvals.values,
    }).to_csv(filepath, index=False)


def fmt_coef(v):
    return f"{v:>11.4f}" if abs(v) < 1 else f"{v:>11.3f}"

def fmt_se(v):
    inner = f"{v:.4f}" if abs(v) < 1 else f"{v:.3f}"
    return f"{'(' + inner + ')':>21}"

def fmt_stars(p):
    s = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
    return f"\\sym{{{s}}}" if s else ""


# ================================================================
# ROBUSTNESS 1: Without ln_swh (extends to 2005-2016)
# ================================================================
print("=" * 80)
print("ROBUSTNESS 1: PPML without ln_swh (2005-2016 sample)")
print("=" * 80)

ess_cols_no_swh = ["biomass_sum", "area_sum", "model_2008",
                    "ln_mean_length_2009_jan_jun_20km",
                    "ln_mean_count_2009_jan_jun_20km",
                    "ln_sst", "ln_precipitation"]
df1 = df.dropna(subset=ess_cols_no_swh).copy()

print(f"Sample: {len(df1)} obs, {df1['FE_i'].nunique()} cells, "
      f"year range: {df1['year'].min()}-{df1['year'].max()}")

est_r1 = {}
for depvar, depvar_label in [("biomass_sum", "biomass"), ("area_sum", "area")]:
    treat_var = "ln_mean_length_2009_jan_jun_20km"
    fml = (f"{depvar} ~ i(period, {treat_var}, ref=2008.0) "
           f"+ ln_sst + ln_precipitation | FE_i + FE_t")
    try:
        est = pf.fepois(fml, data=df1, vcov=VCOV)
        est_r1[depvar_label] = est
        print_est(est, f"No SWH: {depvar_label}")
        save_coef_csv(est, os.path.join(CSV_DIR,
                      f"ppml_es_no_swh_vessel_dist_{depvar_label}.csv"))
    except Exception as e:
        print(f"  FAILED ({depvar_label}): {e}")

# Also vessel count x biomass
treat_vc = "ln_mean_count_2009_jan_jun_20km"
fml_vc = (f"biomass_sum ~ i(period, {treat_vc}, ref=2008.0) "
          f"+ ln_sst + ln_precipitation | FE_i + FE_t")
try:
    est_vc = pf.fepois(fml_vc, data=df1, vcov=VCOV)
    est_r1["count_biomass"] = est_vc
    print_est(est_vc, "No SWH: count x biomass")
except Exception as e:
    print(f"  FAILED: {e}")


# ================================================================
# ROBUSTNESS 2: Cell-specific linear time trends
# ================================================================
print("\n" + "=" * 80)
print("ROBUSTNESS 2: PPML with cell-specific linear time trends")
print("=" * 80)

ess_cols_full = ["biomass_sum", "area_sum", "model_2008",
                  "ln_mean_length_2009_jan_jun_20km",
                  "ln_mean_count_2009_jan_jun_20km",
                  "ln_sst", "ln_precipitation", "ln_swh"]
df2 = df.dropna(subset=ess_cols_full).copy()

print(f"Sample: {len(df2)} obs, {df2['FE_i'].nunique()} cells, "
      f"year range: {df2['year'].min()}-{df2['year'].max()}")

est_r2 = {}
for depvar, depvar_label in [("biomass_sum", "biomass"), ("area_sum", "area")]:
    treat_var = "ln_mean_length_2009_jan_jun_20km"
    fml = (f"{depvar} ~ i(period, {treat_var}, ref=2008.0) "
           f"+ ln_sst + ln_precipitation + ln_swh "
           f"| FE_i + FE_t + i(FE_i, time_trend)")
    print(f"\n  Trying cell trends for {depvar_label}...")
    try:
        est = pf.fepois(fml, data=df2, vcov=VCOV)
        est_r2[depvar_label] = est
        print_est(est, f"Cell trends: {depvar_label}")
        save_coef_csv(est, os.path.join(CSV_DIR,
                      f"ppml_es_cell_trends_vessel_dist_{depvar_label}.csv"))
    except Exception as e:
        print(f"  FAILED ({depvar_label}): {e}")
        try:
            fml_alt = (f"{depvar} ~ i(period, {treat_var}, ref=2008.0) "
                       f"+ ln_sst + ln_precipitation + ln_swh "
                       f"| FE_i[time_trend] + FE_t")
            print(f"  Trying alternative syntax: FE_i[time_trend]...")
            est = pf.fepois(fml_alt, data=df2, vcov=VCOV)
            est_r2[depvar_label] = est
            print_est(est, f"Cell trends (alt): {depvar_label}")
        except Exception as e2:
            print(f"  Alt also FAILED: {e2}")


# ================================================================
# ROBUSTNESS 3: Restricted sample 2005-2012 (before warm blob)
# ================================================================
print("\n" + "=" * 80)
print("ROBUSTNESS 3: Restricted sample 2005-2012 (pre-warm blob)")
print("=" * 80)

df3 = df[df["year"] <= 2012].copy()
df3 = df3.dropna(subset=ess_cols_full).copy()

print(f"Sample: {len(df3)} obs, {df3['FE_i'].nunique()} cells, "
      f"year range: {df3['year'].min()}-{df3['year'].max()}")

est_r3 = {}
for depvar, depvar_label in [("biomass_sum", "biomass"), ("area_sum", "area")]:
    treat_var = "ln_mean_length_2009_jan_jun_20km"
    fml = (f"{depvar} ~ i(period, {treat_var}, ref=2008.0) "
           f"+ ln_sst + ln_precipitation + ln_swh | FE_i + FE_t")
    try:
        est = pf.fepois(fml, data=df3, vcov=VCOV)
        est_r3[depvar_label] = est
        print_est(est, f"2005-2012: {depvar_label}")
        save_coef_csv(est, os.path.join(CSV_DIR,
                      f"ppml_es_restricted_2012_vessel_dist_{depvar_label}.csv"))
    except Exception as e:
        print(f"  FAILED ({depvar_label}): {e}")


# ================================================================
# ROBUSTNESS 4: Model_2008 with and without vessel traffic control
# ================================================================
print("\n" + "=" * 80)
print("ROBUSTNESS 4: model_2008 controlling for vessel traffic")
print("=" * 80)

df4 = df.dropna(subset=ess_cols_full).copy()

est_r4 = {}

# Panel A: model_2008 alone
fml_m = ("biomass_sum ~ i(period, model_2008, ref=2008.0) "
         "+ ln_sst + ln_precipitation + ln_swh | FE_i + FE_t")
try:
    est_alone = pf.fepois(fml_m, data=df4, vcov=VCOV)
    est_r4["model_2008 alone"] = est_alone
    print_est(est_alone, "model_2008 alone")
except Exception as e:
    print(f"  FAILED: {e}")

# Panel B: model_2008 + vessel traffic control
fml_mc = ("biomass_sum ~ i(period, model_2008, ref=2008.0) "
          "+ i(period, ln_mean_length_2009_jan_jun_20km, ref=2008.0) "
          "+ ln_sst + ln_precipitation + ln_swh | FE_i + FE_t")
try:
    est_combined = pf.fepois(fml_mc, data=df4, vcov=VCOV)
    est_r4["model_2008 + vessel dist"] = est_combined
    print(f"\n  --- model_2008 controlling for vessel traffic ---")
    params = est_combined.coef()
    se_vals = est_combined.se()
    pvals = est_combined.pvalue()
    for k in params.index:
        if "model" in k.lower():
            sig = stars(pvals[k])
            print(f"  {params[k]:>10.4f} ({se_vals[k]:.4f}) {sig}")
except Exception as e:
    print(f"  FAILED: {e}")


# ================================================================
# SUMMARY: Combined robustness figure for vessel traffic
# ================================================================
print("\n" + "=" * 80)
print("VESSEL TRAFFIC EVENT STUDY: ROBUSTNESS SUMMARY")
print("=" * 80)

# Estimate baseline (with swh)
df_base = df.dropna(subset=ess_cols_full).copy()
fml_base = ("biomass_sum ~ i(period, ln_mean_length_2009_jan_jun_20km, ref=2008.0) "
            "+ ln_sst + ln_precipitation + ln_swh | FE_i + FE_t")

summary_ests = {}
try:
    est_base = pf.fepois(fml_base, data=df_base, vcov=VCOV)
    summary_ests["Baseline (w/ SWH)"] = (est_base, TICK_2016)
except Exception:
    pass

if "biomass" in est_r1:
    summary_ests["No SWH control"] = (est_r1["biomass"], TICK_2016)
if "biomass" in est_r2:
    summary_ests["Cell trends"] = (est_r2["biomass"], TICK_2016)
if "biomass" in est_r3:
    summary_ests["Pre-blob (2005-12)"] = (est_r3["biomass"], TICK_2012)

if summary_ests:
    print(f"\n  {'Spec':<20} {'Pre-trend p':>12} {'Nobs':>8}")
    for label, (est, _) in summary_ests.items():
        chi2, p = pretrend_ftest(est)
        nobs = est.nobs() if hasattr(est, 'nobs') else "?"
        print(f"  {label:<20} {p:>12.4f} {nobs:>8}")


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
