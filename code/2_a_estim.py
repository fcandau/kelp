"""
Generate final LaTeX tables for the paper.

1. Main table: Vessel Traffic and Vessel Count (4 columns, unweighted PPML)
   Sample restricted to cells with >= 2 vessels/month (count threshold).
   -> output/tables/ship_exposure_kelp_2009_main.tex

2. Weighted PPML table (appendix, 4 columns: vessel distance + count)
   Uses offset trick with log vessel count as weight.
   -> output/tables/cdid_weighted.tex

3. Event study figure:
   - Main: PPML vessel traffic (unweighted) -> output/figures/event_study_main.png

4. Placebo test (unweighted PPML, fake treatment 2007Q3)

Standard errors clustered at the cell level.
"""

import os
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED = os.path.join(BASE, 'data', 'generated')
TABLE_DIR = os.path.join(BASE, 'output', 'tables')
FIG_DIR = os.path.join(BASE, 'output', 'figures')
CSV_DIR = os.path.join(BASE, 'output', 'csv')
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


# -- Data preparation -------------------------------------------------------
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

# Alias column names for cleaner formulas
df["ln_sst"] = df["ln_sst_mean"]
df["ln_precipitation"] = df["ln_precipitation_mean"]
df["ln_swh"] = df["ln_swh_mean"]

# -- Sample restriction: cells with >= 2 vessels/month ---------------------
# ln(count) >= ln(2) <=> count >= 2
VESSEL_THRESHOLD = np.log(2)
treat_count_var = "ln_mean_count_2009_jan_jun_20km"
cell_count = df.groupby("cell_id")[treat_count_var].first()
cells_above = cell_count[cell_count >= VESSEL_THRESHOLD].index
df = df[df["cell_id"].isin(cells_above)].copy()

print(f"Sample: {len(df)} obs, {df['cell_id'].nunique()} cells")
print(f"  (Restricted to cells with >= 2 vessels/month; threshold = ln(2) = {VESSEL_THRESHOLD:.4f})")

# Variance-covariance: cluster at cell level
VCOV = {"CRV1": "cell_id"}


# -- Stars and formatting ---------------------------------------------------
def stars(pval):
    if pval < 0.01: return "***"
    elif pval < 0.05: return "**"
    elif pval < 0.10: return "*"
    return ""


def fmt_stars(pval):
    """Return \\sym{...} only when significant, empty string otherwise."""
    s = stars(pval)
    return f"\\sym{{{s}}}" if s else ""


def fmt_coef(val):
    if abs(val) >= 1:
        return f"{val:>11.3f}"
    elif abs(val) >= 0.1:
        return f"{val:>11.4f}"
    else:
        return f"{val:>11.4f}"


def fmt_se(val):
    if abs(val) >= 1:
        return f"{'(' + f'{val:.3f}' + ')':>21}"
    else:
        return f"{'(' + f'{val:.4f}' + ')':>21}"


def get_int_key(est, treatment, policy_var="policy_eca_2009"):
    for key in est.coef().index:
        if treatment in key and policy_var in key:
            return key
    raise KeyError(f"No interaction key found for '{treatment}' x '{policy_var}' "
                   f"in coefficients: {list(est.coef().index)}")


# ================================================================
# TABLE 1: Main table (unweighted PPML, vessel traffic + count)
# ================================================================
print("\n" + "=" * 80)
print("GENERATING MAIN TABLE (unweighted PPML, count >= 2 sample)")
print("=" * 80)

controls = "ln_sst + ln_precipitation + ln_swh"
fe = "cell_id + yq"

specs_main = []
spec_info_main = [
    ("biomass_sum", "ln_mean_length_2009_jan_jun_20km", "Biomass"),
    ("area_sum",    "ln_mean_length_2009_jan_jun_20km", "Area"),
    ("biomass_sum", "ln_mean_count_2009_jan_jun_20km",  "Biomass"),
    ("area_sum",    "ln_mean_count_2009_jan_jun_20km",  "Area"),
]

for depvar, treat, label in spec_info_main:
    fml = f"{depvar} ~ {treat}:policy_eca_2009 + {controls} | {fe}"
    est = pf.fepois(fml, data=df, vcov=VCOV)
    specs_main.append(est)
    int_key = get_int_key(est, treat)
    c = est.coef()[int_key]
    p = est.pvalue()[int_key]
    print(f"  {label} x {treat[:15]}... : coef={c:.4f}, p={p:.4f} {stars(p)}")

# Build LaTeX
lines = []
lines.append(r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
lines.append(r"\begin{tabular}{l*{4}{c}}")
lines.append(r"\hline\hline")
lines.append(r"                    &\multicolumn{2}{c}{Distance Traveled}&\multicolumn{2}{c}{Vessel Count}\\")
lines.append(r"                    \cmidrule(lr){2-3}\cmidrule(lr){4-5}")
lines.append(r"                    &\multicolumn{1}{c}{(1)}&\multicolumn{1}{c}{(2)}&\multicolumn{1}{c}{(3)}&\multicolumn{1}{c}{(4)}\\")
lines.append(r"                    &\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}&\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}\\")
lines.append(r"\hline")

# Single treatment row with all 4 coefficients
treat_label = r"Post 2009 $\times$ Exposure"
coef_line = f"{treat_label:<20}"
se_line = "                    "

for i, (depvar, t, _) in enumerate(spec_info_main):
    est = specs_main[i]
    int_key = get_int_key(est, t)
    c = est.coef()[int_key]
    s = est.se()[int_key]
    p = est.pvalue()[int_key]
    coef_line += f"&{fmt_coef(c)}{fmt_stars(p)}"
    se_line += f"&{fmt_se(s)}"

lines.append(coef_line + r"\\")
lines.append(se_line + r"\\")
lines.append("[1em]")

# Control rows
ctrl_names = [("ln_sst", "SST"), ("ln_precipitation", "Precipitation"),
              ("ln_swh", "Significant Wave Height")]

for ctrl, ctrl_label in ctrl_names:
    coef_line = f"{ctrl_label:<20}"
    se_line = "                    "

    for est in specs_main:
        coefs = est.coef()
        ses = est.se()
        pvals = est.pvalue()
        if ctrl in coefs.index:
            c = coefs[ctrl]
            s = ses[ctrl]
            p = pvals[ctrl]
            coef_line += f"&{fmt_coef(c)}{fmt_stars(p)}"
            se_line += f"&{fmt_se(s)}"
        else:
            coef_line += "&                     "
            se_line += "&                     "

    lines.append(coef_line + r"\\")
    lines.append(se_line + r"\\")
    lines.append("[1em]")

# Footer: FE and controls indicators
lines.append(r"\hline")
for row_label in ["Cell FE", "Year-Quarter FE"]:
    ind_line = f"{row_label:<20}"
    for _ in specs_main:
        ind_line += f"&{'Yes':>12}         "
    lines.append(ind_line + r"\\")
obs_line = "Observations        "
for est in specs_main:
    obs_line += f"&{est._N:>12d}         "
lines.append(obs_line + r"\\")
lines.append(r"\hline\hline")

lines.append(r"\end{tabular}")

main_tex = "\n".join(lines)
main_path = os.path.join(TABLE_DIR, "ship_exposure_kelp_2009_main.tex")
with open(main_path, "w", encoding="utf-8") as f:
    f.write(main_tex)
print(f"\nMain table saved to {main_path}")


# ================================================================
# EVENT STUDY FIGURES using pf.iplot()
# ================================================================
print("\n" + "=" * 80)
print("GENERATING EVENT STUDY FIGURES (unweighted PPML)")
print("=" * 80)


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

# Clean x-axis labels: iplot uses 0-indexed positions for the i() coefficients.
# Periods in sorted order: 2005, 2006, 2007, 2009.1, 2009.2, 2010, ..., 2016
# That's 12 coefficients (2008 is the omitted reference).
TICK_LABELS = ["2005", "2006", "2007", "2009Q1–Q2", "2009Q3–Q4",
               "2010", "2011", "2012", "2013", "2014", "2015", "2016"]
# Line separates the pre-policy years from 2009: between 2007 (index 2) and 2009Q1-Q2 (index 3)
TREAT_XINTERCEPT = 2.5


def pretrend_wald_test(est):
    """Joint Wald chi-squared test that pre-period coefficients = 0 (using full VCV)."""
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


def annotate_pretrend(ax, est):
    """Add pre-trend F-test annotation to an axis."""
    chi2, p = pretrend_wald_test(est)
    if not np.isnan(chi2):
        ax.text(0.02, 0.98, f"Pre-trend test: p = {p:.3f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    return chi2, p


def style_es_axis(ax, title, tick_labels=TICK_LABELS):
    """Apply consistent styling to an event study axis."""
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=45,
                       ha="right", rotation_mode="anchor")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Coefficient", fontsize=10)


# -- Estimate event studies ---------------------------------------------------
treat_vd = "ln_mean_length_2009_jan_jun_20km"
treat_vc = "ln_mean_count_2009_jan_jun_20km"
es_controls = "ln_sst + ln_precipitation + ln_swh"
es_fe = "FE_i + FE_t"

est_models = {}
for depvar, dep_label in [("biomass_sum", "Biomass"), ("area_sum", "Area")]:
    for treat, treat_label in [(treat_vd, "vessel_dist"), (treat_vc, "vessel_count")]:
        fml = (f"{depvar} ~ i(period, {treat}, ref=2008.0) "
               f"+ {es_controls} | {es_fe}")
        est = pf.fepois(fml, data=df, vcov=VCOV)
        key = f"{treat_label}_{dep_label.lower()}"
        est_models[key] = est

        chi2, p = pretrend_wald_test(est)
        int_coefs = [(k, est.coef()[k], est.pvalue()[k])
                     for k in est.coef().index if "period" in k.lower() or "2009" in k]
        print(f"\n  {dep_label} x {treat_label}:")
        for k, c, pv in int_coefs:
            print(f"    {c:>8.4f} ({est.se()[k]:.4f}) {stars(pv)}")
        print(f"    Pre-trend F-test: chi2={chi2:.2f}, p={p:.4f}")

# -- Export event study coefficients to CSV -----------------------------------
for key, est in est_models.items():
    params = est.coef()
    se_vals = est.se()
    pvals = est.pvalue()
    csv_path = os.path.join(CSV_DIR, f"ppml_es_{key}.csv")
    pd.DataFrame({
        "coef_name": params.index,
        "coef": params.values,
        "se": se_vals.values,
        "pval": pvals.values,
    }).to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

# Also save under the legacy "pre_blob" filenames (used by 5_financial_crisis.py)
for dep_label in ["biomass", "area"]:
    key = f"vessel_dist_{dep_label}"
    if key in est_models:
        est = est_models[key]
        params = est.coef()
        se_vals = est.se()
        pvals = est.pvalue()
        csv_path = os.path.join(CSV_DIR,
                                f"ppml_es_pre_blob_vessel_dist_{dep_label}.csv")
        pd.DataFrame({
            "coef_name": params.index,
            "coef": params.values,
            "se": se_vals.values,
            "pval": pvals.values,
        }).to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}")


# -- Figure 1: Main event study (vessel traffic, biomass + area) -------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, dep_label in enumerate(["Biomass", "Area"]):
    est = est_models[f"vessel_dist_{dep_label.lower()}"]
    pf.iplot(est, alpha=0.05, coord_flip=False, ax=axes[i],
             xintercept=TREAT_XINTERCEPT)
    annotate_pretrend(axes[i], est)
    style_es_axis(axes[i], f"Panel {chr(65+i)}: {dep_label}")

fig.suptitle("Event Study: Distance Traveled and Kelp Outcomes (PPML)",
             fontsize=12, y=1.02)
fig.tight_layout()
path = os.path.join(FIG_DIR, "event_study_main.png")
fig.savefig(path, dpi=300, bbox_inches="tight")
print(f"\nEvent study figure saved to {path}")
plt.close(fig)


# ================================================================
# APPENDIX: Weighted PPML (offset trick)
# ================================================================
print("\n" + "=" * 80)
print("GENERATING WEIGHTED PPML TABLE (appendix)")
print("=" * 80)

# Weighted PPML uses the full sample with w > 0 (broader than count >= 2)
# But for consistency, we use the same count >= 2 sample + weights
df["w"] = df["ln_sum_count_2009_jan_jun_20km"]
assert (df["w"] > 0).all(), "Some weights are non-positive on the count >= 2 sample"
df["log_w"] = np.log(df["w"])
df["biomass_w"] = df["w"] * df["biomass_sum"]
df["area_w"] = df["w"] * df["area_sum"]

controls_w = "ln_sst + ln_precipitation + ln_swh + log_w"

specs_w = []
spec_info_w = [
    ("biomass_w", "ln_mean_length_2009_jan_jun_20km", "Biomass"),
    ("area_w",    "ln_mean_length_2009_jan_jun_20km", "Area"),
    ("biomass_w", "ln_mean_count_2009_jan_jun_20km",  "Biomass"),
    ("area_w",    "ln_mean_count_2009_jan_jun_20km",  "Area"),
]

for depvar, treat, label in spec_info_w:
    fml = f"{depvar} ~ {treat}:policy_eca_2009 + {controls_w} | {fe}"
    est = pf.fepois(fml, data=df, vcov=VCOV)
    specs_w.append(est)
    int_key = get_int_key(est, treat)
    c = est.coef()[int_key]
    p = est.pvalue()[int_key]
    print(f"  {label} x {treat[:15]}... : coef={c:.4f}, p={p:.4f} {stars(p)}")

# Build LaTeX (4 columns, same structure as main table)
lines_w = []
lines_w.append(r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
lines_w.append(r"\begin{tabular}{l*{4}{c}}")
lines_w.append(r"\hline\hline")
lines_w.append(r"                    &\multicolumn{2}{c}{Distance Traveled}&\multicolumn{2}{c}{Vessel Count}\\")
lines_w.append(r"                    \cmidrule(lr){2-3}\cmidrule(lr){4-5}")
lines_w.append(r"                    &\multicolumn{1}{c}{(1)}&\multicolumn{1}{c}{(2)}&\multicolumn{1}{c}{(3)}&\multicolumn{1}{c}{(4)}\\")
lines_w.append(r"                    &\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}&\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}\\")
lines_w.append(r"\hline")

treat_label_w = r"Post 2009 $\times$ Exposure"
coef_line = f"{treat_label_w:<20}"
se_line = "                    "
for i, (depvar, t, _) in enumerate(spec_info_w):
    est = specs_w[i]
    int_key = get_int_key(est, t)
    c = est.coef()[int_key]
    s = est.se()[int_key]
    p = est.pvalue()[int_key]
    coef_line += f"&{fmt_coef(c)}{fmt_stars(p)}"
    se_line += f"&{fmt_se(s)}"
lines_w.append(coef_line + r"\\")
lines_w.append(se_line + r"\\")
lines_w.append("[1em]")

ctrl_names_w = [("ln_sst", "SST"), ("ln_precipitation", "Precipitation"),
                ("ln_swh", "Significant Wave Height")]
for ctrl, ctrl_label in ctrl_names_w:
    coef_line = f"{ctrl_label:<20}"
    se_line = "                    "
    for est in specs_w:
        coefs = est.coef()
        ses = est.se()
        pvals = est.pvalue()
        if ctrl in coefs.index:
            c = coefs[ctrl]
            s = ses[ctrl]
            p = pvals[ctrl]
            coef_line += f"&{fmt_coef(c)}{fmt_stars(p)}"
            se_line += f"&{fmt_se(s)}"
        else:
            coef_line += "&                     "
            se_line += "&                     "
    lines_w.append(coef_line + r"\\")
    lines_w.append(se_line + r"\\")
    lines_w.append("[1em]")

lines_w.append(r"\hline")
for row_label in ["Cell FE", "Year-Quarter FE"]:
    ind_line = f"{row_label:<20}"
    for _ in specs_w:
        ind_line += f"&{'Yes':>12}         "
    lines_w.append(ind_line + r"\\")
obs_line = "Observations        "
for est in specs_w:
    obs_line += f"&{est._N:>12d}         "
lines_w.append(obs_line + r"\\")
lines_w.append(r"\hline\hline")
lines_w.append(r"\end{tabular}")

w_tex = "\n".join(lines_w)
w_path = os.path.join(TABLE_DIR, "cdid_weighted.tex")
with open(w_path, "w", encoding="utf-8") as f:
    f.write(w_tex)
print(f"\nWeighted table saved to {w_path}")


print("\nDone. All tables and figures generated.")


# ============================================================
# PLACEBO TEST: Fake treatment date (2007Q3) on pre-period only
# ============================================================
print("\n" + "=" * 80)
print("PLACEBO TEST: Fake treatment date (2007Q3) on pre-period only (2005-2008)")
print("=" * 80)

# Restrict to pre-policy period: year <= 2008
df_placebo = df[df["year"] <= 2008].copy()

# Define fake "post" indicator: 1 for 2007Q3 onward, 0 before
df_placebo["fake_post"] = 0
df_placebo.loc[(df_placebo["year"] == 2007) & (df_placebo["quarter"] >= 3), "fake_post"] = 1
df_placebo.loc[df_placebo["year"] >= 2008, "fake_post"] = 1

print(f"Placebo sample: {len(df_placebo)} obs, {df_placebo['cell_id'].nunique()} cells")
print(f"  Fake pre (2005Q1-2007Q2): {(df_placebo['fake_post'] == 0).sum()} obs")
print(f"  Fake post (2007Q3-2008Q4): {(df_placebo['fake_post'] == 1).sum()} obs")

# Unweighted PPML (same as main spec)
placebo_controls = "ln_sst + ln_precipitation + ln_swh"
placebo_fe = "cell_id + yq"

placebo_specs = [
    ("biomass_sum", "ln_mean_length_2009_jan_jun_20km", "Biomass", "Vessel Distance"),
    ("area_sum",    "ln_mean_length_2009_jan_jun_20km", "Area",    "Vessel Distance"),
    ("biomass_sum", "ln_mean_count_2009_jan_jun_20km",  "Biomass", "Vessel Count"),
    ("area_sum",    "ln_mean_count_2009_jan_jun_20km",  "Area",    "Vessel Count"),
]

print(f"\n{'Outcome':<10} {'Treatment':<20} {'Coef':>10} {'SE':>10} {'p-value':>10} {'Sig':>5}")
print("-" * 70)

placebo_results = []
for depvar, treat, dep_label, treat_label in placebo_specs:
    fml = f"{depvar} ~ {treat}:fake_post + {placebo_controls} | {placebo_fe}"
    est = pf.fepois(fml, data=df_placebo, vcov=VCOV)

    # Find the interaction key
    int_key = None
    for key in est.coef().index:
        if treat in key and "fake_post" in key:
            int_key = key
            break

    if int_key is None:
        print(f"  WARNING: interaction key not found for {dep_label} x {treat_label}")
        continue

    c = est.coef()[int_key]
    s = est.se()[int_key]
    p = est.pvalue()[int_key]
    n = est._N

    placebo_results.append({
        "outcome": dep_label, "treatment": treat_label,
        "coef": c, "se": s, "pvalue": p, "nobs": n,
    })
    print(f"{dep_label:<10} {treat_label:<20} {c:>10.4f} {s:>10.4f} {p:>10.4f} {stars(p):>5}")

print("-" * 70)
print("Placebo coefficients should be near zero and insignificant.")
print("=" * 80)

# --- Export placebo table as LaTeX ---
plines = []
plines.append(r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
plines.append(r"\begin{tabular}{l*{4}{c}}")
plines.append(r"\hline\hline")
plines.append(r"                    &\multicolumn{2}{c}{Distance Traveled}&\multicolumn{2}{c}{Vessel Count}\\")
plines.append(r"                    \cmidrule(lr){2-3}\cmidrule(lr){4-5}")
plines.append(r"                    &\multicolumn{1}{c}{(1)}&\multicolumn{1}{c}{(2)}&\multicolumn{1}{c}{(3)}&\multicolumn{1}{c}{(4)}\\")
plines.append(r"                    &\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}&\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}\\")
plines.append(r"\hline")

# Treatment row (order: dist-bio, dist-area, count-bio, count-area)
coef_line = r"Fake Post 2007Q3 $\times$ Exposure"
se_line = "                    "
for r in placebo_results:
    c, s, p = r["coef"], r["se"], r["pvalue"]
    coef_line += f"&{fmt_coef(c)}{fmt_stars(p)}"
    se_line += f"&{fmt_se(s)}"
plines.append(coef_line + r"\\")
plines.append(se_line + r"\\")
plines.append("[1em]")

# Observations and indicator rows
obs_line = "Observations        "
for r in placebo_results:
    obs_line += f"&{r['nobs']:>12d}         "
plines.append(obs_line + r"\\")
yes4 = "&         Yes         " * 4
plines.append(f"Controls            {yes4}" + r"\\")
plines.append(f"Cell FE             {yes4}" + r"\\")
plines.append(f"Year-Quarter FE     {yes4}" + r"\\")
plines.append(r"\hline\hline")
plines.append(r"\end{tabular}")

placebo_path = os.path.join(TABLE_DIR, "placebo_test.tex")
with open(placebo_path, "w", encoding="utf-8") as f:
    f.write("\n".join(plines))
print(f"Placebo table saved to {placebo_path}")
