"""
Referee robustness: does the main 2009 continuous-DiD result survive controlling
for DENSITY-DEPENDENT kelp growth?

Concern (editor / R1): high pre-policy shipping-exposure cells start with ~5x LOWER
baseline kelp. If depleted stands rebound faster PROPORTIONALLY (density dependence),
high-exposure/low-baseline cells would show larger proportional gains regardless of
shipping, confounding the positive exposure:post coefficient.

This script:
  0. Replicates the main spec sample (81 cells, 3880 obs) and baseline coefficients.
  1. Adds log pre-policy baseline biomass x post to the main regression (core test).
  2. Uses baseline-kelp-quartile x post dummies (flexible / nonlinear rebound).
  3. Interacts log baseline biomass with time (linear trend and full year-quarter FE).
  4. Stratifies: exposure:post within high- vs low-baseline subsamples.
  5. Reports corr(exposure treatment, log baseline kelp) — the collinearity R1 worries about.

Does NOT touch the numbered pipeline or paper/*.tex. Outputs go to output/tables
and output/csv under content-descriptive names.
"""

import os
import numpy as np
import pandas as pd
import pyfixest as pf
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED = os.path.join(BASE, "data", "generated")
TABLE_DIR = os.path.join(BASE, "output", "tables")
CSV_DIR = os.path.join(BASE, "output", "csv")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

VCOV = {"CRV1": "cell_id"}
TREAT_D = "ln_mean_length_2009_jan_jun_20km"  # distance
TREAT_C = "ln_mean_count_2009_jan_jun_20km"   # count
CONTROLS = "ln_sst + ln_precipitation + ln_swh"
FE = "cell_id + yq"


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# =====================================================================
# 0. BUILD THE MAIN-SPEC SAMPLE EXACTLY (mirrors code/2_a_estim.py)
# =====================================================================
df = pd.read_csv(os.path.join(GENERATED, "new_model_kelp_exposure_3.csv"))
df = df[(df["year"] <= 2016) & (df["latitude"] >= 32) & (df["latitude"] <= 36)].copy()

df["cell_id"] = df["eca_index_official"].astype(str) + "_" + df["year_eca"].astype(str)
df["yq"] = df["year"].astype(str) + "_" + df["quarter"].astype(str)
df = df.drop_duplicates(subset=["cell_id", "yq"])

df["policy_eca_2009"] = 0
df.loc[df["year"] > 2008, "policy_eca_2009"] = 1
df.loc[(df["year"] == 2009) & (df["quarter"] < 3), "policy_eca_2009"] = 0

key_cols = ["biomass_sum", "area_sum", "model_2008",
            TREAT_D, TREAT_C, "ln_sst_mean", "ln_precipitation_mean",
            "ln_swh_mean", "ln_sum_count_2009_jan_jun_20km"]
df = df.dropna(subset=key_cols)

df["ln_sst"] = df["ln_sst_mean"]
df["ln_precipitation"] = df["ln_precipitation_mean"]
df["ln_swh"] = df["ln_swh_mean"]

# Sample restriction: cells with >= 2 vessels/month
cell_count = df.groupby("cell_id")[TREAT_C].first()
cells_above = cell_count[cell_count >= np.log(2)].index
df = df[df["cell_id"].isin(cells_above)].copy()

print("=" * 78)
print("SAMPLE CHECK")
print("=" * 78)
print(f"  obs   = {len(df)}   (expected 3880)")
print(f"  cells = {df['cell_id'].nunique()}   (expected 81)")
assert len(df) == 3880, "Sample size mismatch!"
assert df["cell_id"].nunique() == 81, "Cell count mismatch!"

# ---- Baseline (pre-policy) biomass per cell: mean 2005-2008, logged ----
pre = df[df["year"] <= 2008].copy()
base_bio = pre.groupby("cell_id")["biomass_sum"].mean()
# 3 cells have exactly zero mean pre-policy biomass (genuinely depleted stands,
# the extreme low-baseline cells). Use log1p so log(0)=0 stays as the legitimate
# minimum; for the million-kg nonzero cells log1p ~ log to <1e-5.
ln_base_bio = np.log1p(base_bio)
ln_base_bio.name = "ln_base_bio"
df = df.merge(ln_base_bio.reset_index(), on="cell_id", how="left")
# Standardize baseline (mean 0, sd 1) so its :post coef is per-SD interpretable
df["ln_base_bio_z"] = (df["ln_base_bio"] - df["ln_base_bio"].mean()) / df["ln_base_bio"].std()
df["base_x_post"] = df["ln_base_bio_z"] * df["policy_eca_2009"]

# Baseline quartiles (cell level)
cell_base = df.groupby("cell_id")["ln_base_bio"].first()
base_q = pd.qcut(cell_base, 4, labels=[1, 2, 3, 4]).astype(int)
base_q.name = "base_q"
df = df.merge(base_q.reset_index(), on="cell_id", how="left")
for q in [1, 2, 3, 4]:
    df[f"baseq{q}_post"] = ((df["base_q"] == q) & (df["policy_eca_2009"] == 1)).astype(int)

# Linear time trend (period index) for test 3
period_order = sorted(df["yq"].unique(), key=lambda s: (int(s.split("_")[0]), int(s.split("_")[1])))
period_map = {p: i for i, p in enumerate(period_order)}
df["t"] = df["yq"].map(period_map).astype(float)
df["base_x_t"] = df["ln_base_bio_z"] * df["t"]


def run(depvar, treat, extra="", data=None):
    d = df if data is None else data
    fml = f"{depvar} ~ {treat}:policy_eca_2009{extra} + {CONTROLS} | {FE}"
    est = pf.fepois(fml, data=d, vcov=VCOV)
    key = [k for k in est.coef().index if treat in k and "policy_eca_2009" in k][0]
    return est, est.coef()[key], est.se()[key], est.pvalue()[key], est._N


SPECS = [
    ("biomass_sum", TREAT_D, "Biomass x Distance", 0.136),
    ("area_sum",    TREAT_D, "Area x Distance",    0.132),
    ("biomass_sum", TREAT_C, "Biomass x Count",    0.155),
    ("area_sum",    TREAT_C, "Area x Count",       0.149),
]

records = []  # collect for CSV


def report(tag, depvar, treat, label, coef, se, p, n, base_coef=None, note=""):
    line = f"  {label:<22} coef={coef:+.4f} ({se:.4f}) p={p:.4f} {stars(p):<3} N={n}"
    if base_coef is not None:
        line += f"   [baseline was {base_coef:+.3f}]"
    if note:
        line += f"   {note}"
    print(line)
    records.append({"test": tag, "outcome": depvar, "treatment": treat, "label": label,
                    "coef": coef, "se": se, "pvalue": p, "nobs": n})


# =====================================================================
# BASELINE (confirm reproduction)
# =====================================================================
print("\n" + "=" * 78)
print("TEST 0 — BASELINE (reproduce main table before adding controls)")
print("=" * 78)
base_coefs = {}
for depvar, treat, label, target in SPECS:
    est, c, s, p, n = run(depvar, treat)
    base_coefs[(depvar, treat)] = c
    report("baseline", depvar, treat, label, c, s, p, n, note=f"(target {target:+.3f})")

# =====================================================================
# TEST 1 — log baseline biomass (z) x post, added linearly
# =====================================================================
print("\n" + "=" * 78)
print("TEST 1 — add  ln(baseline biomass, z) x post   [CORE TEST]")
print("=" * 78)
for depvar, treat, label, _ in SPECS:
    est, c, s, p, n = run(depvar, treat, extra=" + base_x_post")
    bk = [k for k in est.coef().index if k == "base_x_post"][0]
    bc, bp = est.coef()[bk], est.pvalue()[bk]
    report("test1_base_x_post", depvar, treat, label, c, s, p, n,
           base_coef=base_coefs[(depvar, treat)],
           note=f"| base:post={bc:+.4f} p={bp:.3f}{stars(bp)}")

# =====================================================================
# TEST 2 — baseline-kelp-quartile x post dummies (nonlinear rebound)
# =====================================================================
print("\n" + "=" * 78)
print("TEST 2 — add baseline-quartile x post dummies (Q1..Q4; nonlinear)")
print("=" * 78)
# Include Q1..Q4 x post; drop Q1 to avoid collinearity w/ absorbed policy main effect
quart_extra = " + baseq2_post + baseq3_post + baseq4_post"
for depvar, treat, label, _ in SPECS:
    est, c, s, p, n = run(depvar, treat, extra=quart_extra)
    report("test2_quartile_post", depvar, treat, label, c, s, p, n,
           base_coef=base_coefs[(depvar, treat)])

# =====================================================================
# TEST 3 — baseline biomass x time (linear trend, then full time FE)
# =====================================================================
print("\n" + "=" * 78)
print("TEST 3a — add ln(baseline) x linear time trend")
print("=" * 78)
for depvar, treat, label, _ in SPECS:
    est, c, s, p, n = run(depvar, treat, extra=" + base_x_t")
    report("test3a_base_x_trend", depvar, treat, label, c, s, p, n,
           base_coef=base_coefs[(depvar, treat)])

print("\n" + "=" * 78)
print("TEST 3b — interact ln(baseline) with YEAR dummies (i(year, ln_base_bio_z, ref=2008))")
print("        high/low-baseline cells free to follow different ANNUAL trajectories")
print("        (full year-quarter interaction is singular on this 81-cell panel)")
print("=" * 78)
for depvar, treat, label, _ in SPECS:
    fml = (f"{depvar} ~ {treat}:policy_eca_2009 + i(year, ln_base_bio_z, ref=2008) "
           f"+ {CONTROLS} | {FE}")
    try:
        est = pf.fepois(fml, data=df, vcov=VCOV)
        key = [k for k in est.coef().index if treat in k and "policy_eca_2009" in k][0]
        c, s, p, n = est.coef()[key], est.se()[key], est.pvalue()[key], est._N
        report("test3b_base_x_yearFE", depvar, treat, label, c, s, p, n,
               base_coef=base_coefs[(depvar, treat)])
    except Exception as e:
        print(f"  {label:<22} SINGULAR / not identified ({type(e).__name__})")
        records.append({"test": "test3b_base_x_yearFE", "outcome": depvar,
                        "treatment": treat, "label": label, "coef": np.nan,
                        "se": np.nan, "pvalue": np.nan, "nobs": np.nan})

# =====================================================================
# TEST 4 — stratify at median baseline kelp
# =====================================================================
print("\n" + "=" * 78)
print("TEST 4 — split 81 cells at MEDIAN baseline kelp; estimate exposure:post in each")
print("=" * 78)
med = cell_base.median()
low_cells = cell_base[cell_base <= med].index   # low baseline
high_cells = cell_base[cell_base > med].index   # high baseline
df_low = df[df["cell_id"].isin(low_cells)].copy()
df_high = df[df["cell_id"].isin(high_cells)].copy()
print(f"  Low-baseline subsample : {df_low['cell_id'].nunique()} cells, {len(df_low)} obs")
print(f"  High-baseline subsample: {df_high['cell_id'].nunique()} cells, {len(df_high)} obs")
for depvar, treat, label, _ in SPECS:
    est_l, cl, sl, pl, nl = run(depvar, treat, data=df_low)
    est_h, ch, sh, ph, nh = run(depvar, treat, data=df_high)
    report("test4_lowbase", depvar, treat, label, cl, sl, pl, nl,
           base_coef=base_coefs[(depvar, treat)], note="[LOW baseline]")
    report("test4_highbase", depvar, treat, label, ch, sh, ph, nh,
           base_coef=base_coefs[(depvar, treat)], note="[HIGH baseline]")

# =====================================================================
# TEST 5 — collinearity: corr(exposure, log baseline kelp)
# =====================================================================
print("\n" + "=" * 78)
print("TEST 5 — corr(exposure treatment, log baseline biomass)  [cell level, n=81]")
print("=" * 78)
cell_d = df.groupby("cell_id")[TREAT_D].first()
cell_c = df.groupby("cell_id")[TREAT_C].first()
cb = cell_base
for name, v in [("Distance (length)", cell_d), ("Count", cell_c)]:
    pear = np.corrcoef(v, cb)[0, 1]
    sp = pd.Series(v).rank().corr(pd.Series(cb).rank())
    print(f"  {name:<20}: Pearson r = {pear:+.3f}   Spearman rho = {sp:+.3f}")
    records.append({"test": "test5_corr", "outcome": "", "treatment": name,
                    "label": "corr(exposure, ln_base_bio)", "coef": pear,
                    "se": np.nan, "pvalue": sp, "nobs": 81})

# Save all coefficients to CSV
out_csv = os.path.join(CSV_DIR, "density_dependence_tests.csv")
pd.DataFrame(records).to_csv(out_csv, index=False)
print(f"\nAll coefficients saved to {out_csv}")

# =====================================================================
# Compact LaTeX summary table (biomass + area, both treatments)
# =====================================================================
def cell_from(tag, depvar, treat):
    r = next((x for x in records if x["test"] == tag and x["outcome"] == depvar
              and x["treatment"] == treat), None)
    if r is None or pd.isna(r["coef"]):
        return "--"
    return f"{r['coef']:.3f}{{\\sym{{{stars(r['pvalue'])}}}}}"

rows = [
    ("Baseline (main spec)", "baseline"),
    ("+ ln(baseline) $\\times$ post", "test1_base_x_post"),
    ("+ baseline-quartile $\\times$ post", "test2_quartile_post"),
    ("+ ln(baseline) $\\times$ trend", "test3a_base_x_trend"),
    ("+ ln(baseline) $\\times$ year FE", "test3b_base_x_yearFE"),
    ("Low-baseline subsample", "test4_lowbase"),
    ("High-baseline subsample", "test4_highbase"),
]
L = []
L.append(r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
L.append(r"\begin{tabular}{l*{4}{c}}")
L.append(r"\hline\hline")
L.append(r" &\multicolumn{2}{c}{Distance Traveled}&\multicolumn{2}{c}{Vessel Count}\\")
L.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
L.append(r" &Biomass&Area&Biomass&Area\\")
L.append(r"\hline")
for lab, tag in rows:
    L.append(f"{lab:<40}"
             f"&{cell_from(tag,'biomass_sum',TREAT_D)}"
             f"&{cell_from(tag,'area_sum',TREAT_D)}"
             f"&{cell_from(tag,'biomass_sum',TREAT_C)}"
             f"&{cell_from(tag,'area_sum',TREAT_C)}" + r"\\")
L.append(r"\hline\hline")
L.append(r"\end{tabular}")
tex_path = os.path.join(TABLE_DIR, "density_dependence_robustness.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print(f"Summary LaTeX table saved to {tex_path}")
print("\nDONE.")
