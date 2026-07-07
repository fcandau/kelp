"""
Continuous-treatment DiD estimand check (Callaway, Goodman-Bacon & Sant'Anna 2024).

The main 2009 specification is a PPML continuous-DiD with a treatment that enters
LINEARLY in log-exposure: `exposure : policy_eca_2009`. This delivers a single
coefficient. Callaway, Goodman-Bacon & Sant'Anna (2024, "Difference-in-Differences
with a Continuous Treatment") point out that such a linear-in-dose object is an
"average causal response" (ACR): (a) it is point-identified only under a STRONG
parallel-trends assumption (treatment-effect homogeneity across dosage levels), and
(b) as a linear TWFE-style functional it can be a weighted average of underlying
causal responses with possibly-negative weights. The safer, ATT-type target is the
"average LEVEL effect" -- the effect of a high dosage relative to a (near-)zero /
lowest dosage -- which is identified under ORDINARY parallel trends.

This script does three things and touches NOTHING in the numbered pipeline:

  1. Reproduces the baseline LINEAR coefficients (+0.136/+0.132/+0.155/+0.149).
  2. NONPARAMETRIC DOSE-RESPONSE: partitions cells into dosage groups (terciles;
     quartiles as robustness), interacts group dummies with the post-2009 indicator,
     and estimates group-specific post effects with the lowest-dosage group as base.
     If the group effects rise with dosage, the linear coefficient is not a
     functional-form artifact.
  3. AVERAGE LEVEL EFFECT: the top-vs-bottom dosage-group coefficient (identified
     under ordinary parallel trends), contrasted with the linear ACR coefficient
     (the linear model's implied top-vs-bottom prediction is computed for comparison).

Outputs:
  output/csv/doseresponse_2009.csv            (all group effects, terciles + quartiles)
  output/tables/tab_doseresponse_2009.tex     (group-specific dose-response table)

Unweighted PPML throughout, SEs clustered at the cell level (matches main spec).
Python: pyfixest. Run with the r-env interpreter.
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


def stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Data preparation -- identical to code/2_a_estim.py main spec
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(GENERATED, "new_model_kelp_exposure_3.csv"))
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

# Sample restriction: cells with >= 2 vessels/month
VESSEL_THRESHOLD = np.log(2)
cell_count = df.groupby("cell_id")["ln_mean_count_2009_jan_jun_20km"].first()
cells_above = cell_count[cell_count >= VESSEL_THRESHOLD].index
df = df[df["cell_id"].isin(cells_above)].copy()

N_OBS = len(df)
N_CELLS = df["cell_id"].nunique()
print(f"Sample: {N_OBS} obs, {N_CELLS} cells")
assert N_OBS == 3880, f"Expected 3880 obs, got {N_OBS}"
assert N_CELLS == 81, f"Expected 81 cells, got {N_CELLS}"
print("Self-check PASSED: 3880 obs, 81 cells.\n")

CONTROLS = "ln_sst + ln_precipitation + ln_swh"
FE = "cell_id + yq"

# Treatment / outcome grid
TREATS = [
    ("ln_mean_length_2009_jan_jun_20km", "Distance Traveled"),
    ("ln_mean_count_2009_jan_jun_20km",  "Vessel Count"),
]
OUTCOMES = [
    ("biomass_sum", "Biomass"),
    ("area_sum",    "Area"),
]

# ---------------------------------------------------------------------------
# STEP 1 -- Reproduce the baseline LINEAR (ACR) coefficients
# ---------------------------------------------------------------------------
print("=" * 78)
print("STEP 1  Baseline linear continuous-DiD (ACR) coefficients")
print("=" * 78)
print(f"{'Outcome':<10}{'Treatment':<20}{'coef':>10}{'se':>10}{'p':>10}   sig")
print("-" * 78)

linear_coef = {}  # (treat_var, outcome_var) -> dict
for tvar, tlab in TREATS:
    for yvar, ylab in OUTCOMES:
        fml = f"{yvar} ~ {tvar}:policy_eca_2009 + {CONTROLS} | {FE}"
        est = pf.fepois(fml, data=df, vcov=VCOV)
        key = next(k for k in est.coef().index
                   if tvar in k and "policy_eca_2009" in k)
        c, s, p = est.coef()[key], est.se()[key], est.pvalue()[key]
        linear_coef[(tvar, yvar)] = {"coef": c, "se": s, "p": p}
        print(f"{ylab:<10}{tlab:<20}{c:>10.4f}{s:>10.4f}{p:>10.4f}   {stars(p)}")

print("\nExpected (memo): Distance +0.136/+0.132 ; Count +0.155/+0.149")
print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# STEP 2 -- Nonparametric dose-response by dosage group
# ---------------------------------------------------------------------------
def assign_dose_groups(data, tvar, n_groups):
    """Assign each cell a dosage group (1..n_groups) from quantiles of the
    cell-level (time-invariant) treatment value. Returns a Series aligned to data
    plus a summary of cell counts / mean exposure per group."""
    cellval = data.groupby("cell_id")[tvar].first()
    grp = pd.qcut(cellval, q=n_groups, labels=list(range(1, n_groups + 1)))
    grp = grp.astype(int)
    summ = pd.DataFrame({
        "group": grp.values,
        "cellval": cellval.values,
    }).groupby("group").agg(n_cells=("cellval", "size"),
                            mean_exposure=("cellval", "mean"),
                            min_exposure=("cellval", "min"),
                            max_exposure=("cellval", "max"))
    mapping = grp.to_dict()
    return data["cell_id"].map(mapping).astype(int), summ


def dose_response(data, yvar, tvar, n_groups):
    """Estimate group-specific post-2009 LEVEL effects (base = lowest group).
    Uses i(dose_group, policy, ref=1): each coefficient is the DiD of group g
    versus the lowest-dosage group -- an ATT-type average level effect,
    identified under ordinary parallel trends."""
    d = data.copy()
    d["dose_group"], summ = assign_dose_groups(d, tvar, n_groups)
    fml = (f"{yvar} ~ i(dose_group, policy_eca_2009, ref=1) "
           f"+ {CONTROLS} | {FE}")
    est = pf.fepois(fml, data=d, vcov=VCOV)

    coefs, ses, pvals = est.coef(), est.se(), est.pvalue()
    ci = est.confint(alpha=0.05)
    rows = []
    # Reference group (group 1): effect fixed at 0 by construction
    rows.append({"group": 1, "coef": 0.0, "se": np.nan, "p": np.nan,
                 "ci_low": np.nan, "ci_high": np.nan,
                 "n_cells": int(summ.loc[1, "n_cells"]),
                 "mean_exposure": float(summ.loc[1, "mean_exposure"]),
                 "is_ref": True})
    for g in range(2, n_groups + 1):
        key = next(k for k in coefs.index
                   if f"dose_group" in k and (f"[{g}]" in k or k.endswith(str(g)))
                   and "policy_eca_2009" in k)
        rows.append({"group": g, "coef": coefs[key], "se": ses[key],
                     "p": pvals[key],
                     "ci_low": ci.loc[key].iloc[0], "ci_high": ci.loc[key].iloc[1],
                     "n_cells": int(summ.loc[g, "n_cells"]),
                     "mean_exposure": float(summ.loc[g, "mean_exposure"]),
                     "is_ref": False})
    return pd.DataFrame(rows), est


print("=" * 78)
print("STEP 2  Nonparametric dose-response (group-specific level effects)")
print("=" * 78)

all_rows = []
dr_results = {}  # (tvar, yvar, n_groups) -> DataFrame
for n_groups, gname in [(3, "tercile"), (4, "quartile")]:
    for tvar, tlab in TREATS:
        for yvar, ylab in OUTCOMES:
            res, est = dose_response(df, yvar, tvar, n_groups)
            dr_results[(tvar, yvar, n_groups)] = res
            # monotonicity check on point estimates (group 1 = 0 baseline)
            vals = res.sort_values("group")["coef"].values
            monotone = bool(np.all(np.diff(vals) >= 0))
            all_positive = bool(np.all(vals[1:] > 0))
            top = res[res["group"] == n_groups].iloc[0]
            print(f"\n[{gname}] {ylab} x {tlab}   "
                  f"(monotone_up={monotone}, all_pos={all_positive})")
            for _, r in res.iterrows():
                tag = " (ref, =0)" if r["is_ref"] else \
                      f"  {stars(r['p'])}  p={r['p']:.3f}  " \
                      f"[{r['ci_low']:.3f},{r['ci_high']:.3f}]"
                print(f"    G{int(r['group'])} "
                      f"(n={int(r['n_cells'])}, mean_x={r['mean_exposure']:.2f}): "
                      f"coef={r['coef']:>7.4f}{tag}")
            # accumulate for CSV
            for _, r in res.iterrows():
                all_rows.append({
                    "partition": gname, "n_groups": n_groups,
                    "treatment": tlab, "treatment_var": tvar,
                    "outcome": ylab, "outcome_var": yvar,
                    "group": int(r["group"]), "n_cells": int(r["n_cells"]),
                    "mean_exposure": r["mean_exposure"],
                    "coef": r["coef"], "se": r["se"], "pval": r["p"],
                    "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                    "is_reference": r["is_ref"],
                    "monotone_increasing": monotone,
                    "all_nonref_positive": all_positive,
                })

dr_csv = pd.DataFrame(all_rows)
csv_path = os.path.join(CSV_DIR, "doseresponse_2009.csv")
dr_csv.to_csv(csv_path, index=False)
print(f"\nSaved group-effect CSV -> {csv_path}")
print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# STEP 3 -- Top-vs-bottom level effect vs the linear ACR prediction (printed)
# ---------------------------------------------------------------------------
print("=" * 78)
print("STEP 3  Average LEVEL effect (top vs bottom group) vs linear ACR")
print("=" * 78)
print("Nonparametric top-vs-bottom = coef on highest dosage group (ref=lowest).")
print("Linear-implied top-vs-bottom = ACR_coef * (mean exposure gap top-bottom).\n")

for n_groups, gname in [(3, "tercile"), (4, "quartile")]:
    for tvar, tlab in TREATS:
        for yvar, ylab in OUTCOMES:
            res = dr_results[(tvar, yvar, n_groups)]
            top = res[res["group"] == n_groups].iloc[0]
            bot = res[res["group"] == 1].iloc[0]
            gap = top["mean_exposure"] - bot["mean_exposure"]
            acr = linear_coef[(tvar, yvar)]["coef"]
            print(f"[{gname}] {ylab} x {tlab}:")
            print(f"    nonparametric top-vs-bottom = {top['coef']:.4f} "
                  f"(se {top['se']:.4f}, p {top['p']:.3f} {stars(top['p'])})")
            print(f"    linear ACR                  = {acr:.4f}  "
                  f"x  exposure gap {gap:.2f}  =  {acr * gap:.4f} (implied)")
            print()
print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# LaTeX table -- group-specific dose-response (terciles Panel A, quartiles Panel B)
# ---------------------------------------------------------------------------
COLORDER = [(TREATS[0][0], OUTCOMES[0][0]), (TREATS[0][0], OUTCOMES[1][0]),
            (TREATS[1][0], OUTCOMES[0][0]), (TREATS[1][0], OUTCOMES[1][0])]


def _grp_row(label, n_groups, g):
    coefs, ses = [], []
    for tv, yv in COLORDER:
        rr = dr_results[(tv, yv, n_groups)]
        r = rr[rr["group"] == g].iloc[0]
        st = "" if pd.isna(r["p"]) else stars(r["p"])
        coefs.append(f"{r['coef']:.3f}\\sym{{{st}}}")
        ses.append("" if pd.isna(r["se"]) else f"({r['se']:.3f})")
    cline = f"{label:<22}" + "".join(f"&{c}" for c in coefs) + r"\\"
    sline = " " * 22 + "".join(f"&{c}" for c in ses) + r"\\"
    return [cline, sline]


L = [r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}",
     r"\begin{tabular}{l*{4}{c}}", r"\hline\hline",
     r" &\multicolumn{2}{c}{Distance Traveled}&\multicolumn{2}{c}{Vessel Count}\\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
     r" &\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}&\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}\\",
     r"\hline",
     r"\multicolumn{5}{l}{\emph{Panel A: Exposure terciles (base = lowest tercile)}}\\"]
L += _grp_row("Middle tercile", 3, 2) + _grp_row("Top tercile", 3, 3) + ["[0.4em]"]
L += [r"\multicolumn{5}{l}{\emph{Panel B: Exposure quartiles (base = lowest quartile)}}\\"]
L += _grp_row("Second quartile", 4, 2) + _grp_row("Third quartile", 4, 3) + _grp_row("Top quartile", 4, 4)
L += [r"\hline",
      "Linear coefficient" + "".join(f"&{linear_coef[(t, y)]['coef']:.3f}" for t, y in COLORDER) + r"\\",
      r"\hline\hline", r"\end{tabular}"]

tab_path = os.path.join(TABLE_DIR, "tab_doseresponse_2009.tex")
with open(tab_path, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print(f"Saved group-specific dose-response table -> {tab_path}")

print("\nDone.")
