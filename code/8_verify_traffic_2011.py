"""
8_verify_traffic_2011.py -- Verify DIRECTLY in our AIS data (no reliance on
Klotz): do the high-2011-exposure cells (the Section 4 treatment cells) GAIN
absolute vessel traffic after the Dec-2011 boundary modification?

The mechanism is absolute vessel density, so the relevant first stage is in
LEVELS (additive), not the earlier PPML/log (multiplicative) first stage. The
log version was dominated by near-empty low-exposure cells whose tiny bases
produce huge proportional swings; that is why it came out negative. This
script reports BOTH so the fork is transparent for R1 #9.

Tests (2011 dropped from the traffic panel; pre = {2009,2010}, post = {2012..2016}):
  (1) Quartile pre/post ABSOLUTE means + change.
  (2) Cross-sectional: change in absolute traffic on 2011 log-exposure (OLS).
  (3) Levels DiD (feols): absolute traffic ~ exposure : post2011 | cell + year.
  (4) Proportional DiD (fepois): same, for contrast (the earlier -0.076).
Positive (1)-(3) => high-exposure cells gained absolute vessels => the
negative kelp coefficient in Table 2 is consistent with more ship -> less kelp.

Output (NOTHING written to the paper): output/csv/verify_traffic_2011.csv
"""

import os
import numpy as np
import pandas as pd
import pyfixest as pf
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED = os.path.join(BASE, "data", "generated")
CSV_DIR = os.path.join(BASE, "output", "csv")
os.makedirs(CSV_DIR, exist_ok=True)

VCOV = {"CRV1": "cell_id"}
YEARS = list(range(2009, 2017))
PRE, POST = [2009, 2010], [2012, 2013, 2014, 2015, 2016]


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ---- Load + 2011 sample (matches 3_estim2011.do), plus kelp-matched subset ----
raw = pd.read_csv(os.path.join(GENERATED, "new_model_kelp_exposure_3.csv"), low_memory=False)
raw = raw[(raw["year"] <= 2016) & (raw["latitude"] >= 32) & (raw["latitude"] <= 36)].copy()
raw["cell_id"] = raw["eca_index_official"].astype(str) + "_" + raw["year_eca"].astype(str)
raw["yq"] = raw["year"].astype(str) + "_" + raw["quarter"].astype(str)
raw = raw.drop_duplicates(subset=["cell_id", "yq"]).copy()

EXP_C11, EXP_D11 = "ln_mean_count_2011_20km", "ln_mean_length_2011_20km"
# Same cells as the 2011 kelp tables (Table 2, event study): non-missing 2011
# exposure and wave height, and positive kelp at least once (PPML drops the
# all-zero cells). 127 cells, 6,000 cell-quarters.
df11 = raw.dropna(subset=[EXP_C11, "ln_swh_mean"]).copy()
KELP_POS = set(df11.groupby("cell_id")["biomass_sum"].max().pipe(lambda s: s[s > 0]).index)
df11 = df11[df11["cell_id"].isin(KELP_POS)].copy()
cell = df11.groupby("cell_id").first().reset_index()
print(f"[2011 sample] {cell['cell_id'].nunique()} cells, {len(df11)} cell-quarters")


# ---- (1) Quartile pre/post ABSOLUTE means -------------------------------------
cell["q"] = pd.qcut(cell[EXP_C11], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
print("\n=== (1) ABSOLUTE mean vessel count by 2011-exposure quartile ===")
print(f"{'Q':4s}{'pre(09-10)':>12s}{'post(12-16)':>12s}{'change':>10s}")
for q in ["Q1", "Q2", "Q3", "Q4"]:
    sub = cell[cell["q"] == q]
    pre = np.mean([sub[f"mean_count_{y}_20km"].mean() for y in PRE])
    post = np.mean([sub[f"mean_count_{y}_20km"].mean() for y in POST])
    print(f"{q:4s}{pre:12.2f}{post:12.2f}{post - pre:+10.2f}")


# ---- Build cell x year traffic panel (2011 dropped) ---------------------------
rows = []
for _, r in cell.iterrows():
    for y in YEARS:
        rows.append({"cell_id": r["cell_id"], "year": y,
                     "traffic_count": r.get(f"mean_count_{y}_20km", np.nan),
                     "traffic_length": r.get(f"mean_length_{y}_20km", np.nan),
                     "exp_count": r[EXP_C11], "exp_length": r[EXP_D11]})
panel = pd.DataFrame(rows)
fs_panel = panel[panel["year"] != 2011].copy()
fs_panel["post2011"] = (fs_panel["year"] > 2011).astype(int)


# ---- (2) Cross-sectional: change in absolute traffic on log-exposure -----------
print("\n=== (2) Cross-sectional: d(absolute traffic) on 2011 log-exposure (OLS) ===")
xs = cell[["cell_id", EXP_C11]].dropna().copy()
xs["d_count"] = [
    np.mean([cell.loc[cell["cell_id"] == cid, f"mean_count_{y}_20km"].values[0] for y in POST]) -
    np.mean([cell.loc[cell["cell_id"] == cid, f"mean_count_{y}_20km"].values[0] for y in PRE])
    for cid in xs["cell_id"]]
xs = xs.dropna(subset=["d_count"]).copy()
xs_est = pf.feols(f"d_count ~ {EXP_C11}", data=xs, vcov="HC1")
b = xs_est.coef()[EXP_C11]; s = xs_est.se()[EXP_C11]; p = xs_est.pvalue()[EXP_C11]
print(f"  slope = {b:+.3f} ({s:.3f}) {stars(p)}  [N={len(xs)}]  (>0 => high-exposure cells gained more absolute vessels)")


# ---- (3) Levels DiD (feols) and (4) proportional DiD (fepois) ------------------
def dids(outcome, exposure, cells=None):
    d = fs_panel.dropna(subset=[outcome, exposure]).copy()
    if cells is not None:
        d = d[d["cell_id"].isin(cells)].copy()
    # Common sample: drop cells Poisson would separate (all-zero outcome across
    # retained years) so the levels-vs-proportional contrast is PURE functional
    # form, not sample composition.
    d = d[d.groupby("cell_id")[outcome].transform("max") > 0].copy()
    fml = f"{outcome} ~ {exposure}:post2011 | cell_id + year"
    lev = pf.feols(fml, data=d, vcov=VCOV)
    prop = pf.fepois(fml, data=d, vcov=VCOV)
    kl = [k for k in lev.coef().index if exposure in k and "post2011" in k][0]
    kp = [k for k in prop.coef().index if exposure in k and "post2011" in k][0]
    return (lev.coef()[kl], lev.se()[kl], lev.pvalue()[kl],
            prop.coef()[kp], prop.se()[kp], prop.pvalue()[kp],
            lev._N, prop._N, d["cell_id"].nunique())

print("\n=== (3) LEVELS DiD (feols)  vs  (4) PROPORTIONAL DiD (fepois) ===")
print("    exposure:post2011 | cell + year   [kelp-matched, common sample]")
records = []
for out, exp, tag in [("traffic_count", "exp_count", "count"),
                      ("traffic_length", "exp_length", "dist")]:
    lc, ls, lp, pc, ps, pp, lN, pN, nc = dids(out, exp, KELP_POS)
    print(f"  {tag:5s}: LEVELS {lc:+.3f} ({ls:.3f}){stars(lp):3s}  |  "
          f"PROP {pc:+.4f} ({ps:.4f}){stars(pp):3s}  [cells={nc}, N_lev={lN}, N_prop={pN}]")
    records.append({"measure": tag, "levels_coef": lc, "levels_se": ls, "levels_p": lp,
                    "prop_coef": pc, "prop_se": ps, "prop_p": pp, "cells": nc,
                    "n_levels": lN, "n_prop": pN})
pd.DataFrame(records).to_csv(os.path.join(CSV_DIR, "verify_traffic_2011.csv"), index=False)

print("\n" + "=" * 74)
print("READ (honest framing):")
print("- LEVELS DiD > 0 => in the ABSOLUTE-density metric (the stated mechanism),")
print("  high-2011-exposure cells gained more absolute vessels after 2011, so")
print("  Table 2's negative kelp coef is SIGN-consistent with more ship -> less kelp.")
print("  This answers R1 #9 as a DESCRIPTIVE sign fact.")
print("- NOT clean causal attribution to the modification: the 2-year AIS pre-window")
print("  {2009,2010} cannot separate the boundary mod from a proportional post-crisis")
print("  RECOVERY that also loads more absolute vessels onto busy lanes. A spatial")
print("  check (gain concentrated in the targeted zone vs all busy lanes) is needed.")
print("- Table 2 is itself PPML/proportional, so the COMMENSURATE first stage is the")
print("  PROPORTIONAL one (which stays negative). Do NOT build a Wald from the levels")
print("  first stage and the proportional kelp reduced form -- different questions.")
print("=" * 74)
print(f"\nSaved {os.path.join(CSV_DIR, 'verify_traffic_2011.csv')}")
print("DONE. No paper files were modified.")
