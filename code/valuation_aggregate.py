"""
valuation_aggregate.py -- Defensible aggregate valuation of the ECA-induced kelp
improvement, replacing the flawed conclusion footnote that multiplied total
hectares by the 15-17% per-unit-log-exposure semi-elasticity.

The 15-17% is a MARGINAL semi-elasticity (per unit of log exposure), not a
study-area-wide percentage gain. To aggregate, we back out each cell's no-policy
counterfactual canopy from its OBSERVED post-2009 canopy by removing the
estimated exposure effect relative to the LEAST-exposed in-sample cells:
    counterfactual_i = observed_post_i / exp(beta_A * (X_i - X_ref))
    gain_i           = observed_post_i * (1 - exp(-beta_A * (X_i - X_ref)))
Aggregate gain (ha) x $72,000/ha/yr (Eger et al. 2023) = annual service value.

IMPORTANT (audit C1): the year-quarter fixed effects absorb the COMMON post-2009
change, so beta identifies only the exposure GRADIENT of that change -- never its
level. Backing out relative to the least-exposed cell therefore drops that
reference cell's own (unidentified) benefit. The resulting figure is a
CONSERVATIVE LOWER BOUND on the canopy benefit attributable to the shipping-
exposure gradient, excluding any coast-wide component. It is not "the total value
of the ECA." We report x_ref = min (all gains >= 0) and x_ref = p10 (net of the
few sub-reference cells), and the COMPUTED aggregate % of counterfactual canopy.

Output (nothing written to the paper): output/csv/valuation_aggregate.csv
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

VALUE_PER_HA = 72000.0   # Eger et al. (2023) point estimate, $/ha/year (single value-transfer figure)
M2_PER_HA = 10000.0
TREAT_D = "ln_mean_length_2009_jan_jun_20km"
TREAT_C = "ln_mean_count_2009_jan_jun_20km"

# ---- Load + full lat 32-36 region (before count>=2 filter) ----
raw = pd.read_csv(os.path.join(GENERATED, "new_model_kelp_exposure_3.csv"))
raw = raw[(raw["year"] <= 2016) & (raw["latitude"] >= 32) & (raw["latitude"] <= 36)].copy()
raw["cell_id"] = raw["eca_index_official"].astype(str) + "_" + raw["year_eca"].astype(str)
raw["yq"] = raw["year"].astype(str) + "_" + raw["quarter"].astype(str)
raw = raw.drop_duplicates(subset=["cell_id", "yq"])
raw["policy_eca_2009"] = 0
raw.loc[raw["year"] > 2008, "policy_eca_2009"] = 1
raw.loc[(raw["year"] == 2009) & (raw["quarter"] < 3), "policy_eca_2009"] = 0

# ---- 81-cell estimation sample (main spec) ----
key = ["biomass_sum", "area_sum", "model_2008", TREAT_D, TREAT_C,
       "ln_sst_mean", "ln_precipitation_mean", "ln_swh_mean", "ln_sum_count_2009_jan_jun_20km"]
df = raw.dropna(subset=key).copy()
df["ln_sst"] = df["ln_sst_mean"]; df["ln_precipitation"] = df["ln_precipitation_mean"]; df["ln_swh"] = df["ln_swh_mean"]
cc = df.groupby("cell_id")[TREAT_C].first()
df = df[df["cell_id"].isin(cc[cc >= np.log(2)].index)].copy()
assert df["cell_id"].nunique() == 81 and len(df) == 3880, "sample mismatch"
print("[sample] 81 cells / 3880 obs confirmed")

# ---- Estimate the AREA coefficients (canopy area is what the $/ha valuation needs) ----
def area_beta(treat):
    est = pf.fepois(f"area_sum ~ {treat}:policy_eca_2009 + ln_sst + ln_precipitation + ln_swh | cell_id + yq",
                    data=df, vcov={"CRV1": "cell_id"})
    k = [c for c in est.coef().index if treat in c and "policy_eca_2009" in c][0]
    return est.coef()[k]

beta_dist, beta_count = area_beta(TREAT_D), area_beta(TREAT_C)
print(f"[area coefs] distance={beta_dist:.4f}  count={beta_count:.4f}")
# W5: guard against silent estimation drift feeding the valuation
assert abs(beta_dist - 0.132) < 0.01 and abs(beta_count - 0.149) < 0.01, "area betas drifted from published"

# ---- Reconcile total study-area canopy (the '5,500 ha' claim) ----
def canopy_ha(data, how):
    g = data.groupby("cell_id")["area_sum"]
    per_cell = g.mean() if how == "mean" else g.max()
    return per_cell.sum() / M2_PER_HA

print("\n=== TOTAL STUDY-AREA CANOPY (reconcile the 5,500 ha claim) ===")
recon = {}
for label, data in [("81-cell estimation sample", df), ("full lat 32-36 region", raw)]:
    mean_ha, peak_ha = canopy_ha(data, "mean"), canopy_ha(data, "max")
    recon[label] = (mean_ha, peak_ha)
    print(f"  {label:28s}: seasonal-mean {mean_ha:8.0f} ha | peak {peak_ha:8.0f} ha")
# W3: units sanity check -- peak full-region canopy must be in a plausible hectare range
peak_full = recon["full lat 32-36 region"][1]
assert 1000 < peak_full < 50000, f"units look wrong: peak full-region canopy = {peak_full:.0f} (expected 1e3-5e4 ha)"

# ---- Cell-level: exposure + observed post-2009 canopy area (W2: explicit subset) ----
post_area = df[df["policy_eca_2009"] == 1].groupby("cell_id")["area_sum"].mean()
cell = df.groupby("cell_id").agg(X_d=(TREAT_D, "first"), X_c=(TREAT_C, "first")).reset_index()
cell = cell.merge(post_area.rename("area_post").reset_index(), on="cell_id")

def aggregate_gain(beta, Xcol, x_ref):
    X = cell[Xcol].values
    ap = cell["area_post"].values                         # observed post-2009 mean canopy (m2)
    gain_m2 = ap * (1.0 - np.exp(-beta * (X - x_ref)))
    total_gain_ha = gain_m2.sum() / M2_PER_HA
    total_cf_ha = (ap - gain_m2).sum() / M2_PER_HA        # counterfactual canopy
    pct = 100.0 * total_gain_ha / total_cf_ha             # COMPUTED aggregate % improvement
    return total_gain_ha, pct, total_gain_ha * VALUE_PER_HA / 1e6

print("\n=== DEFENSIBLE AGGREGATE (conservative LOWER BOUND on the exposure-gradient benefit) ===")
records = []
for name, beta, Xcol in [("Distance", beta_dist, "X_d"), ("Count", beta_count, "X_c")]:
    for ref_name, x_ref in [("min", cell[Xcol].min()), ("p10 (net)", np.percentile(cell[Xcol], 10))]:
        ha, pct, val = aggregate_gain(beta, Xcol, x_ref)
        print(f"  {name:8s} | ref={ref_name:9s}: +{ha:6.0f} ha  "
              f"({pct:4.1f}% of counterfactual canopy)  = ${val:5.1f}M/yr")
        records.append({"treatment": name, "reference": ref_name, "x_ref": float(x_ref),
                        "ha_gained": ha, "pct_aggregate": pct, "value_musd_at72k": val})

# append reconciliation rows for a single self-contained artifact
for label, (mean_ha, peak_ha) in recon.items():
    records.append({"treatment": "RECON", "reference": label, "x_ref": np.nan,
                    "ha_gained": np.nan, "pct_aggregate": np.nan,
                    "value_musd_at72k": np.nan, "mean_ha": mean_ha, "peak_ha": peak_ha})
pd.DataFrame(records).to_csv(os.path.join(CSV_DIR, "valuation_aggregate.csv"), index=False)
print(f"\nSaved {os.path.join(CSV_DIR, 'valuation_aggregate.csv')}")
print("\nFraming for the paper:")
print("  - The COMPUTED aggregate % (above) is the honest study-area gain; the 15-17% was")
print("    only ever a per-unit-log-exposure marginal effect, never a study-area percentage.")
print("  - Figure is a conservative LOWER BOUND (excludes the coast-wide component absorbed")
print("    by time effects); $72k/ha is a single Eger value-transfer estimate and satellite")
print("    canopy area is a lower-bound proxy for service-relevant forest area.")
