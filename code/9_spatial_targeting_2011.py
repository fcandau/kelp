"""
9_spatial_targeting_2011.py -- Spatial-targeting check for the Dec-2011 ECA
boundary modification (referee R1 follow-up to 8_verify_traffic_2011.py).

QUESTION. Script 8 showed high-2011-exposure cells gained ABSOLUTE vessel
traffic after 2011 (levels DiD > 0). But a 2-year AIS pre-window {2009,2010}
cannot separate the boundary MODIFICATION from a general post-financial-crisis
shipping RECOVERY that would also load more absolute vessels onto already-busy
lanes. This script tests whether the post-2011 absolute gain is SPATIALLY
CONCENTRATED in the modification's targeted geography (the Santa Barbara
Channel / Channel Islands corridor) versus DIFFUSE across all busy lanes.

MECHANISM (documented in the ship-strike / air-quality literature, e.g. McKenna
et al. 2012; Redfern et al. 2013). After the July-2009 California OGV fuel rule,
a large share of container traffic diverted OUT of the Santa Barbara Channel
(SBC) to a route SOUTH of the northern Channel Islands, to shorten the distance
sailed inside the low-sulfur-fuel zone. The Dec-2011 boundary modification
extended the regulated zone to the waters south of the islands, removing that
incentive, so traffic partially RETURNED to the SBC shipping lanes.
  => Modification fingerprint (NOT reproducible by a uniform recovery):
       SBC lanes GAIN absolute traffic; south-of-islands diversion route LOSES.
  => Uniform recovery fingerprint:
       every busy lane gains in proportion to its baseline (dchg_i ~= pre_i * g),
       including the (already-busy) LA/Long Beach approaches; no zone loses.

DATA. data/generated/new_model_kelp_exposure_3.csv (cell x quarter), study
sample lat 32-36, year<=2016. Cell geography from data/raw/ECA_10km.gpkg
(centroids; join key gpkg.id == csv.eca_index_official). Traffic = annual
mean_count_YYYY_20km (same variable family as the 2011 exposure).
Pre = mean of {2009,2010}; Post = mean of {2012..2016}; 2011 (transition) dropped.
Per-cell absolute change: dchg = mean(post) - mean(pre).

TARGETED-ZONE DEFINITION (approximate; documented; robustness reported).
Using gpkg cell centroids (clon, clat):
  SBC-return  (TARGETED, "ships returned here"):
      clon in [-120.80, -119.15] AND clat in [34.00, 34.70]
      = the channel between the mainland (~34.45N) and the north side of the
        northern Channel Islands (~34.0N), incl. the western approach.
  South-of-islands (TARGETED, "diversion route abandoned"):
      clon in [-120.55, -118.90] AND clat in [33.15, 34.00]
      = seaward (south) of the northern Channel Islands.
  Other / non-targeted (incl. LA/Long Beach + San Diego busy lanes):
      everything else (the busiest cells sit at clon ~ -118.3, clat ~ 33.7).
CAVEAT. There is NO exact boundary shapefile here, so the zone is an
approximation from coordinates. A ~10km-wide band sits right along the island
chain (clat 33.95-34.05) where a cell can be channel-side or seaward depending
on exact longitude relative to each island, and the 20km AIS radius blurs it
further. This mislabels a few low-traffic western island-footprint cells as SBC
(they behave like the diversion route), which BIASES AGAINST the concentration
hypothesis. We therefore report (a) full-zone results, (b) an
unambiguous-cells-only version (SBC = clat>=34.30 mainland lane; South =
clat<=33.95), and (c) sensitivity of the lat split.

OUTPUTS (NOTHING written to paper/):
  output/csv/spatial_targeting_2011.csv           (per-cell classification)
  output/csv/spatial_targeting_2011_zonestats.csv (zone summary)
  output/figures/fig_traffic_reshuffle_2011.png   (map, png only)
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import pyfixest as pf
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED = os.path.join(BASE, "data", "generated")
RAW = os.path.join(BASE, "data", "raw")
CSV_DIR = os.path.join(BASE, "output", "csv")
FIG_DIR = os.path.join(BASE, "output", "figures")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

PRE, POST = [2009, 2010], [2012, 2013, 2014, 2015, 2016]


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ---------------------------------------------------------------------------
# 1. Load, collapse to one row per cell, attach centroid geography
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(GENERATED, "new_model_kelp_exposure_3.csv"), low_memory=False)
df = df[(df["year"] <= 2016) & (df["latitude"] >= 32) & (df["latitude"] <= 36)].copy()

# TRUE CELL KEY = (eca_index_official, year_eca). The 2009 and 2011 ECA grids
# are DISTINCT spatial partitions; the same eca_index_official can appear in both
# at different locations/traffic. The Stata 2011 pipeline (3_estim2011.do) uses
# egen FE_i = group(eca_index_official year_eca), i.e. both grids as 153 cells.
CELLKEY = ["eca_index_official", "year_eca"]
traffic_cols = [f"mean_count_{y}_20km" for y in PRE + POST]
# AUDIT: annual-traffic covariates must be constant within a cell so the collapse
# is well-defined.
maxnun = df.groupby(CELLKEY)[traffic_cols + ["ln_mean_count_2011_20km"]].nunique().max().max()
assert maxnun == 1, f"traffic covariate varies within (eca,year_eca) cell (nunique={maxnun})"

c = df.groupby(CELLKEY).first().reset_index()

# gpkg cell centroids; join on BOTH id AND grid-year (57 ids exist in both grids).
g = gpd.read_file(os.path.join(RAW, "ECA_10km.gpkg"))
cent = g.to_crs(3857).geometry.centroid.to_crs(4326)  # centroid in meters, back to lon/lat
g["clon"], g["clat"] = cent.x.values, cent.y.values
g = g[["id", "year", "clon", "clat"]].drop_duplicates(subset=["id", "year"])
c = c.merge(g, left_on=CELLKEY, right_on=["id", "year"], how="left")
assert c["clon"].isna().sum() == 0, "some cells failed to match a gpkg centroid"
c["csv_lat"], c["csv_lon"] = c["latitude"], c["longitude"]  # CSV coords, for robustness

c["pre"] = c[[f"mean_count_{y}_20km" for y in PRE]].mean(axis=1)
c["post"] = c[[f"mean_count_{y}_20km" for y in POST]].mean(axis=1)
c["dchg"] = c["post"] - c["pre"]
c["lnc2011"] = c["ln_mean_count_2011_20km"]
c = c.replace([np.inf, -np.inf], np.nan)
c = c.dropna(subset=["dchg", "lnc2011"]).copy()
print(f"[sample] {len(c)} study cells with traffic change + 2011 exposure")


# ---------------------------------------------------------------------------
# 2. Zone classification (documented above)
# ---------------------------------------------------------------------------
def classify(clon, clat, lat_split=34.00):
    if (-120.80 <= clon <= -119.15) and (lat_split <= clat <= 34.70):
        return "SBC_targeted"
    if (-120.55 <= clon <= -118.90) and (33.15 <= clat < lat_split):
        return "South_islands"
    return "Other"

c["zone"] = [classify(lo, la) for lo, la in zip(c["clon"], c["clat"])]
c["sbc"] = (c["zone"] == "SBC_targeted").astype(int)
c["south"] = (c["zone"] == "South_islands").astype(int)
c["corridor"] = ((c["sbc"] == 1) | (c["south"] == 1)).astype(int)
print("[zones]", c["zone"].value_counts().to_dict())


# ---------------------------------------------------------------------------
# 3. Zone descriptives
# ---------------------------------------------------------------------------
print("\n=== (1) Zone descriptives (absolute vessel count, 20km) ===")
print(f"{'zone':16s}{'N':>4s}{'pre':>9s}{'post':>9s}{'dchg':>9s}{'%<0':>7s}{'medPre':>9s}")
rows = []
for z in ["SBC_targeted", "South_islands", "Other"]:
    s = c[c["zone"] == z]
    frac_neg = 100 * (s["dchg"] < 0).mean()
    print(f"{z:16s}{len(s):4d}{s['pre'].mean():9.2f}{s['post'].mean():9.2f}"
          f"{s['dchg'].mean():+9.2f}{frac_neg:7.0f}{s['pre'].median():9.2f}")
    rows.append({"zone": z, "n": len(s), "mean_pre": s["pre"].mean(),
                 "mean_post": s["post"].mean(), "mean_dchg": s["dchg"].mean(),
                 "pct_dchg_neg": frac_neg, "median_pre": s["pre"].median()})


# ---------------------------------------------------------------------------
# 4. Reshuffle within the targeted corridor: SBC vs South
# ---------------------------------------------------------------------------
print("\n=== (2) Reshuffle within targeted corridor: SBC-return vs South-of-islands ===")
sbc, south = c[c["sbc"] == 1], c[c["south"] == 1]
t, p = stats.ttest_ind(sbc["dchg"], south["dchg"], equal_var=False)
print(f"  SBC mean dchg   = {sbc['dchg'].mean():+.2f}  (N={len(sbc)}, median pre={sbc['pre'].median():.1f})")
print(f"  South mean dchg = {south['dchg'].mean():+.2f}  (N={len(south)}, median pre={south['pre'].median():.1f})")
print(f"  Welch t = {t:.2f}, p = {p:.4g} {stars(p)}")
# south cells that are genuinely busy (pre >= median south) yet still lost:
south_busy = south[south["pre"] >= south["pre"].median()]
print(f"  busy South cells (pre>=median): mean dchg = {south_busy['dchg'].mean():+.2f} "
      f"(recovery would predict +{(south_busy['pre'].mean()*(c['post'].sum()/c['pre'].sum()-1)):.2f})")


# ---------------------------------------------------------------------------
# 5. Baseline-controlled regression (the task's suggested test)
#    Recovery is proportional to LEVEL baseline (dchg = pre*g), so control for
#    pre (level) primarily; also report the ln(2011-exposure) control.
# ---------------------------------------------------------------------------
print("\n=== (3) Baseline-controlled OLS: dchg ~ baseline + I(SBC) + I(South) ===")
print("    (ref = all non-targeted cells incl. LA/LB; HC1 SEs)")
specs = [("pre", "level baseline (mean 2009-10 count)"),
         ("lnc2011", "log 2011 exposure (task-suggested)")]
for base, lab in specs:
    est = pf.feols(f"dchg ~ {base} + sbc + south", data=c, vcov="HC1")
    bs, ss, ps = est.coef()["sbc"], est.se()["sbc"], est.pvalue()["sbc"]
    bo, so, po = est.coef()["south"], est.se()["south"], est.pvalue()["south"]
    print(f"  control={lab}")
    print(f"     SBC   = {bs:+.2f} ({ss:.2f}){stars(ps):3s}   South = {bo:+.2f} ({so:.2f}){stars(po):3s}")
# flexible baseline (quadratic in pre) to absorb scale-dependent recovery
c["pre2"] = c["pre"] ** 2
estq = pf.feols("dchg ~ pre + pre2 + sbc + south", data=c, vcov="HC1")
print(f"  control=pre + pre^2 (flexible scale)")
print(f"     SBC   = {estq.coef()['sbc']:+.2f} ({estq.se()['sbc']:.2f}){stars(estq.pvalue()['sbc']):3s}"
      f"   South = {estq.coef()['south']:+.2f} ({estq.se()['south']:.2f}){stars(estq.pvalue()['south']):3s}")


# ---------------------------------------------------------------------------
# 6. Excess over a uniform-recovery benchmark
#    (a) global g;  (b) out-of-sample counterfactual fit on NON-targeted cells.
# ---------------------------------------------------------------------------
g_agg = c["post"].sum() / c["pre"].sum() - 1.0
c["exp_recov"] = c["pre"] * g_agg
c["excess_g"] = c["dchg"] - c["exp_recov"]
print(f"\n=== (4a) Excess over UNIFORM recovery (aggregate g = {g_agg:.3f}) ===")
print("    excess_i = dchg_i - pre_i * g ; >0 => gained more than pure recovery")
for z in ["SBC_targeted", "South_islands", "Other"]:
    s = c[c["zone"] == z]["excess_g"]
    ts, pz = stats.ttest_1samp(s, 0.0)
    print(f"  {z:16s} mean excess = {s.mean():+.2f}  (t={ts:+.2f}, p={pz:.3g} {stars(pz)}, N={len(s)})")

# (b) out-of-sample: fit dchg ~ pre (+pre^2) on NON-targeted cells within the
# corridor's baseline support (drop LA/LB giants so we compare like with like).
corr_max_pre = c.loc[c["corridor"] == 1, "pre"].max()
fit = c[(c["corridor"] == 0) & (c["pre"] <= corr_max_pre + 1e-9)].copy()
co = np.polyfit(fit["pre"], fit["dchg"], 2)
c["cf"] = np.polyval(co, c["pre"])
c["excess_cf"] = c["dchg"] - c["cf"]
print(f"\n=== (4b) Excess over NON-TARGETED counterfactual (fit on {len(fit)} non-targeted cells, pre<= {corr_max_pre:.1f}) ===")
print("    counterfactual = how absolute traffic changed vs baseline in NON-targeted areas")
for z in ["SBC_targeted", "South_islands", "Other"]:
    s = c[c["zone"] == z]["excess_cf"]
    ts, pz = stats.ttest_1samp(s, 0.0)
    print(f"  {z:16s} mean excess = {s.mean():+.2f}  (t={ts:+.2f}, p={pz:.3g} {stars(pz)}, N={len(s)})")


# ---------------------------------------------------------------------------
# 7. Concentration among BUSY cells (directly addresses the confound)
# ---------------------------------------------------------------------------
BUSY = 20.0
print(f"\n=== (5) Among BUSY cells (pre >= {BUSY:.0f}): targeted corridor vs non-targeted ===")
busy = c[c["pre"] >= BUSY]
for z in ["SBC_targeted", "South_islands", "Other"]:
    s = busy[busy["zone"] == z]
    if len(s):
        print(f"  {z:16s} N={len(s):2d}  mean dchg={s['dchg'].mean():+.2f}  "
              f"mean excess(g)={s['excess_g'].mean():+.2f}")
cb, ob = busy[busy["corridor"] == 1], busy[busy["corridor"] == 0]
if len(cb) and len(ob):
    t2, p2 = stats.ttest_ind(cb["excess_g"], ob["excess_g"], equal_var=False)
    print(f"  busy corridor vs busy non-targeted, excess(g): "
          f"{cb['excess_g'].mean():+.2f} vs {ob['excess_g'].mean():+.2f} "
          f"(Welch t={t2:.2f}, p={p2:.3g} {stars(p2)})")


# ---------------------------------------------------------------------------
# 8. Robustness: unambiguous-only + lat-split sensitivity
# ---------------------------------------------------------------------------
print("\n=== (6) Robustness ===")
# (a) unambiguous cells only: SBC = mainland lane clat>=34.30; South = clat<=33.95
unamb = c.copy()
unamb["z2"] = "drop"
m_sbc = (unamb["clon"].between(-120.80, -119.15)) & (unamb["clat"] >= 34.30)
m_sth = (unamb["clon"].between(-120.55, -118.90)) & (unamb["clat"] <= 33.95) & (unamb["clat"] >= 33.15)
unamb.loc[m_sbc, "z2"] = "SBC"; unamb.loc[m_sth, "z2"] = "South"
us, ut = unamb[unamb["z2"] == "SBC"], unamb[unamb["z2"] == "South"]
tt, pp = stats.ttest_ind(us["dchg"], ut["dchg"], equal_var=False)
print(f"  (a) unambiguous only: SBC dchg={us['dchg'].mean():+.2f} (N={len(us)}) vs "
      f"South dchg={ut['dchg'].mean():+.2f} (N={len(ut)})  Welch p={pp:.3g} {stars(pp)}")
# (b) lat-split sensitivity for SBC vs South within corridor
for ls in [33.95, 34.00, 34.05]:
    zz = [classify(lo, la, lat_split=ls) for lo, la in zip(c["clon"], c["clat"])]
    zz = pd.Series(zz, index=c.index)
    a = c.loc[zz == "SBC_targeted", "dchg"]; b = c.loc[zz == "South_islands", "dchg"]
    print(f"  (b) lat_split={ls}: SBC={a.mean():+.2f} (N={len(a)})  South={b.mean():+.2f} (N={len(b)})  diff={a.mean()-b.mean():+.2f}")
# (c) use CSV coords (kelp-masked centroid, ~2-3km from polygon centroid) instead
zc = pd.Series([classify(lo, la) for lo, la in zip(c["csv_lon"], c["csv_lat"])], index=c.index)
a = c.loc[zc == "SBC_targeted", "dchg"]; b = c.loc[zc == "South_islands", "dchg"]
tt2, pp2 = stats.ttest_ind(a, b, equal_var=False)
print(f"  (c) CSV coords: SBC={a.mean():+.2f} (N={len(a)})  South={b.mean():+.2f} (N={len(b)})  Welch p={pp2:.3g} {stars(pp2)}")


# ---------------------------------------------------------------------------
# 9. Save per-cell + zone-stats CSVs
# ---------------------------------------------------------------------------
out = c[["eca_index_official", "year_eca", "clon", "clat", "zone", "sbc", "south", "corridor",
         "pre", "post", "dchg", "lnc2011", "exp_recov", "excess_g", "cf", "excess_cf"]].copy()
out.to_csv(os.path.join(CSV_DIR, "spatial_targeting_2011.csv"), index=False)
pd.DataFrame(rows).to_csv(os.path.join(CSV_DIR, "spatial_targeting_2011_zonestats.csv"), index=False)


# ---------------------------------------------------------------------------
# 10. Map figure (png only)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 7))
vmax = np.nanpercentile(np.abs(c["dchg"]), 98)
sizes = 25 + 3.2 * c["pre"]
sc = ax.scatter(c["clon"], c["clat"], c=c["dchg"], s=sizes, cmap="RdBu_r",
                vmin=-vmax, vmax=vmax, edgecolor="k", linewidth=0.4, zorder=3)
cb = plt.colorbar(sc, ax=ax); cb.set_label("Post-2011 absolute traffic change  (mean 2012-16 - mean 2009-10, vessels)")
# zone rectangles
ax.add_patch(Rectangle((-120.80, 34.00), 1.65, 0.70, fill=False, ec="green", lw=1.8, ls="--", zorder=2,
                       label="SBC-return (targeted)"))
ax.add_patch(Rectangle((-120.55, 33.15), 1.65, 0.85, fill=False, ec="purple", lw=1.8, ls=":", zorder=2,
                       label="South-of-islands (targeted)"))
# northern Channel Islands markers
isl = {"San Miguel": (-120.37, 34.05), "Santa Rosa": (-120.09, 33.98),
       "Santa Cruz": (-119.73, 34.01), "Anacapa": (-119.37, 34.01)}
for nm, (lo, la) in isl.items():
    ax.plot(lo, la, marker="^", color="saddlebrown", ms=8, zorder=4)
ax.text(-119.9, 33.86, "northern Channel Islands", color="saddlebrown", fontsize=8, ha="center")
ax.annotate("LA / Long Beach\napproaches", (-118.35, 33.72), (-118.9, 33.35), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Spatial targeting of the Dec-2011 ECA boundary modification\n"
             "SBC lanes gain (red), south-of-islands diversion route loses (blue); marker size ~ baseline traffic")
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax.set_xlim(-121.0, -117.0); ax.set_ylim(32.4, 35.0)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_traffic_reshuffle_2011.png"), dpi=150)
plt.close(fig)

print("\nSaved:")
print(" ", os.path.join(CSV_DIR, "spatial_targeting_2011.csv"))
print(" ", os.path.join(CSV_DIR, "spatial_targeting_2011_zonestats.csv"))
print(" ", os.path.join(FIG_DIR, "fig_traffic_reshuffle_2011.png"))
print("DONE. No paper/ files modified.")
