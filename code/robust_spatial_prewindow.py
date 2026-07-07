"""
Two robustness checks on the main 2009 PPML result (referee requests).

Baseline main spec (from 2_a_estim.py): unweighted PPML on cells with
>= 2 vessels/month; 81 cells x 48 quarters = 3,880 obs (2005-2016).
4 specs = {biomass_sum, area_sum} x {distance, count}. FE: cell_id + yq.
Controls: ln_sst + ln_precipitation + ln_swh. SEs clustered at cell level.

CHECK 1 -- SPATIAL / TWO-WAY CLUSTERING (referee point: 20km overlapping
buffers + cell-level clustering may understate spatial correlation).
Point estimates are unchanged; we re-compute the standard errors under
coarser SPATIAL clustering and two-way clustering:
  - cell_id (baseline)
  - cell_id + yq (two-way: space x time)
  - latitude bands (0.25 deg, 0.5 deg, 1.0 deg -- contiguous coastal blocks)
  - KMeans spatial clusters on projected cell centroids
Conley spatial-HAC is deliberately NOT hand-rolled: our estimator is
fixed-effects PPML (fepois), and Conley-HAC is an OLS-oriented tool whose
Poisson-FE analog we address via coarser spatial clustering. (R's conleyreg,
which does support FE-Poisson Conley, is not installed in this environment.)

CHECK 2 -- 2009 EFFECT ON THE CLEAN PRE-2011 WINDOW (referee point: the
post-2009 period spans 2009-2016, most of it AFTER the Dec-2011 boundary
modification, so the dynamic 2009 effect could be contaminated by 2011).
Re-estimate the 4 main specs on year <= 2011 (post = 2009Q3-2011Q4, entirely
BEFORE the Dec-2011 boundary change). Same FE, controls, clustering.

Outputs:
  output/csv/robust_spatial_se.csv
  output/csv/robust_prewindow.csv
  output/tables/tab_spatialse_2009.tex
  output/tables/tab_prewindow_2009.tex
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import pyfixest as pf
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED = os.path.join(BASE, "data", "generated")
RAW = os.path.join(BASE, "data", "raw")
TABLE_DIR = os.path.join(BASE, "output", "tables")
CSV_DIR = os.path.join(BASE, "output", "csv")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


def stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Data preparation -- IDENTICAL to code/2_a_estim.py through the count>=2 filter
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

VESSEL_THRESHOLD = np.log(2)
cell_count = df.groupby("cell_id")["ln_mean_count_2009_jan_jun_20km"].first()
cells_above = cell_count[cell_count >= VESSEL_THRESHOLD].index
df = df[df["cell_id"].isin(cells_above)].reset_index(drop=True).copy()

print("=" * 78)
print(f"BASELINE SAMPLE: {len(df)} obs, {df['cell_id'].nunique()} cells "
      f"(expected 3880 obs / 81 cells)")
print("=" * 78)

# ---------------------------------------------------------------------------
# Spatial block assignments from gpkg cell centroids
# ---------------------------------------------------------------------------
g = gpd.read_file(os.path.join(RAW, "ECA_10km.gpkg"))
gp = g.to_crs(3310)                      # California Albers (meters) for distances
cent_m = gp.geometry.centroid
cent_ll = g.to_crs(4326).geometry.centroid
cent = pd.DataFrame({
    "cell_id": g["id"].astype(str) + "_" + g["year"].astype(str),
    "x_m": cent_m.x.values, "y_m": cent_m.y.values,
    "clon": cent_ll.x.values, "clat": cent_ll.y.values,
})
cent = cent.drop_duplicates("cell_id")

cells = df.drop_duplicates("cell_id")[["cell_id"]].merge(cent, on="cell_id", how="left")
assert cells["clat"].notna().all(), "Some cells failed to match a centroid"

# Coastal latitude bands (the CA coast here runs ~N-S, so lat bands are
# contiguous spatial blocks of adjacent cells).
cells["lat_band_025"] = np.floor(cells["clat"] / 0.25).astype(int).astype(str)
cells["lat_band_05"] = np.floor(cells["clat"] / 0.50).astype(int).astype(str)
cells["lat_band_10"] = np.floor(cells["clat"] / 1.00).astype(int).astype(str)

# KMeans spatial clusters on projected centroids (geometric adjacency).
K_KMEANS = 12
km = KMeans(n_clusters=K_KMEANS, random_state=0, n_init=10)
cells["sp_kmeans"] = ("k" + pd.Series(
    km.fit_predict(cells[["x_m", "y_m"]].values)).astype(str)).values

block_cols = ["lat_band_025", "lat_band_05", "lat_band_10", "sp_kmeans"]
df = df.merge(cells[["cell_id"] + block_cols], on="cell_id", how="left")

print("\nSpatial block schemes (number of clusters across 81 cells):")
for c in block_cols:
    print(f"  {c:<14}: {df[c].nunique()} clusters")

# ---------------------------------------------------------------------------
# Specifications
# ---------------------------------------------------------------------------
CONTROLS = "ln_sst + ln_precipitation + ln_swh"
FE = "cell_id + yq"
SPECS = [
    ("biomass_sum", "ln_mean_length_2009_jan_jun_20km", "Biomass", "Distance"),
    ("area_sum",    "ln_mean_length_2009_jan_jun_20km", "Area",    "Distance"),
    ("biomass_sum", "ln_mean_count_2009_jan_jun_20km",  "Biomass", "Count"),
    ("area_sum",    "ln_mean_count_2009_jan_jun_20km",  "Area",    "Count"),
]


def int_key(est, treat):
    for k in est.coef().index:
        if treat in k and "policy_eca_2009" in k:
            return k
    raise KeyError(treat)


# ---------------------------------------------------------------------------
# CHECK 1 : spatial / two-way clustering
# ---------------------------------------------------------------------------
CLUSTER_SCHEMES = [
    ("cell_id",            "Cell (baseline)"),
    ("cell_id + yq",       "Cell + Year-Qtr (two-way)"),
    ("lat_band_025",       "Lat band 0.25deg"),
    ("lat_band_05",        "Lat band 0.50deg"),
    ("lat_band_10",        "Lat band 1.00deg"),
    ("sp_kmeans",          f"KMeans spatial (k={K_KMEANS})"),
]

print("\n" + "=" * 78)
print("CHECK 1: SPATIAL / TWO-WAY CLUSTERING")
print("=" * 78)

rows1 = []
for depvar, treat, dlab, tlab in SPECS:
    fml = f"{depvar} ~ {treat}:policy_eca_2009 + {CONTROLS} | {FE}"
    # point estimate identical across schemes -> fit once with baseline, then re-vcov
    base = pf.fepois(fml, data=df, vcov={"CRV1": "cell_id"})
    key = int_key(base, treat)
    coef = base.coef()[key]
    rec = {"outcome": dlab, "treatment": tlab, "coef": coef}
    print(f"\n[{tlab} / {dlab}]  coef = {coef:+.4f}")
    for cvar, clab in CLUSTER_SCHEMES:
        est = pf.fepois(fml, data=df, vcov={"CRV1": cvar})
        se = est.se()[key]
        p = est.pvalue()[key]
        ng = est._G[0] if hasattr(est, "_G") and est._G is not None else np.nan
        rec[f"se__{cvar}"] = se
        rec[f"p__{cvar}"] = p
        print(f"    {clab:<28} SE={se:.4f}  p={p:.4f} {stars(p):<3}  "
              f"(G={df[cvar.split(' + ')[0]].nunique() if '+' not in cvar else 'twoway'})")
    rows1.append(rec)

res1 = pd.DataFrame(rows1)
res1.to_csv(os.path.join(CSV_DIR, "robust_spatial_se.csv"), index=False)
print(f"\nSaved {os.path.join(CSV_DIR, 'robust_spatial_se.csv')}")

# ---------------------------------------------------------------------------
# CHECK 2 : 2009 effect on the clean pre-2011 window (year <= 2011)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("CHECK 2: 2009 EFFECT ON THE PRE-2011 WINDOW (year <= 2011)")
print("=" * 78)

dfp = df[df["year"] <= 2011].copy()
n_cells = dfp["cell_id"].nunique()
n_q = dfp["yq"].nunique()
print(f"Restricted sample: {len(dfp)} obs, {n_cells} cells x {n_q} quarters")
print(f"  post period (2009Q3-2011Q4): {(dfp['policy_eca_2009']==1).sum()} obs; "
      f"pre: {(dfp['policy_eca_2009']==0).sum()} obs")

rows2 = []
for depvar, treat, dlab, tlab in SPECS:
    fml = f"{depvar} ~ {treat}:policy_eca_2009 + {CONTROLS} | {FE}"
    # full-sample baseline
    fb = pf.fepois(fml, data=df, vcov={"CRV1": "cell_id"})
    kb = int_key(fb, treat)
    # pre-2011 window
    ep = pf.fepois(fml, data=dfp, vcov={"CRV1": "cell_id"})
    kp = int_key(ep, treat)
    rec = {
        "outcome": dlab, "treatment": tlab,
        "coef_full": fb.coef()[kb], "se_full": fb.se()[kb], "p_full": fb.pvalue()[kb],
        "coef_pre2011": ep.coef()[kp], "se_pre2011": ep.se()[kp],
        "p_pre2011": ep.pvalue()[kp], "n_pre2011": ep._N,
    }
    rows2.append(rec)
    print(f"\n[{tlab} / {dlab}]")
    print(f"    full 2005-2016 : {rec['coef_full']:+.4f} ({rec['se_full']:.4f}) "
          f"p={rec['p_full']:.4f} {stars(rec['p_full'])}")
    print(f"    pre-2011       : {rec['coef_pre2011']:+.4f} ({rec['se_pre2011']:.4f}) "
          f"p={rec['p_pre2011']:.4f} {stars(rec['p_pre2011'])}   N={rec['n_pre2011']}")

res2 = pd.DataFrame(rows2)
res2.to_csv(os.path.join(CSV_DIR, "robust_prewindow.csv"), index=False)
print(f"\nSaved {os.path.join(CSV_DIR, 'robust_prewindow.csv')}")

# ---------------------------------------------------------------------------
# LaTeX table -- CHECK 1 (spatial / two-way SEs)
# ---------------------------------------------------------------------------
col_order = [(d, t) for d, _, t, _ in
             [("Biomass", None, "Distance", None), ("Area", None, "Distance", None),
              ("Biomass", None, "Count", None), ("Area", None, "Count", None)]]


def find_row(dlab, tlab):
    m = res1[(res1["outcome"] == dlab) & (res1["treatment"] == tlab)]
    return m.iloc[0]


L = []
L.append(r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
L.append(r"\begin{tabular}{l*{4}{c}}")
L.append(r"\hline\hline")
L.append(r" &\multicolumn{2}{c}{Distance Traveled (km)}&\multicolumn{2}{c}{Vessel Count}\\")
L.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
L.append(r" &\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}&\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}\\")
L.append(r"\hline")

# point estimate row
line = r"Post 2009 $\times$ Exposure"
for dlab, tlab in col_order:
    r = find_row(dlab, tlab)
    line += f" & {r['coef']:.4f}"
L.append(line + r"\\")
L.append(r"[0.3em]")

gcell = df["cell_id"].nunique()
gtime = df["yq"].nunique()
se_label = {
    "cell_id": f"Cell cluster, baseline ($G={gcell}$)",
    "cell_id + yq": f"Two-way: cell $\\times$ year-qtr ($G={gcell},{gtime}$)",
    "lat_band_025": f"Lat band $0.25^\\circ$ ($G={df['lat_band_025'].nunique()}$)",
    "lat_band_05": f"Lat band $0.50^\\circ$ ($G={df['lat_band_05'].nunique()}$)",
    "lat_band_10": f"Lat band $1.00^\\circ$ ($G={df['lat_band_10'].nunique()}$)",
    "sp_kmeans": f"KMeans spatial ($k={K_KMEANS}$)",
}
for cvar, _ in CLUSTER_SCHEMES:
    lab = se_label[cvar]
    line = f"\\quad {lab}"
    for dlab, tlab in col_order:
        r = find_row(dlab, tlab)
        se = r[f"se__{cvar}"]
        p = r[f"p__{cvar}"]
        line += f" & ({se:.4f})\\sym{{{stars(p)}}}"
    L.append(line + r"\\")

L.append(r"\hline")
L.append(f"Observations & {len(df)} & {len(df)} & {len(df)} & {len(df)}" + r"\\")
L.append(r"\hline\hline")
L.append(r"\end{tabular}")
with open(os.path.join(TABLE_DIR, "tab_spatialse_2009.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print(f"Saved {os.path.join(TABLE_DIR, 'tab_spatialse_2009.tex')}")

# ---------------------------------------------------------------------------
# LaTeX table -- CHECK 2 (pre-2011 window)
# ---------------------------------------------------------------------------
def find2(dlab, tlab):
    m = res2[(res2["outcome"] == dlab) & (res2["treatment"] == tlab)]
    return m.iloc[0]


L2 = []
L2.append(r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
L2.append(r"\begin{tabular}{l*{4}{c}}")
L2.append(r"\hline\hline")
L2.append(r" &\multicolumn{2}{c}{Distance Traveled (km)}&\multicolumn{2}{c}{Vessel Count}\\")
L2.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
L2.append(r" &\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}&\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}\\")
L2.append(r"\hline")
# full sample
line = r"Post 2009 $\times$ Exposure (2005--2016)"
for dlab, tlab in col_order:
    r = find2(dlab, tlab)
    line += f" & {r['coef_full']:.4f}\\sym{{{stars(r['p_full'])}}}"
L2.append(line + r"\\")
line = " "
for dlab, tlab in col_order:
    r = find2(dlab, tlab)
    line += f" & ({r['se_full']:.4f})"
L2.append(line + r"\\")
L2.append(r"[0.4em]")
# pre-2011
line = r"Post 2009 $\times$ Exposure (2005--2011)"
for dlab, tlab in col_order:
    r = find2(dlab, tlab)
    line += f" & {r['coef_pre2011']:.4f}\\sym{{{stars(r['p_pre2011'])}}}"
L2.append(line + r"\\")
line = " "
for dlab, tlab in col_order:
    r = find2(dlab, tlab)
    line += f" & ({r['se_pre2011']:.4f})"
L2.append(line + r"\\")
L2.append(r"\hline")
line = "Observations (2005--2011)"
for dlab, tlab in col_order:
    r = find2(dlab, tlab)
    line += f" & {int(r['n_pre2011'])}"
L2.append(line + r"\\")
L2.append(r"\hline\hline")
L2.append(r"\end{tabular}")
with open(os.path.join(TABLE_DIR, "tab_prewindow_2009.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(L2))
print(f"Saved {os.path.join(TABLE_DIR, 'tab_prewindow_2009.tex')}")

print("\nDone.")
