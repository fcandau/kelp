"""
Build the estimation panel for Conley (spatial-HAC) standard errors on the
main 2009 PPML specifications, and reproduce the baseline cell-clustered
results as a sanity check.

- Reproduces EXACTLY the sample of code/2_a_estim.py:
  filter year<=2016 & lat in [32,36]; cell_id = eca_index_official + "_" + year_eca;
  yq = year + "_" + quarter; drop_duplicates(cell_id, yq); dropna on key cols;
  keep cells with ln_mean_count_2009_jan_jun_20km >= ln(2). => 81 cells, 3880 obs.

- Attaches each cell's centroid (lon, lat) from data/raw/ECA_10km.gpkg
  (gpkg id = eca_index_official, gpkg year = year_eca).

- Writes a tidy CSV for R conleyreg:
  data/generated/conley_panel_2009.csv

Standalone diagnostic (NOT part of the numbered pipeline).
Python: C:\\Users\\fcanda01\\AppData\\Local\\anaconda3\\envs\\r-env\\python.exe
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import pyfixest as pf
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED = os.path.join(BASE, "data", "generated")
RAW = os.path.join(BASE, "data", "raw")
OUT_CSV = os.path.join(GENERATED, "conley_panel_2009.csv")

# ----------------------------------------------------------------------
# 1. Reproduce the exact main 2009 sample (mirrors 2_a_estim.py)
# ----------------------------------------------------------------------
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
treat_count_var = "ln_mean_count_2009_jan_jun_20km"
cell_count = df.groupby("cell_id")[treat_count_var].first()
cells_above = cell_count[cell_count >= VESSEL_THRESHOLD].index
df = df[df["cell_id"].isin(cells_above)].copy()

print(f"Sample: {len(df)} obs, {df['cell_id'].nunique()} cells "
      f"(expect 3880 obs, 81 cells)")

# ----------------------------------------------------------------------
# 2. Reproduce baseline cell-clustered PPML (4 specs) as sanity check
# ----------------------------------------------------------------------
controls = "ln_sst + ln_precipitation + ln_swh"
fe = "cell_id + yq"
VCOV = {"CRV1": "cell_id"}
specs = [
    ("biomass_sum", "ln_mean_length_2009_jan_jun_20km", "biomass", "distance"),
    ("area_sum",    "ln_mean_length_2009_jan_jun_20km", "area",    "distance"),
    ("biomass_sum", "ln_mean_count_2009_jan_jun_20km",  "biomass", "count"),
    ("area_sum",    "ln_mean_count_2009_jan_jun_20km",  "area",    "count"),
]

print("\nBaseline cell-clustered PPML (must match paper: "
      "+0.136/+0.132/+0.155/+0.149):")
baseline_rows = []
for depvar, treat, dep_label, treat_label in specs:
    fml = f"{depvar} ~ {treat}:policy_eca_2009 + {controls} | {fe}"
    est = pf.fepois(fml, data=df, vcov=VCOV)
    key = [k for k in est.coef().index
           if treat in k and "policy_eca_2009" in k][0]
    c, s, p = est.coef()[key], est.se()[key], est.pvalue()[key]
    print(f"  {dep_label:8s} x {treat_label:8s}: "
          f"coef={c:+.4f}  clustSE={s:.4f}  p={p:.4g}  N={est._N}")
    baseline_rows.append({
        "outcome": dep_label, "exposure": treat_label,
        "coef": c, "cluster_se": s, "cluster_p": p, "N": est._N,
    })

pd.DataFrame(baseline_rows).to_csv(
    os.path.join(GENERATED, "conley_baseline_check.csv"), index=False)

# ----------------------------------------------------------------------
# 3. Attach cell centroids (lon, lat) from the gpkg grid
#    gpkg id = eca_index_official ; gpkg year = year_eca
# ----------------------------------------------------------------------
grid = gpd.read_file(os.path.join(RAW, "ECA_10km.gpkg"))
# Accurate centroid: project to CONUS Albers, take centroid, back to WGS84
cent = grid.to_crs(5070).geometry.centroid.to_crs(4326)
grid_c = pd.DataFrame({
    "eca_index_official": grid["id"].astype(float),
    "year_eca": grid["year"].astype(int),
    "lon": cent.x.values,
    "lat_cent": cent.y.values,
})

df["eca_index_official"] = df["eca_index_official"].astype(float)
df["year_eca"] = df["year_eca"].astype(int)
df = df.merge(grid_c, on=["eca_index_official", "year_eca"], how="left")

n_missing = df["lon"].isna().sum()
print(f"\nCentroid merge: {n_missing} obs with missing centroid "
      f"(of {len(df)}).")
# Cross-check gpkg centroid vs CSV's own latitude/longitude columns
d_lat = (df["lat_cent"] - df["latitude"]).abs()
d_lon = (df["lon"] - df["longitude"]).abs()
print(f"  max |gpkg_lat - csv_lat| = {d_lat.max():.4f} deg ; "
      f"max |gpkg_lon - csv_lon| = {d_lon.max():.4f} deg")

# ----------------------------------------------------------------------
# 4. Write tidy CSV for R conleyreg
# ----------------------------------------------------------------------
keep = ["cell_id", "yq", "year", "quarter",
        "biomass_sum", "area_sum",
        "ln_mean_length_2009_jan_jun_20km", "ln_mean_count_2009_jan_jun_20km",
        "ln_sst", "ln_precipitation", "ln_swh",
        "policy_eca_2009", "lon", "lat_cent"]
tidy = df[keep].rename(columns={"lat_cent": "lat"}).copy()
tidy.to_csv(OUT_CSV, index=False)
print(f"\nWrote tidy panel: {OUT_CSV}")
print(f"  shape={tidy.shape}, cells={tidy['cell_id'].nunique()}, "
      f"periods={tidy['yq'].nunique()}")
print(tidy.head(3).to_string())
