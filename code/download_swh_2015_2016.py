"""
Download ERA5 significant wave height (SWH) for 2015-2016 and append
to the existing swh_mean_official.csv.

This script:
1. Downloads ERA5 monthly-averaged SWH for 2015-2016 (California coast)
2. Computes quarterly zonal means per hexagonal cell (matching existing data)
3. Appends the new rows to data/raw/swh_mean_official.csv

Prerequisites:
- CDS API key in ~/.cdsapirc  (already configured)
- Packages: cdsapi, xarray, netCDF4, geopandas, rasterstats, rasterio, shapely

Run once, then delete this script.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from shapely import wkt
import cdsapi
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
ROOT = r"c:\Users\fcanda01\Desktop\Recherche\kelp"
RAW = os.path.join(ROOT, "data", "raw")
SWH_CSV = os.path.join(RAW, "swh_mean_official.csv")
NC_FILE = os.path.join(RAW, "era5_swh_2015_2016_monthly.nc")

# ── 1. Download ERA5 monthly-averaged SWH ─────────────────────────
# Bounding box from the hexagonal grid (with 1° buffer)
AREA = [43, -126, 26, -113]  # [N, W, S, E]

if not os.path.exists(NC_FILE):
    print("Downloading ERA5 monthly-averaged SWH for 2015-2016...")
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": "significant_height_of_combined_wind_waves_and_swell",
            "year": ["2015", "2016"],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": "00:00",
            "area": AREA,
            "format": "netcdf",
        },
        NC_FILE,
    )
    print(f"Downloaded: {NC_FILE}")
else:
    print(f"NetCDF already exists: {NC_FILE}")

# ── 2. Read existing CSV to get cell geometries ───────────────────
print("Reading existing SWH data...")
existing = pd.read_csv(SWH_CSV)

# Extract unique cells with their geometries
cells = existing[["ECA_index_official", "geometry"]].drop_duplicates("ECA_index_official")
cells = cells.reset_index(drop=True)

# Build GeoDataFrame from WKT geometries
# Some cells have NaN geometry — keep them (they'll get NaN SWH, matching existing pattern)
cells_with_geom = cells[cells["geometry"].notna()].copy()
cells_with_geom["geom"] = cells_with_geom["geometry"].apply(wkt.loads)
gdf = gpd.GeoDataFrame(cells_with_geom, geometry="geom", crs="EPSG:4326")

cells_nan = cells[cells["geometry"].isna()]["ECA_index_official"].tolist()
print(f"  {len(gdf)} cells with geometry, {len(cells_nan)} cells with NaN geometry")

# ── 3. Open NetCDF and compute zonal stats ─────────────────────────
print("Computing zonal statistics...")
ds = xr.open_dataset(NC_FILE)

# ERA5 variable name (may vary: 'swh' or 'shww' or long name)
# Try common names
for vname in ["swh", "shww", "significant_height_of_combined_wind_waves_and_swell"]:
    if vname in ds.data_vars:
        swh_var = vname
        break
else:
    # Pick the first non-coordinate variable
    swh_var = list(ds.data_vars)[0]
print(f"  SWH variable name in NetCDF: '{swh_var}'")

# Get coordinate/dimension names (ERA5 new CDS uses 'valid_time' instead of 'time')
time_name = "valid_time" if "valid_time" in ds.coords else "time"
lon_name = "longitude" if "longitude" in ds.coords else "lon"
lat_name = "latitude" if "latitude" in ds.coords else "lat"

print(f"  Time steps: {len(ds[time_name])}")
print(f"  Spatial shape: {ds[swh_var].shape}")

lons = ds[lon_name].values
lats = ds[lat_name].values

# Compute cell centroids once (for interpolation)
centroids_lon = gdf.geometry.centroid.x.values
centroids_lat = gdf.geometry.centroid.y.values
centroid_points = np.column_stack([centroids_lat, centroids_lon])  # (lat, lon) order for grid

# ERA5 latitudes must be ascending for RegularGridInterpolator
lats_asc = np.sort(lats)
flip_lat = lats[0] > lats[-1]

results = []

for t_idx, t in enumerate(ds[time_name].values):
    t_pd = pd.Timestamp(t)
    year = t_pd.year
    quarter = (t_pd.month - 1) // 3 + 1
    month = t_pd.month

    # Extract 2D array for this time step
    data_2d = ds[swh_var].isel({time_name: t_idx}).values
    data_2d = np.squeeze(data_2d)

    # Flip to ascending latitude order
    if flip_lat:
        data_2d = data_2d[::-1, :]

    # Interpolate to cell centroids using nearest-neighbor
    # (matches how co-author likely extracted: each cell gets its nearest ERA5 ocean pixel)
    # Use nearest to avoid interpolating NaN land pixels into ocean values
    interp = RegularGridInterpolator(
        (lats_asc, lons), data_2d,
        method="nearest", bounds_error=False, fill_value=np.nan,
    )
    swh_values = interp(centroid_points)

    for idx_pos, (i, row) in enumerate(gdf.iterrows()):
        cell_id = row["ECA_index_official"]
        val = swh_values[idx_pos]
        swh_mean = float(val) if np.isfinite(val) else None
        results.append({
            "ECA_index_official": cell_id,
            "swh_mean": swh_mean,
            "year": year,
            "month": month,
            "quarter": quarter,
        })

    print(f"  Processed {t_pd.strftime('%Y-%m')} (Q{quarter})")

ds.close()

# ── 4. Aggregate monthly → quarterly means ─────────────────────────
print("Aggregating to quarterly means...")
df_new = pd.DataFrame(results)
df_quarterly = (
    df_new.groupby(["ECA_index_official", "year", "quarter"])["swh_mean"]
    .mean()
    .reset_index()
)

# Add NaN cells (matching existing pattern)
nan_rows = []
for year in [2015, 2016]:
    for quarter in [1, 2, 3, 4]:
        for cell_id in cells_nan:
            nan_rows.append({
                "ECA_index_official": cell_id,
                "swh_mean": np.nan,
                "year": year,
                "quarter": quarter,
            })
if nan_rows:
    df_quarterly = pd.concat([df_quarterly, pd.DataFrame(nan_rows)], ignore_index=True)

# Add geometry column (matching existing CSV format)
geom_lookup = dict(zip(cells["ECA_index_official"], cells["geometry"]))
df_quarterly["geometry"] = df_quarterly["ECA_index_official"].map(geom_lookup)

# ── 5. Append to existing CSV ──────────────────────────────────────
print("Appending to existing CSV...")

# Match column order of existing file: Unnamed: 0, ECA_index_official, geometry, swh_mean, year, quarter
# Reset the row index to continue from existing
max_idx = existing["Unnamed: 0"].max()
df_quarterly = df_quarterly.sort_values(["ECA_index_official", "year", "quarter"]).reset_index(drop=True)
df_quarterly["Unnamed: 0"] = range(max_idx + 1, max_idx + 1 + len(df_quarterly))
df_quarterly = df_quarterly[["Unnamed: 0", "ECA_index_official", "geometry", "swh_mean", "year", "quarter"]]

# Sanity checks
n_cells_expected = existing["ECA_index_official"].nunique()
n_cells_new = df_quarterly["ECA_index_official"].nunique()
assert n_cells_new == n_cells_expected, f"Cell count mismatch: {n_cells_new} vs {n_cells_expected}"

n_quarters_new = df_quarterly.groupby(["year", "quarter"]).ngroups
assert n_quarters_new == 8, f"Expected 8 quarters (2015Q1-2016Q4), got {n_quarters_new}"

n_nan_new = df_quarterly["swh_mean"].isna().sum()
n_nan_expected = len(cells_nan) * 8  # 58 cells × 8 quarters
print(f"  New rows: {len(df_quarterly)}")
print(f"  Cells: {n_cells_new} (expected {n_cells_expected})")
print(f"  NaN values: {n_nan_new} (expected {n_nan_expected})")

# Append
combined = pd.concat([existing, df_quarterly], ignore_index=True)
combined.to_csv(SWH_CSV, index=False)

print(f"\nDone! Updated {SWH_CSV}")
print(f"  Old: {len(existing)} rows ({existing['year'].min()}-{existing['year'].max()})")
print(f"  New: {len(combined)} rows ({combined['year'].min()}-{combined['year'].max()})")

# Quick validation
print("\nValidation — rows per year:")
print(combined.groupby("year").size().to_string())

# Clean up NetCDF
# os.unlink(NC_FILE)  # Uncomment to delete after successful run
print(f"\nNetCDF kept at {NC_FILE} — delete manually when satisfied.")
