"""
Build the regression-ready dataset.

Steps 1-5: Build new_model_kelp_exposure_2.csv from raw data
  (kelp NetCDF, AIS trajectories, Halpern raster, climate covariates)
Step 6: Extract shipping-specific treatment from impact_shipping.tif
  -> Produces new_model_kelp_exposure_3.csv (final dataset for estimation)

Each step caches its output; re-running skips completed steps.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from rasterstats import zonal_stats
import rasterio
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, 'data', 'raw')
GENERATED = os.path.join(BASE, 'data', 'generated')
os.makedirs(GENERATED, exist_ok=True)

# ============================================================
# STEP 1: KELP ANALYSIS
# ============================================================
def kelp_analysis_official(start_year=2005, end_year=2019):
    print("=" * 80)
    print("STEP 1: KELP ANALYSIS")
    print("=" * 80)

    eca_path = os.path.join(RAW, 'ECA_10km.gpkg')
    eca_hexagons = gpd.read_file(eca_path)
    eca_hexagons = eca_hexagons.to_crs(epsg=4326)
    eca_hexagons = eca_hexagons.rename(columns={'year': 'year_eca', 'id': 'eca_index_official'})

    sanctuary = gpd.read_file(os.path.join(
        RAW, 'Marine_Protected_Areas_and_National_Marine_Sanctuaries',
        'Marine_Protected_Areas_and_National_Marine_Sanctuaries.shp'))
    sanctuary = sanctuary[sanctuary['Display'] == 'Either'][['NAME', 'SANCTUARY', 'geometry']]
    sanctuary['sanctuary_dummy'] = 1
    sanctuary = sanctuary.dropna(subset=['SANCTUARY']).to_crs(epsg=4326)

    kelp_file = os.path.join(RAW, 'knb-lter-sbc.74.16',
                             'LandsatKelpBiomass_2022_Q1_withmetadata.nc')
    print(f"Loading kelp dataset from: {kelp_file}")
    data = xr.open_dataset(kelp_file)

    all_stats = []

    for year in range(start_year, end_year + 1):
        print(f"  Processing year {year}...")
        try:
            data_kelp_year = data.sel(time=(data['time.year'] == year))
            if len(data_kelp_year.time) == 0:
                continue
            final_data = data_kelp_year.to_dataframe().reset_index()
            if len(final_data) == 0:
                continue
            kelp = final_data.copy()
            kelp['year'] = kelp['time'].dt.year
            kelp['quarter'] = kelp['time'].dt.quarter

            for quarter in range(1, 5):
                kelp_quarter = kelp[kelp['quarter'] == quarter].copy()
                if len(kelp_quarter) == 0:
                    continue
                kelp_gdf = gpd.GeoDataFrame(
                    kelp_quarter,
                    geometry=gpd.points_from_xy(kelp_quarter.longitude, kelp_quarter.latitude),
                    crs="EPSG:4326")

                kelp_eca = gpd.sjoin(kelp_gdf, eca_hexagons, how='inner', predicate='within')
                kelp_eca = kelp_eca.drop(columns=['index_right'], errors='ignore')

                kelp_inside = gpd.sjoin(kelp_eca, sanctuary, how='left', predicate='within')
                kelp_inside = kelp_inside.drop(columns=['index_right', 'index_left'], errors='ignore')

                if 'NAME' in kelp_inside.columns:
                    kelp_inside = kelp_inside.rename(columns={'NAME': 'sanctuary_name', 'SANCTUARY': 'sanctuary_type'})
                if 'sanctuary_dummy' not in kelp_inside.columns:
                    kelp_inside['sanctuary_dummy'] = 0
                else:
                    kelp_inside['sanctuary_dummy'] = kelp_inside['sanctuary_dummy'].fillna(0)

                stats = kelp_inside.groupby(['eca_index_official', 'year_eca']).agg({
                    'biomass': ['mean', 'sum', 'count'],
                    'area': ['mean', 'sum', 'count'],
                    'sanctuary_dummy': 'sum',
                    'latitude': 'first',
                    'longitude': 'first'
                }).reset_index()
                stats.columns = ['eca_index_official', 'year_eca',
                                 'biomass_mean', 'biomass_sum', 'biomass_count',
                                 'area_mean', 'area_sum', 'area_count',
                                 'n_in_sanctuary', 'latitude', 'longitude']
                stats['year'] = year
                stats['quarter'] = quarter
                all_stats.append(stats)

                del kelp_gdf, kelp_eca, kelp_inside, kelp_quarter
            del final_data, kelp
        except Exception as e:
            print(f"  Error processing year {year}: {e}")
            continue

    data.close()
    if not all_stats:
        raise RuntimeError("No kelp data processed!")

    kelp_stats = pd.concat(all_stats, ignore_index=True)
    out_path = os.path.join(GENERATED, 'kelp_analysis_official.csv')
    kelp_stats.to_csv(out_path, index=False)
    print(f"  Saved {len(kelp_stats)} rows to {out_path}")
    return kelp_stats


# ============================================================
# STEP 2: SHIPPING EXPOSURE
# ============================================================
def compute_shipping_exposure_flexible(shipping_exposure, kelp_stats, eca_gpkg_path,
                                       year_filter=2009, start_month=1, end_month=12,
                                       buffer_km=20, suffix=""):
    eca_hexagons = gpd.read_file(eca_gpkg_path).to_crs(epsg=4326)
    eca_hexagons = eca_hexagons.rename(columns={'year': 'year_eca'})

    shipping_1 = shipping_exposure.rename(columns={'year_ECA': 'year_eca'}).copy()
    shipping_1 = shipping_1[['id', 'year', 'month', 'LENGTH', 'COUNT', 'year_eca']].copy()
    shipping_1['id_shipping'] = shipping_1['id']

    shipping_2 = pd.merge(shipping_1, eca_hexagons, on=["year_eca", "id"], how="inner")
    shipping_gdf = gpd.GeoDataFrame(shipping_2, geometry='geometry', crs="EPSG:4326")
    shipping_gdf = shipping_gdf[
        (shipping_gdf['year'] == year_filter) &
        (shipping_gdf['month'] >= start_month) &
        (shipping_gdf['month'] <= end_month)]

    if shipping_gdf.empty:
        return None

    shipping_proj = shipping_gdf.to_crs(epsg=3857)

    agg_dict = {col: ['sum', 'mean'] for col in ['LENGTH', 'COUNT']}
    agg_exposure = shipping_1[
        (shipping_1['year'] == year_filter) &
        (shipping_1['month'] >= start_month) &
        (shipping_1['month'] <= end_month)
    ].groupby(['id_shipping', 'year_eca']).agg(agg_dict)
    agg_exposure.columns = ['_'.join(col).strip() for col in agg_exposure.columns.values]
    shipping_quarter = agg_exposure.reset_index()

    kelp_gdf = gpd.GeoDataFrame(
        kelp_stats,
        geometry=gpd.points_from_xy(kelp_stats.longitude, kelp_stats.latitude),
        crs="EPSG:4326")
    kelp_ref = kelp_gdf[kelp_gdf['year'] == year_filter].copy()
    if kelp_ref.empty:
        return None
    kelp_ref = kelp_ref[['eca_index_official', 'geometry', 'latitude', 'longitude', 'year_eca']].drop_duplicates()
    kelp_proj = kelp_ref.to_crs(epsg=3857)

    buffer_m = buffer_km * 1000
    kelp_proj['buffer_geom'] = kelp_proj.geometry.buffer(buffer_m)

    intersections = []
    for idx, kelp_row in kelp_proj.iterrows():
        nearby_hex = shipping_proj[shipping_proj.geometry.intersects(kelp_row['buffer_geom'])].copy()
        if not nearby_hex.empty:
            nearby_hex['eca_index_official'] = kelp_row['eca_index_official']
            intersections.append(nearby_hex)

    if not intersections:
        return None

    intersect_gdf = gpd.GeoDataFrame(pd.concat(intersections, ignore_index=True), crs=shipping_proj.crs)
    intersect_df = pd.DataFrame(intersect_gdf.drop(columns='geometry'))
    equivalence_table = intersect_df[['id', 'id_shipping', 'eca_index_official', 'year_eca']].drop_duplicates()

    exposure_temp = pd.merge(equivalence_table, shipping_quarter,
                             on=["id_shipping", "year_eca"], how="left")

    exposure_aggregated = exposure_temp.groupby(["eca_index_official", "year_eca"]).agg(
        sum_LENGTH=('LENGTH_sum', 'sum'),
        mean_LENGTH=('LENGTH_mean', 'mean'),
        sum_COUNT=('COUNT_sum', 'sum'),
        mean_COUNT=('COUNT_mean', 'mean'),
        id=('id', 'first')
    ).reset_index()

    exposure_aggregated = exposure_aggregated.rename(columns={
        'sum_LENGTH': f'sum_length{suffix}',
        'mean_LENGTH': f'mean_length{suffix}',
        'sum_COUNT': f'sum_count{suffix}',
        'mean_COUNT': f'mean_count{suffix}'
    })
    return exposure_aggregated


def create_complete_exposure_dataset(kelp_stats, years=range(2009, 2023)):
    print("=" * 80)
    print("STEP 2: SHIPPING EXPOSURE")
    print("=" * 80)

    shipping_exposure = pd.read_csv(os.path.join(RAW, 'exposure_shipping.csv'))
    eca_gpkg_path = os.path.join(RAW, 'ECA_10km.gpkg')

    print(f"  Shipping data: {len(shipping_exposure)} rows")
    print(f"  Kelp data: {len(kelp_stats)} rows")

    final_df = kelp_stats.copy()

    # Annual dynamic exposure (20km and 15km)
    for buffer_km in [20, 15]:
        print(f"  Buffer = {buffer_km}km")
        for year in years:
            suffix = f"_{year}_{buffer_km}km"
            exposure = compute_shipping_exposure_flexible(
                shipping_exposure=shipping_exposure.copy(), kelp_stats=kelp_stats,
                eca_gpkg_path=eca_gpkg_path, year_filter=year,
                start_month=1, end_month=12, buffer_km=buffer_km, suffix=suffix)
            if exposure is not None:
                final_df = pd.merge(final_df,
                    exposure[['eca_index_official', 'year_eca'] +
                             [c for c in exposure.columns if suffix in c]],
                    on=['eca_index_official', 'year_eca'], how='left')

    # 2009 Jan-Jun reference
    print("  Computing 2009 Jan-Jun reference...")
    for buffer_km in [20, 15]:
        suffix = f"_2009_jan_jun_{buffer_km}km"
        exposure = compute_shipping_exposure_flexible(
            shipping_exposure=shipping_exposure.copy(), kelp_stats=kelp_stats,
            eca_gpkg_path=eca_gpkg_path, year_filter=2009,
            start_month=1, end_month=6, buffer_km=buffer_km, suffix=suffix)
        if exposure is not None:
            final_df = pd.merge(final_df,
                exposure[['eca_index_official', 'year_eca'] +
                         [c for c in exposure.columns if suffix in c]],
                on=['eca_index_official', 'year_eca'], how='left')

    # Pre-ECA 2009 (Jan-Jul)
    print("  Computing pre-ECA 2009 (Jan-Jul)...")
    for buffer_km in [20, 15]:
        suffix = f"_pre_eca_2009_{buffer_km}km"
        exposure = compute_shipping_exposure_flexible(
            shipping_exposure=shipping_exposure.copy(), kelp_stats=kelp_stats,
            eca_gpkg_path=eca_gpkg_path, year_filter=2009,
            start_month=1, end_month=7, buffer_km=buffer_km, suffix=suffix)
        if exposure is not None:
            exposure = exposure.rename(columns={
                f'sum_length{suffix}': f'pre_eca_2009_sum_length_{buffer_km}km',
                f'mean_length{suffix}': f'pre_eca_2009_mean_length_{buffer_km}km',
                f'sum_count{suffix}': f'pre_eca_2009_sum_count_{buffer_km}km',
                f'mean_count{suffix}': f'pre_eca_2009_mean_count_{buffer_km}km'
            })
            final_df = pd.merge(final_df,
                exposure[['eca_index_official', 'year_eca'] +
                         [c for c in exposure.columns if 'pre_eca_2009' in c]],
                on=['eca_index_official', 'year_eca'], how='left')

    # Pre-ECA 2011 (Jul-Dec)
    print("  Computing pre-ECA 2011 (Jul-Dec)...")
    for buffer_km in [20, 15]:
        suffix = f"_pre_eca_2011_{buffer_km}km"
        exposure = compute_shipping_exposure_flexible(
            shipping_exposure=shipping_exposure.copy(), kelp_stats=kelp_stats,
            eca_gpkg_path=eca_gpkg_path, year_filter=2011,
            start_month=7, end_month=12, buffer_km=buffer_km, suffix=suffix)
        if exposure is not None:
            exposure = exposure.rename(columns={
                f'sum_length{suffix}': f'pre_eca_2011_sum_length_{buffer_km}km',
                f'mean_length{suffix}': f'pre_eca_2011_mean_length_{buffer_km}km',
                f'sum_count{suffix}': f'pre_eca_2011_sum_count_{buffer_km}km',
                f'mean_count{suffix}': f'pre_eca_2011_mean_count_{buffer_km}km'
            })
            final_df = pd.merge(final_df,
                exposure[['eca_index_official', 'year_eca'] +
                         [c for c in exposure.columns if 'pre_eca_2011' in c]],
                on=['eca_index_official', 'year_eca'], how='left')

    out_path = os.path.join(GENERATED, 'complete_kelp_shipping_exposure.csv')
    final_df.to_csv(out_path, index=False)
    print(f"  Saved {final_df.shape} to {out_path}")
    return final_df


# ============================================================
# STEP 3: SENSITIVITY INDEX (Halpern cumulative)
# ============================================================
def compute_sensitivity_index():
    print("=" * 80)
    print("STEP 3: SENSITIVITY INDEX (Halpern cumulative)")
    print("=" * 80)

    agg_path = os.path.join(GENERATED, 'agg_raster_results.csv')
    if os.path.exists(agg_path):
        print(f"  Found pre-computed {agg_path}")
        return pd.read_csv(agg_path)

    exposure_path = os.path.join(RAW, 'full_model.tif')
    eca_path = os.path.join(RAW, 'ECA_10km.gpkg')

    eca_hex = gpd.read_file(eca_path)
    eca_hex = eca_hex.rename(columns={'id': 'eca_index_official', 'year': 'year_eca'})

    eca_hex = eca_hex[~eca_hex.geometry.isna() & ~eca_hex.geometry.is_empty].copy()
    eca_hex = eca_hex.reset_index(drop=True)
    invalid_count = (~eca_hex.geometry.is_valid).sum()
    if invalid_count > 0:
        eca_hex['geometry'] = eca_hex.geometry.buffer(0)

    with rasterio.open(exposure_path) as src:
        raster_crs = src.crs
        raster_nodata = src.nodata

    if eca_hex.crs != raster_crs:
        eca_hex = eca_hex.to_crs(raster_crs)

    stats = zonal_stats(eca_hex, exposure_path, stats=['mean', 'count'],
                        nodata=raster_nodata, all_touched=True)

    eca_hex['model_2008'] = [s.get('mean', np.nan) if s else np.nan for s in stats]
    eca_hex['pixel_count'] = [s.get('count', 0) if s else 0 for s in stats]

    agg_raster = eca_hex[['eca_index_official', 'model_2008', 'pixel_count', 'year_eca']].copy()
    agg_raster.to_csv(agg_path, index=False)
    print(f"  Saved to {agg_path}")
    return agg_raster


# ============================================================
# STEP 4: COVARIATES MERGE
# ============================================================
def merge_covariables_with_exposure(exposure_final_with_model):
    print("=" * 80)
    print("STEP 4: COVARIATES MERGE")
    print("=" * 80)

    kelp_equivalence_path = os.path.join(RAW, 'kelp_equivalence_covariates.gpkg')
    kelp_equivalence = gpd.read_file(kelp_equivalence_path)
    kelp_equivalence = kelp_equivalence[['id', 'ECA_index_official']].rename(
        columns={'ECA_index_official': 'eca_index_official'})

    cov_csv_paths = {
        "precipitation": os.path.join(RAW, "precipitation_mean_official.csv"),
        "radiation": os.path.join(RAW, "radiation_mean_official.csv"),
        "sst": os.path.join(RAW, "sst_mean_official.csv"),
        "swh": os.path.join(RAW, "swh_mean_official.csv"),
    }

    processed = []
    for var_name, path in cov_csv_paths.items():
        df = pd.read_csv(path)
        df.columns = [col.lower() for col in df.columns]
        if "geometry" in df.columns:
            df = df.drop(columns="geometry")
        if "month" in df.columns:
            df["quarter"] = ((df["month"] - 1) // 3 + 1).astype(int)

        exclude_cols = ["unnamed: 0", "eca_index_official", "year", "month", "quarter"]
        value_cols = [c for c in df.columns if c not in exclude_cols]
        if len(value_cols) != 1:
            raise ValueError(f"Cannot determine value column in {path}: {value_cols}")
        value_col = value_cols[0]

        group_cols = ["eca_index_official", "year"]
        if "quarter" in df.columns:
            group_cols.append("quarter")

        grouped = (df.groupby(group_cols)[value_col].mean().reset_index()
                   .rename(columns={value_col: f"{var_name}_mean"}))
        processed.append(grouped)

    covariables_df = processed[0]
    for df in processed[1:]:
        merge_cols = [c for c in ["eca_index_official", "year", "quarter"]
                      if c in covariables_df.columns and c in df.columns]
        covariables_df = pd.merge(covariables_df, df, on=merge_cols, how='left')

    covariables_df = pd.merge(covariables_df, kelp_equivalence,
                              on='eca_index_official', how='left')
    del covariables_df['eca_index_official']

    if '_merge' in exposure_final_with_model.columns:
        del exposure_final_with_model['_merge']

    exposure_final_with_model['id'] = exposure_final_with_model['eca_index_official']

    exposure_with_cov = pd.merge(covariables_df, exposure_final_with_model,
                                 on=['year', 'quarter', 'id'], how='right')

    print(f"  Merged covariates: {exposure_with_cov.shape}")
    return exposure_with_cov


# ============================================================
# STEP 6: SHIPPING-SPECIFIC TREATMENT (impact_shipping.tif)
# ============================================================
def compute_shipping_sensitivity():
    """Extract mean shipping pressure per ECA hexagon from impact_shipping.tif."""
    print("=" * 80)
    print("STEP 6: SHIPPING SENSITIVITY INDEX")
    print("=" * 80)

    agg_path = os.path.join(GENERATED, 'agg_shipping_raster_results.csv')
    if os.path.exists(agg_path):
        print(f"  Found pre-computed {agg_path}")
        return pd.read_csv(agg_path)

    ship_path = os.path.join(RAW, 'impact_shipping', 'impact_shipping.tif')
    eca_path = os.path.join(RAW, 'ECA_10km.gpkg')

    # Load hexagons (same pattern as Step 3)
    eca_hex = gpd.read_file(eca_path)
    eca_hex = eca_hex.rename(columns={'id': 'eca_index_official', 'year': 'year_eca'})
    eca_hex = eca_hex[~eca_hex.geometry.isna() & ~eca_hex.geometry.is_empty].copy()
    eca_hex = eca_hex.reset_index(drop=True)
    invalid_count = (~eca_hex.geometry.is_valid).sum()
    if invalid_count > 0:
        print(f"  Fixing {invalid_count} invalid geometries")
        eca_hex['geometry'] = eca_hex.geometry.buffer(0)

    # Read raster CRS and detect nodata
    with rasterio.open(ship_path) as src:
        raster_crs = src.crs
        raster_nodata = src.nodata
        raster_data = src.read(1)
        print(f"  Raster shape: {raster_data.shape}, CRS: {raster_crs}")
        print(f"  Raster metadata nodata: {raster_nodata}")
        # Handle implicit nodata (very large negative float)
        if raster_nodata is None:
            raster_nodata = float(raster_data.min())
            print(f"  Using min value as nodata sentinel: {raster_nodata:.2e}")
        # Positive pixel statistics
        pos_mask = raster_data > 0
        print(f"  Positive pixels: {pos_mask.sum()} / {raster_data.size} "
              f"({100 * pos_mask.sum() / raster_data.size:.1f}%)")
        if pos_mask.sum() > 0:
            pos_vals = raster_data[pos_mask]
            print(f"  Positive values: mean={pos_vals.mean():.3f}, "
                  f"median={np.median(pos_vals):.3f}, "
                  f"min={pos_vals.min():.3f}, max={pos_vals.max():.3f}")

    # Reproject hexagons if needed
    if eca_hex.crs != raster_crs:
        eca_hex = eca_hex.to_crs(raster_crs)

    # Zonal statistics
    print("  Running zonal_stats...")
    stats = zonal_stats(eca_hex, ship_path, stats=['mean', 'count'],
                        nodata=raster_nodata, all_touched=True)

    eca_hex['shipping_2008'] = [s.get('mean', np.nan) if s else np.nan for s in stats]
    eca_hex['shipping_pixel_count'] = [s.get('count', 0) if s else 0 for s in stats]

    # Summary
    ship_vals = eca_hex['shipping_2008'].dropna()
    print(f"\n  Hexagons with valid shipping values: {len(ship_vals)} / {len(eca_hex)}")
    print(f"  shipping_2008 summary:")
    print(f"    mean={ship_vals.mean():.4f}, std={ship_vals.std():.4f}")
    print(f"    min={ship_vals.min():.4f}, max={ship_vals.max():.4f}")
    print(f"    zeros: {(ship_vals == 0).sum()}, NaN: {eca_hex['shipping_2008'].isna().sum()}")

    # Save
    agg_shipping = eca_hex[['eca_index_official', 'shipping_2008',
                            'shipping_pixel_count', 'year_eca']].copy()
    agg_shipping.to_csv(agg_path, index=False)
    print(f"  Saved to {agg_path}")
    return agg_shipping


def augment_with_shipping(df, agg_shipping):
    """Merge shipping_2008 into the regression dataset."""
    print("\n  Merging shipping treatment into dataset...")
    n_before = len(df)

    merge_cols = ['eca_index_official', 'year_eca']
    ship_merge = agg_shipping[merge_cols + ['shipping_2008', 'shipping_pixel_count']]
    df = df.merge(ship_merge, on=merge_cols, how='left')
    assert len(df) == n_before, f"Row count changed: {n_before} -> {len(df)}"

    # Log transform
    df['ln_shipping_2008'] = np.log1p(df['shipping_2008'])

    # Validation
    print(f"    shipping_2008 NaN: {df['shipping_2008'].isna().sum()}")

    # Correlation with model_2008
    valid = df[['model_2008', 'shipping_2008']].dropna()
    if len(valid) > 0:
        corr = valid['model_2008'].corr(valid['shipping_2008'])
        print(f"    Correlation(model_2008, shipping_2008) = {corr:.4f}")

    # Check variation in regression sample (lat 32-36, year <= 2016)
    reg_sample = df[(df['year'] <= 2016) & (df['latitude'] >= 32) & (df['latitude'] <= 36)]
    ship_unique = reg_sample.drop_duplicates('eca_index_official')['shipping_2008']
    print(f"    Regression sample: {len(ship_unique)} unique cells, "
          f"std={ship_unique.std():.4f}, "
          f"IQR={ship_unique.quantile(0.75) - ship_unique.quantile(0.25):.4f}")

    return df


# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":
    # Step 1: Kelp analysis
    kelp_stats_path = os.path.join(GENERATED, 'kelp_analysis_official.csv')
    if os.path.exists(kelp_stats_path):
        print(f"Found existing {kelp_stats_path}, loading...")
        kelp_stats = pd.read_csv(kelp_stats_path)
    else:
        kelp_stats = kelp_analysis_official(start_year=2005, end_year=2019)

    # Step 2: Shipping exposure
    exposure_path = os.path.join(GENERATED, 'complete_kelp_shipping_exposure.csv')
    if os.path.exists(exposure_path):
        print(f"Found existing {exposure_path}, loading...")
        final_dataset = pd.read_csv(exposure_path)
    else:
        final_dataset = create_complete_exposure_dataset(kelp_stats, years=range(2009, 2023))

    # Step 3: Sensitivity index
    agg_raster = compute_sensitivity_index()
    exposure_final_with_model = final_dataset.merge(
        agg_raster, on=['eca_index_official', 'year_eca'], how='left')

    # Step 4: Covariates merge
    exposure_final_with_cov = merge_covariables_with_exposure(exposure_final_with_model)

    # Step 5: Log transformation
    print("=" * 80)
    print("STEP 5: LOG TRANSFORMATION")
    print("=" * 80)
    exclude = ['eca_index_official', 'id', 'year', 'quarter', 'year_eca']
    cols_to_log = (exposure_final_with_cov
                   .select_dtypes(include=[np.number])
                   .columns.difference(exclude))
    exposure_final_with_cov[[f"ln_{col}" for col in cols_to_log]] = np.log1p(
        exposure_final_with_cov[cols_to_log])

    # Save intermediate (kept for backward compatibility)
    intermediate_path = os.path.join(GENERATED, 'new_model_kelp_exposure_2.csv')
    exposure_final_with_cov.to_csv(intermediate_path, index=False)
    print(f"  Intermediate dataset: {exposure_final_with_cov.shape}")
    print(f"  Saved to: {intermediate_path}")

    # Step 6: Shipping-specific treatment
    agg_shipping = compute_shipping_sensitivity()
    final_df = augment_with_shipping(exposure_final_with_cov, agg_shipping)

    # Save final
    final_path = os.path.join(GENERATED, 'new_model_kelp_exposure_3.csv')
    final_df.to_csv(final_path, index=False)
    print("=" * 80)
    print(f"DONE! Final dataset: {final_df.shape}")
    print(f"Saved to: {final_path}")
    print("=" * 80)
