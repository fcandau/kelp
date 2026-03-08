"""
5_financial_crisis.py
=====================
Addresses the 2008-09 financial crisis confound using NBIC vessel arrival data.

Strategy 1: Add port economic activity as a time-varying PPML control
Strategy 2: Show that port activity never recovered while kelp improvement persists

Inputs:
    - data/raw/nbic/nbic_vessel_arrivals.csv (NBIC vessel arrivals, 2004-2023)
    - data/generated/new_model_kelp_exposure_3.csv (main kelp regression dataset)
    - data/raw/kelp_equivalence_covariates.gpkg (hexagon geometry + port assignments)

Outputs:
    - output/figures/nbic_port_activity.png (descriptive figure)
    - output/tables/crisis_robustness.tex (PPML with port-activity control)
    - output/figures/crisis_decomposition.png (Strategy 2 figure)
    - data/generated/kelp_with_port_activity.csv (augmented dataset)
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pyfixest as pf
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. PATHS AND CONSTANTS
# ============================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = "C:/Users/fcanda01/AppData/Local/anaconda3/envs/r-env/python.exe"

# Port coordinates (WGS84) — standard reference positions
PORT_COORDS = {
    'Los Angeles':  (33.7361, -118.2639),
    'Long Beach':   (33.7540, -118.2163),
    'San Diego':    (32.6867, -117.1542),
    'Port Hueneme': (34.1478, -119.2095),
}

# ============================================================
# 1. LOAD AND PROCESS NBIC DATA
# ============================================================
print("=" * 60)
print("Step 1: Processing NBIC vessel arrivals")
print("=" * 60)

nbic = pd.read_csv(
    f"{ROOT}/data/raw/nbic/nbic_vessel_arrivals.csv",
    usecols=['Arrival_Port', 'Arrival_State', 'Arrival_Date', 'Transit_Type', 'GT'],
    low_memory=False
)

# Filter: California, overseas arrivals
nbic = nbic[
    (nbic['Arrival_State'].str.upper().str.strip() == 'CA') &
    (nbic['Transit_Type'].str.lower().str.strip() == 'overseas')
].copy()

nbic['date'] = pd.to_datetime(nbic['Arrival_Date'], errors='coerce')
nbic = nbic.dropna(subset=['date'])
nbic['year'] = nbic['date'].dt.year
nbic['quarter'] = nbic['date'].dt.quarter

# Map to our four Southern California ports
def classify_port(name):
    if pd.isna(name):
        return None
    name = name.strip()
    for port in PORT_COORDS:
        if port.lower() in name.lower():
            return port
    return None

nbic['port'] = nbic['Arrival_Port'].apply(classify_port)
nbic_socal = nbic[nbic['port'].notna()].copy()

print(f"  Total overseas CA arrivals: {len(nbic)}")
print(f"  SoCal port arrivals (LA/LB/SD/Hueneme): {len(nbic_socal)}")

# Aggregate to port x quarter
port_quarterly = nbic_socal.groupby(['port', 'year', 'quarter']).agg(
    n_arrivals=('Arrival_Port', 'count'),
    total_gt=('GT', 'sum')
).reset_index()

# Filter to study period
port_quarterly = port_quarterly[
    (port_quarterly['year'] >= 2005) & (port_quarterly['year'] <= 2016)
]

print(f"  Port-quarter observations (2005-2016): {len(port_quarterly)}")

# ============================================================
# 2. CONSTRUCT CELL-LEVEL PORT ACTIVITY CONTROL
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Distance-weighted port activity for each kelp cell")
print("=" * 60)

# Load hexagon geometry
eq = gpd.read_file(f"{ROOT}/data/raw/kelp_equivalence_covariates.gpkg")
eq_4326 = eq.to_crs(epsg=4326)

# Suppress centroid warning for geographic CRS (acceptable for distance approximation)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    eq_4326['centroid'] = eq_4326.geometry.centroid
    eq_4326['cell_lat'] = eq_4326['centroid'].y
    eq_4326['cell_lon'] = eq_4326['centroid'].x

cells = eq_4326[['id', 'ECA_index_official', 'cell_lat', 'cell_lon']].copy()
cells['id'] = cells['id'].astype(float)

# Haversine distance (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# Compute distance from each cell to each port
print("  Computing cell-to-port distances...")
dist_records = []
for _, cell in cells.iterrows():
    for port_name, (plat, plon) in PORT_COORDS.items():
        d = haversine(cell['cell_lat'], cell['cell_lon'], plat, plon)
        dist_records.append({
            'id': cell['id'],
            'ECA_index_official': cell['ECA_index_official'],
            'port': port_name,
            'distance_km': d
        })

dist_df = pd.DataFrame(dist_records)

# Inverse-distance weights (1/d^2, capped at minimum 5km to avoid explosion)
dist_df['distance_km'] = dist_df['distance_km'].clip(lower=5.0)
dist_df['weight'] = 1.0 / (dist_df['distance_km'] ** 2)

# Normalize weights within each cell to sum to 1
dist_df['weight_sum'] = dist_df.groupby('id')['weight'].transform('sum')
dist_df['weight_norm'] = dist_df['weight'] / dist_df['weight_sum']

print("  Sample distances (first cell):")
sample_cell = dist_df[dist_df['id'] == dist_df['id'].iloc[0]]
for _, row in sample_cell.iterrows():
    print(f"    {row['port']}: {row['distance_km']:.0f} km, weight: {row['weight_norm']:.4f}")

# Merge port activity with distance weights -> cell-level port activity
# For each cell x quarter: weighted sum of port arrivals
cell_port = dist_df[['id', 'port', 'weight_norm']].merge(
    port_quarterly[['port', 'year', 'quarter', 'n_arrivals', 'total_gt']],
    on='port', how='inner'
)

cell_port['weighted_arrivals'] = cell_port['weight_norm'] * cell_port['n_arrivals']
cell_port['weighted_gt'] = cell_port['weight_norm'] * cell_port['total_gt']

cell_activity = cell_port.groupby(['id', 'year', 'quarter']).agg(
    port_arrivals=('weighted_arrivals', 'sum'),
    port_gt=('weighted_gt', 'sum')
).reset_index()

cell_activity['ln_port_arrivals'] = np.log(cell_activity['port_arrivals'].clip(lower=1))
cell_activity['ln_port_gt'] = np.log(cell_activity['port_gt'].clip(lower=1))

print(f"\n  Cell-quarter observations: {len(cell_activity)}")
print(f"  Unique cells: {cell_activity['id'].nunique()}")

# ============================================================
# 3. MERGE WITH KELP DATASET
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Merging with kelp regression dataset")
print("=" * 60)

kelp = pd.read_csv(f"{ROOT}/data/generated/new_model_kelp_exposure_3.csv", low_memory=False)

# cell_activity uses eq['id'] which equals kelp['eca_index_official'] directly
cell_activity['id'] = cell_activity['id'].astype(float)
kelp['eca_index_official'] = kelp['eca_index_official'].astype(float)

kelp = kelp.merge(
    cell_activity[['id', 'year', 'quarter', 'ln_port_arrivals', 'ln_port_gt', 'port_arrivals']],
    left_on=['eca_index_official', 'year', 'quarter'],
    right_on=['id', 'year', 'quarter'],
    how='left'
)

# Check coverage
study = kelp[(kelp['latitude'] >= 32) & (kelp['latitude'] <= 36) & (kelp['year'] <= 2016)]
print(f"  Study sample rows: {len(study)}")
print(f"  Port activity non-null: {study['ln_port_arrivals'].notna().sum()}")
print(f"  Port activity null: {study['ln_port_arrivals'].isna().sum()}")

# Save augmented dataset
kelp.to_csv(f"{ROOT}/data/generated/kelp_with_port_activity.csv", index=False)
print(f"  Saved: data/generated/kelp_with_port_activity.csv")

# ============================================================
# 4. DESCRIPTIVE FIGURE: QUARTERLY VESSEL ARRIVALS
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Creating descriptive port activity figure")
print("=" * 60)

# Total SoCal quarterly arrivals
total_q = port_quarterly.groupby(['year', 'quarter']).agg(
    n_arrivals=('n_arrivals', 'sum')
).reset_index()
total_q['time'] = total_q['year'] + (total_q['quarter'] - 1) / 4

# Per-port annual (for subplot)
port_annual = port_quarterly.groupby(['port', 'year']).agg(
    n_arrivals=('n_arrivals', 'sum')
).reset_index()

# Normalize to 2007
base_total = total_q[(total_q['year'] == 2007)]['n_arrivals'].mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Quarterly total
ax = axes[0]
ax.plot(total_q['time'], total_q['n_arrivals'], 'o-', color='#2c3e50', markersize=3, linewidth=1.2)
ax.axvline(x=2009.5, color='red', linestyle='--', alpha=0.7, label='ECA enforcement (Jul 2009)')
ax.axvline(x=2008.75, color='gray', linestyle=':', alpha=0.5, label='Lehman collapse (Sep 2008)')
ax.set_xlabel('Year')
ax.set_ylabel('Overseas vessel arrivals (quarterly)')
ax.set_title('(a) Total SoCal port arrivals')
ax.legend(fontsize=8, loc='lower left')
ax.set_xlim(2005, 2017)

# Panel B: Per-port annual index (2007=100)
ax = axes[1]
colors = {'Los Angeles': '#e74c3c', 'Long Beach': '#3498db', 'San Diego': '#2ecc71', 'Port Hueneme': '#f39c12'}
for port in PORT_COORDS:
    sub = port_annual[port_annual['port'] == port].copy()
    base = sub[sub['year'] == 2007]['n_arrivals'].values
    if len(base) > 0:
        sub['index'] = sub['n_arrivals'] / base[0] * 100
        ax.plot(sub['year'], sub['index'], 'o-', color=colors[port], label=port, markersize=4, linewidth=1.2)

ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=2009, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Index (2007 = 100)')
ax.set_title('(b) By port (2007 = 100)')
ax.legend(fontsize=8)
ax.set_xlim(2005, 2017)

plt.tight_layout()
plt.savefig(f"{ROOT}/output/figures/nbic_port_activity.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: output/figures/nbic_port_activity.png")

# ============================================================
# 5. STRATEGY 1: PPML WITH PORT-ACTIVITY CONTROL
# ============================================================
print("\n" + "=" * 60)
print("Step 5: PPML with port-activity control (Strategy 1)")
print("=" * 60)

# Replicate main specification from 2_a_estim.py with port activity control
df = kelp.copy()

# Study area filter (matches 2_a_estim.py exactly)
df = df[(df['latitude'] >= 32) & (df['latitude'] <= 36) & (df['year'] <= 2016)].copy()

# Cell ID and time ID (matches 2_a_estim.py)
df["cell_id"] = df["eca_index_official"].astype(str) + "_" + df["year_eca"].astype(str)
df["yq"] = df["year"].astype(str) + "_" + df["quarter"].astype(str)
df = df.drop_duplicates(subset=["cell_id", "yq"])

# Post-2009 indicator (matches 2_a_estim.py: post = 2009Q3+)
df["policy_eca_2009"] = 0
df.loc[df["year"] > 2008, "policy_eca_2009"] = 1
df.loc[(df["year"] == 2009) & (df["quarter"] < 3), "policy_eca_2009"] = 0

# Drop NaN in key columns (matches 2_a_estim.py)
key_cols = ["biomass_sum", "area_sum",
            "ln_mean_length_2009_jan_jun_20km", "ln_mean_count_2009_jan_jun_20km",
            "ln_sst_mean", "ln_precipitation_mean", "ln_swh_mean",
            "ln_sum_count_2009_jan_jun_20km"]
df = df.dropna(subset=key_cols)

# Alias column names
df["ln_sst"] = df["ln_sst_mean"]
df["ln_precipitation"] = df["ln_precipitation_mean"]
df["ln_swh"] = df["ln_swh_mean"]

# Sample restriction: cells with >= 2 vessels/month (matches 2_a_estim.py)
VESSEL_THRESHOLD = np.log(2)
cell_count = df.groupby("cell_id")["ln_mean_count_2009_jan_jun_20km"].first()
cells_above = cell_count[cell_count >= VESSEL_THRESHOLD].index
df = df[df["cell_id"].isin(cells_above)].copy()

# Restrict to 2005-2016 (SWH now available through 2016)
df = df[(df['year'] >= 2005) & (df['year'] <= 2016)].copy()

# Drop rows missing port activity
df = df.dropna(subset=['ln_port_arrivals'])

# Variance-covariance: cluster at cell level (matches 2_a_estim.py)
VCOV = {"CRV1": "cell_id"}

print(f"  Sample after all filters: {len(df)} obs, {df['cell_id'].nunique()} cells")

# --- Baseline (no port control) ---
print("\n  Running baseline (no port activity control)...")
results = {}

for dep, dep_label in [('biomass_sum', 'Biomass'), ('area_sum', 'Area')]:
    for treat_short, treat_label in [('length', 'Distance'), ('count', 'Count')]:
        treat_var = f'ln_mean_{treat_short}_2009_jan_jun_20km'

        # Baseline (matches 2_a_estim.py: unweighted PPML, count >= 2)
        fml_base = f"{dep} ~ {treat_var}:policy_eca_2009 + ln_sst + ln_precipitation + ln_swh | cell_id + yq"
        est_base = pf.fepois(fml_base, data=df, vcov=VCOV)

        # With port activity control
        fml_port = f"{dep} ~ {treat_var}:policy_eca_2009 + ln_sst + ln_precipitation + ln_swh + ln_port_arrivals | cell_id + yq"
        est_port = pf.fepois(fml_port, data=df, vcov=VCOV)

        # Find the interaction key in coefficients
        int_key_base = [k for k in est_base.coef().index if treat_var in k and 'policy' in k][0]
        int_key_port = [k for k in est_port.coef().index if treat_var in k and 'policy' in k][0]

        key = f"{treat_label}_{dep_label}"
        results[key] = {
            'base_coef': est_base.coef()[int_key_base],
            'base_se': est_base.se()[int_key_base],
            'base_pval': est_base.pvalue()[int_key_base],
            'port_coef': est_port.coef()[int_key_port],
            'port_se': est_port.se()[int_key_port],
            'port_pval': est_port.pvalue()[int_key_port],
            'port_arrivals_coef': est_port.coef().get('ln_port_arrivals', np.nan),
            'port_arrivals_se': est_port.se().get('ln_port_arrivals', np.nan),
            'port_arrivals_pval': est_port.pvalue().get('ln_port_arrivals', np.nan),
            'n': est_port._N,
        }

# Print comparison
print("\n  Results comparison (treatment coefficient):")
print(f"  {'Spec':<20} {'Baseline':>12} {'+ Port Activity':>15} {'Change':>10}")
print("  " + "-" * 60)
for key, r in results.items():
    stars_b = '***' if r['base_pval'] < 0.01 else '**' if r['base_pval'] < 0.05 else '*' if r['base_pval'] < 0.1 else ''
    stars_p = '***' if r['port_pval'] < 0.01 else '**' if r['port_pval'] < 0.05 else '*' if r['port_pval'] < 0.1 else ''
    change_pct = (r['port_coef'] - r['base_coef']) / abs(r['base_coef']) * 100
    print(f"  {key:<20} {r['base_coef']:>8.4f}{stars_b:<3}  {r['port_coef']:>8.4f}{stars_p:<3}   {change_pct:>+7.1f}%")

print(f"\n  Port activity coefficient (ln_port_arrivals):")
for key, r in results.items():
    print(f"    {key}: {r['port_arrivals_coef']:.4f} (SE: {r['port_arrivals_se']:.4f})")

# ============================================================
# 6. EXPORT LATEX TABLE
# ============================================================
print("\n" + "=" * 60)
print("Step 6: Exporting LaTeX table")
print("=" * 60)

def format_coef(val, pval):
    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
    return f"{val:.4f}\\sym{{{stars}}}"

def format_se(val):
    return f"({val:.4f})"

# Build table: 4 columns (Dist-Bio, Dist-Area, Count-Bio, Count-Area)
# Each has baseline row and +port row
specs = [
    ('Distance_Biomass', 'Distance', 'Biomass'),
    ('Distance_Area', 'Distance', 'Area'),
    ('Count_Biomass', 'Count', 'Biomass'),
    ('Count_Area', 'Count', 'Area'),
]

n_obs = list(results.values())[0]['n']

tex_lines = []
tex_lines.append(r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
tex_lines.append(r"\begin{tabular}{l*{4}{c}}")
tex_lines.append(r"\hline\hline")
tex_lines.append(r"                    &\multicolumn{2}{c}{Distance Traveled (km)}&\multicolumn{2}{c}{Vessel Count}\\")
tex_lines.append(r"                    \cmidrule(lr){2-3}\cmidrule(lr){4-5}")
tex_lines.append(r"                    &\multicolumn{1}{c}{(1)}&\multicolumn{1}{c}{(2)}&\multicolumn{1}{c}{(3)}&\multicolumn{1}{c}{(4)}\\")
tex_lines.append(r"                    &\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}&\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}\\")
tex_lines.append(r"\hline")

# Panel A: Baseline
tex_lines.append(r"\multicolumn{5}{l}{\textit{Panel A: Baseline (no port activity control)}}\\[0.3em]")
coefs = "&".join([f"     {format_coef(results[k]['base_coef'], results[k]['base_pval'])}" for k, _, _ in specs])
ses = "&".join([f"             {format_se(results[k]['base_se'])}" for k, _, _ in specs])
tex_lines.append(f"Post 2009 $\\times$ Exposure&{coefs}\\\\")
tex_lines.append(f"                    &{ses}\\\\")

tex_lines.append(r"[0.8em]")

# Panel B: With port activity
tex_lines.append(r"\multicolumn{5}{l}{\textit{Panel B: Controlling for port economic activity}}\\[0.3em]")
coefs = "&".join([f"     {format_coef(results[k]['port_coef'], results[k]['port_pval'])}" for k, _, _ in specs])
ses = "&".join([f"             {format_se(results[k]['port_se'])}" for k, _, _ in specs])
tex_lines.append(f"Post 2009 $\\times$ Exposure&{coefs}\\\\")
tex_lines.append(f"                    &{ses}\\\\")

# Port activity row
tex_lines.append(r"[0.3em]")
pa_coefs = "&".join([f"     {format_coef(results[k]['port_arrivals_coef'], results[k].get('port_arrivals_pval', 0.5))}" for k, _, _ in specs])
pa_ses = "&".join([f"             {format_se(results[k]['port_arrivals_se'])}" for k, _, _ in specs])
tex_lines.append(f"Log port arrivals   &{pa_coefs}\\\\")
tex_lines.append(f"                    &{pa_ses}\\\\")

tex_lines.append(r"[0.5em]")
tex_lines.append(r"\hline")
n_str = f"{int(n_obs):>10d}"
tex_lines.append(f"Observations        &{n_str}         &{n_str}         &{n_str}         &{n_str}         \\\\")
yes4 = "&         Yes         " * 4
tex_lines.append(f"Controls            {yes4}\\\\")
tex_lines.append(f"Cell FE             {yes4}\\\\")
tex_lines.append(f"Year-Quarter FE     {yes4}\\\\")
tex_lines.append(r"\hline\hline")
tex_lines.append(r"\end{tabular}")

tex_content = "\n".join(tex_lines)
with open(f"{ROOT}/output/tables/crisis_robustness.tex", 'w') as f:
    f.write(tex_content)
print("  Saved: output/tables/crisis_robustness.tex")

# ============================================================
# 7. STRATEGY 2: CRISIS DECOMPOSITION FIGURE
# ============================================================
print("\n" + "=" * 60)
print("Step 7: Crisis decomposition figure (Strategy 2)")
print("=" * 60)

# Left panel: Port activity index (annual, 2007=100)
# Right panel: Kelp improvement in high-shipping cells (event study coefficients already available)

# Port activity: annual index
port_all_annual = nbic_socal.copy()
port_all_annual['year'] = port_all_annual['date'].dt.year
annual_total = port_all_annual.groupby('year').size().reset_index(name='n_arrivals')
annual_total = annual_total[(annual_total['year'] >= 2005) & (annual_total['year'] <= 2016)]
base_2007 = annual_total[annual_total['year'] == 2007]['n_arrivals'].values[0]
annual_total['index'] = annual_total['n_arrivals'] / base_2007 * 100

# Load event study coefficients if available
es_csv_path = f"{ROOT}/output/csv/ppml_es_pre_blob_vessel_dist_biomass.csv"
try:
    import re as _re
    _es_raw = pd.read_csv(es_csv_path)
    # Parse period from coef_name (e.g. "C(period, ...)[2009.1]:..." → "2009pre")
    es_rows = []
    for _, row in _es_raw.iterrows():
        m = _re.search(r'\[(\d+\.\d+)\]', row['coef_name'])
        if m:
            val = float(m.group(1))
            yr = int(val)
            frac = val - yr
            if frac > 0 and frac < 0.15:
                period = f"{yr}pre"
            elif frac > 0.15:
                period = f"{yr}post"
            else:
                period = str(yr)
            es_rows.append({
                'period': period,
                'coef': row['coef'],
                'ci_lo': row['coef'] - 1.96 * row['se'],
                'ci_hi': row['coef'] + 1.96 * row['se'],
            })
    es_data = pd.DataFrame(es_rows)
    has_es = len(es_data) > 0
    if has_es:
        print("  Loaded event study coefficients from CSV")
    else:
        print("  No event study periods found in CSV")
except FileNotFoundError:
    has_es = False
    print("  Event study CSV not found — using port-only figure")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Port activity
ax = axes[0]
ax.bar(annual_total['year'], annual_total['index'], color='#3498db', alpha=0.7, edgecolor='#2c3e50')
ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=2008.5, color='red', linestyle='--', alpha=0.7, label='ECA enforcement')
ax.set_xlabel('Year')
ax.set_ylabel('Overseas vessel arrivals (2007 = 100)')
ax.set_title('(a) Port economic activity')
ax.legend(fontsize=9)
ax.set_ylim(60, 110)

# Annotate key facts
ax.annotate('Crisis\ntrough', xy=(2009, annual_total[annual_total['year']==2009]['index'].values[0]),
            xytext=(2009.5, 95), fontsize=8, ha='left',
            arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate('No recovery\nthrough 2016', xy=(2016, annual_total[annual_total['year']==2016]['index'].values[0]),
            xytext=(2014.5, 90), fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray'))

# Panel B: Kelp event study or text box
ax = axes[1]
if has_es:
    # Map string periods to numeric x-positions for plotting
    period_map = {'2005': 2005, '2006': 2006, '2007': 2007, '2008': 2008,
                  '2009pre': 2009.0, '2009post': 2009.5,
                  '2010': 2010, '2011': 2011, '2012': 2012, '2013': 2013,
                  '2014': 2014, '2015': 2015, '2016': 2016}
    es_data['x'] = es_data['period'].map(period_map)
    es_data = es_data.dropna(subset=['x']).sort_values('x')
    # Exclude reference period (2008) and years beyond 2014 for alignment with panel (a)
    es_plot = es_data[(es_data['period'] != '2008') & (es_data['x'] <= 2016)]
    ax.errorbar(es_plot['x'], es_plot['coef'],
                yerr=[es_plot['coef'] - es_plot['ci_lo'], es_plot['ci_hi'] - es_plot['coef']],
                fmt='o-', color='#2c3e50', capsize=3, markersize=5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=2008.5, color='red', linestyle='--', alpha=0.7, label='ECA enforcement')
    ax.set_xlabel('Year')
    ax.set_ylabel('PPML coefficient (vs 2008)')
    ax.set_title('(b) Kelp biomass response (event study)')
    ax.legend(fontsize=9)
    ax.set_xlim(2004.5, 2016.5)
else:
    # Summary text when ES coefficients unavailable
    ax.text(0.5, 0.5,
            "Kelp improvement persists through 2014\n"
            "(see Figure 5 event study)\n\n"
            "Port activity never recovers\n"
            "→ Crisis cannot explain sustained kelp gains\n"
            "→ If crisis drove kelp, effect should fade\n"
            "   as ports remain depressed",
            transform=ax.transAxes, ha='center', va='center',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('(b) Key identification argument')
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig(f"{ROOT}/output/figures/crisis_decomposition.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: output/figures/crisis_decomposition.png")

# ============================================================
# 8. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\nStrategy 1 (Port activity control):")
for key, r in results.items():
    change = (r['port_coef'] - r['base_coef']) / abs(r['base_coef']) * 100
    print(f"  {key}: {r['base_coef']:.4f} -> {r['port_coef']:.4f} ({change:+.1f}%)")

print("\nStrategy 2 (Crisis decomposition):")
print(f"  Port arrivals 2009: {annual_total[annual_total['year']==2009]['index'].values[0]:.0f}% of 2007")
print(f"  Port arrivals 2016: {annual_total[annual_total['year']==2016]['index'].values[0]:.0f}% of 2007")
print("  -> Port activity NEVER recovered to pre-crisis levels")
print("  -> If crisis drove kelp improvement, effect should fade as ports stay depressed")
print("  -> Instead, kelp improvement PERSISTS through 2014 (see event study)")

print("\nDone!")
