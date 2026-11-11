import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyfixest as pf
raw = pd.read_csv(r"c:\Users\fcanda01\Desktop\Recherche\kelp\data\generated\new_model_kelp_exposure_3.csv", low_memory=False)
raw = raw[(raw.year<=2016)&(raw.latitude>=32)&(raw.latitude<=36)].copy()
raw["cell_id"]=raw.eca_index_official.astype(str)+"_"+raw.year_eca.astype(str)
raw["yq"]=raw.year.astype(str)+"_"+raw.quarter.astype(str)
raw=raw.drop_duplicates(["cell_id","yq"])
for c in ["sst","precipitation","swh"]: raw["ln_"+c]=raw[f"ln_{c}_mean"]
raw["pol11"]=(raw.year>2011).astype(int)
c09="ln_mean_count_2009_jan_jun_20km"; c11="ln_mean_count_2011_20km"; l11="ln_mean_length_2011_20km"
d=raw.dropna(subset=[c11,"ln_sst","ln_precipitation","ln_swh","biomass_sum"]).copy()
g=d.groupby("cell_id")[c09].first()
d["grp"]=d.cell_id.map(lambda x: "A_no2009" if pd.isna(g[x]) else ("B_lt2" if g[x]<np.log(2) else ("C_2to4" if g[x]<np.log(4) else "D_ge4")))
print(d.groupby("grp").cell_id.nunique())
V={"CRV1":"cell_id"}
def run(dd,t,lab):
    try:
        e=pf.fepois(f"biomass_sum ~ {t}:pol11 + ln_sst + ln_precipitation + ln_swh | cell_id + yq",data=dd,vcov=V)
        k=[x for x in e.coef().index if t in x][0]
        print(f"{lab:38s} {t[8:13]:6s} {e.coef()[k]:+.3f} (se {e.se()[k]:.3f}, p={e.pvalue()[k]:.3f}) cells={dd.cell_id.nunique()} N={e._N}")
    except Exception as ex: print(lab, "ERR", ex)
print("\n2011 design, biomass")
for lab,dd in [("all cells (paper)",d),("81-cell 2009 sample (C+D)",d[d.grp.isin(["C_2to4","D_ge4"])]),("cells outside 2009 sample (A+B)",d[d.grp.isin(["A_no2009","B_lt2"])]),("D only (>=4 vessels 2009)",d[d.grp=="D_ge4"])]:
    for t in [l11,c11]: run(dd,t,lab)
print("\nMean biomass per cell-quarter by group and year, index 2008=100")
m=d.groupby(["grp","year"]).biomass_sum.mean().unstack(0)
print((100*m/m.loc[2008]).round(0).to_string())
print("\nMean 2011 log count by group:"); print(d.groupby("grp")[c11].mean().round(2))
print("\nlat/lon by group:"); print(d.groupby("grp")[["latitude","longitude"]].agg(["mean","min","max"]).round(2).to_string())
