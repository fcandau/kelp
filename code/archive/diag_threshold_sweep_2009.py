import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyfixest as pf
P = r"c:\Users\fcanda01\Desktop\Recherche\kelp\data\generated\new_model_kelp_exposure_3.csv"
raw = pd.read_csv(P, low_memory=False)
raw = raw[(raw.year<=2016)&(raw.latitude>=32)&(raw.latitude<=36)].copy()
raw["cell_id"]=raw.eca_index_official.astype(str)+"_"+raw.year_eca.astype(str)
raw["yq"]=raw.year.astype(str)+"_"+raw.quarter.astype(str)
raw=raw.drop_duplicates(["cell_id","yq"])
for c in ["sst","precipitation","swh"]: raw["ln_"+c]=raw[f"ln_{c}_mean"]
V={"CRV1":"cell_id"}; ctl="ln_sst + ln_precipitation + ln_swh"
def run(d,dep,t,pol):
    e=pf.fepois(f"{dep} ~ {t}:{pol} + {ctl} | cell_id + yq",data=d,vcov=V)
    k=[x for x in e.coef().index if t in x][0]
    return e.coef()[k],e.se()[k],e.pvalue()[k],d.cell_id.nunique(),int(e._N)
# ---- 2009 sweep
df=raw.copy()
df["pol09"]=((df.year>2008)&~((df.year==2009)&(df.quarter<3))).astype(int)
td,tc="ln_mean_length_2009_jan_jun_20km","ln_mean_count_2009_jan_jun_20km"
df=df.dropna(subset=["biomass_sum","area_sum","model_2008",td,tc,"ln_sst","ln_precipitation","ln_swh","ln_sum_count_2009_jan_jun_20km"])
full=df[df.ln_sum_count_2009_jan_jun_20km>0]
print("2009 SWEEP on distance percentile (as in figure); biomass, treat=distance | treat=count")
ct=full.groupby("cell_id")[td].first()
for p in range(0,55,5):
    d=full[full.cell_id.isin(ct[ct>=ct.quantile(p/100)].index)]
    a=run(d,"biomass_sum",td,"pol09"); b=run(d,"biomass_sum",tc,"pol09")
    print(f"P{p:<2} cells={a[3]:>3}  dist {a[0]:+.4f} (se {a[1]:.4f}, p={a[2]:.3f})   count {b[0]:+.4f} (p={b[2]:.3f})")
print("\n2009 SWEEP on vessel-count threshold (vessels/month); biomass")
cc=full.groupby("cell_id")[tc].first()
for thr in [0,0.5,1,1.5,2,3,4,5,7,10]:
    d=full[full.cell_id.isin(cc[cc>=np.log(thr) if thr>0 else cc>-np.inf].index)]
    a=run(d,"biomass_sum",td,"pol09"); b=run(d,"biomass_sum",tc,"pol09")
    print(f">={thr:<4} cells={a[3]:>3}  dist {a[0]:+.4f} (p={a[2]:.3f})   count {b[0]:+.4f} (p={b[2]:.3f})")
# ---- 2011
d11=raw.copy(); d11["pol11"]=(d11.year>2011).astype(int)
d11=d11.dropna(subset=["ln_mean_count_2011_20km"])
print("\n2011 biomass")
for lab,d in [("unweighted, main sample",d11),("unweighted, weighted-table sample",d11[d11.ln_sum_count_2011_20km>0])]:
    for t in ["ln_mean_length_2011_20km","ln_mean_count_2011_20km"]:
        r=run(d,"biomass_sum",t,"pol11"); print(f"{lab:36s} {t:28s} {r[0]:+.4f} (se {r[1]:.4f}) cells={r[3]} N={r[4]}")
