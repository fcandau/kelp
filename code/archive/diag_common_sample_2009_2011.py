"""Diagnostic (2026-11-13, CJE referee minor point 2): is there a common sample on which the 2009 and the
2011 results both hold? Read-only: prints coefficients, writes nothing.

For each sample definition, biomass PPML with cell and year-quarter FE, cluster by cell:
  (a) 2009 design: exposure 2009 (Jan-Jun) x post-2009Q3
  (b) 2011 design: exposure 2011 x post-2011
  (c) joint: both interactions in one regression
Samples: thresholds on the 2009 count, thresholds on the 2011 count, thresholds on both.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyfixest as pf

raw = pd.read_csv(r"c:\Users\fcanda01\Desktop\Recherche\kelp\data\generated\new_model_kelp_exposure_3.csv", low_memory=False)
raw = raw[(raw.year <= 2016) & (raw.latitude >= 32) & (raw.latitude <= 36)].copy()
raw["cell_id"] = raw.eca_index_official.astype(str) + "_" + raw.year_eca.astype(str)
raw["yq"] = raw.year.astype(str) + "_" + raw.quarter.astype(str)
raw = raw.drop_duplicates(["cell_id", "yq"])
for c in ["sst", "precipitation", "swh"]:
    raw["ln_" + c] = raw[f"ln_{c}_mean"]
raw["post09"] = ((raw.year > 2009) | ((raw.year == 2009) & (raw.quarter >= 3))).astype(int)
raw["post11"] = (raw.year > 2011).astype(int)

C09, L09 = "ln_mean_count_2009_jan_jun_20km", "ln_mean_length_2009_jan_jun_20km"
C11, L11 = "ln_mean_count_2011_20km", "ln_mean_length_2011_20km"
CTRL = "ln_sst + ln_precipitation + ln_swh"
V = {"CRV1": "cell_id"}
base = raw.dropna(subset=["ln_sst", "ln_precipitation", "ln_swh", "biomass_sum"]).copy()
c09 = base.groupby("cell_id")[C09].first()
c11 = base.groupby("cell_id")[C11].first()
print("cells with controls:", base.cell_id.nunique(), "| with 2009 exposure:", c09.notna().sum(),
      "| with 2011 exposure:", c11.notna().sum(), "| with both:", (c09.notna() & c11.notna()).sum())


def fit(dd, rhs, keys):
    try:
        e = pf.fepois(f"biomass_sum ~ {rhs} + {CTRL} | cell_id + yq", data=dd, vcov=V)
        out = []
        for k in keys:
            name = [x for x in e.coef().index if k in x][0]
            b, p = e.coef()[name], e.pvalue()[name]
            star = "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
            out.append(f"{b:+.3f}{star:3s}")
        return " ".join(out)
    except Exception as ex:
        return "ERR " + str(ex)[:30]


def row(label, cells):
    dd = base[base.cell_id.isin(cells)]
    d9 = dd.dropna(subset=[C09, L09])
    d11 = dd.dropna(subset=[C11, L11])
    dj = dd.dropna(subset=[C09, L09, C11, L11])
    print(f"{label:34s} n09={d9.cell_id.nunique():3d} n11={d11.cell_id.nunique():3d} | "
          f"2009 dist {fit(d9, L09 + ':post09', [L09])} cnt {fit(d9, C09 + ':post09', [C09])} | "
          f"2011 dist {fit(d11, L11 + ':post11', [L11])} cnt {fit(d11, C11 + ':post11', [C11])} | "
          f"joint dist(09,11) {fit(dj, L09 + ':post09 + ' + L11 + ':post11', [L09, L11])} "
          f"cnt(09,11) {fit(dj, C09 + ':post09 + ' + C11 + ':post11', [C09, C11])}")


allc = set(base.cell_id.unique())
print("\n--- all cells available for each design ---")
row("all cells", allc)
print("\n--- threshold on 2009 count (vessels/month) ---")
for t in [0.5, 1, 1.5, 2, 2.5, 3, 4]:
    row(f"count2009 >= {t}", set(c09[c09 >= np.log(t)].index))
print("\n--- threshold on 2011 count ---")
for t in [0.5, 1, 1.5, 2, 2.5, 3, 4]:
    row(f"count2011 >= {t}", set(c11[c11 >= np.log(t)].index))
print("\n--- threshold on both counts ---")
for t in [1, 1.5, 2, 3]:
    row(f"both >= {t}", set(c09[c09 >= np.log(t)].index) & set(c11[c11 >= np.log(t)].index))
print("\n--- threshold on either count ---")
for t in [1, 1.5, 2, 3]:
    row(f"either >= {t}", set(c09[c09 >= np.log(t)].index) | set(c11[c11 >= np.log(t)].index))
print("\n--- distribution of counts (vessels/month) ---")
print("2009:", np.exp(c09.dropna()).describe().round(2).to_dict())
print("2011:", np.exp(c11.dropna()).describe().round(2).to_dict())
