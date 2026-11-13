"""
event_study_2011_ppml.py -- PPML event study for the 2011 boundary modification,
correctly dated to the modification (post = 2012, since it took effect December
2011) and estimated in PPML to match the main 2011 table.

Design.  Continuous DiD in PPML: annual year dummies interacted with the
time-invariant 2011 pre-modification vessel-count exposure, cell + year-quarter FE,
SEs clustered at the cell level, on the 127-cell 2011 sample (as in 3_estim2011.do).
The modification took effect December 2011, so 2011 is the omitted reference year
(ref=2011): 2005-2010 are pre-trends, 2012-2016 are post-modification effects.

Why a table, not a figure.  The concurrent 2009 fuel rule lifted kelp persistently
in high-2009-exposure cells, and 2009 & 2011 exposure correlate r~0.8, so the
2009-2011 window carries the 2009-rule dynamics, not a 2011 pre-trend. A figure
makes the 2009-2010 coefficients look like a pre-trend; a table lets us LABEL the
three periods (pre-2009 parallel-trends window / 2009 fuel-rule phase-in /
post-2011 modification) so they cannot be misread. The parallel-trends check is the
joint Wald test on the clean pre-both-regulations window (2005-2008).

We also print, for the record, every spec (count & distance, with & without a
2009-rule control) so the choice of the count / no-control spec is transparent:
 - the 2009 control BACKFIRES (pushes 2005-2008 up), so we do not use it;
 - distance is directionally consistent but insignificant year-by-year (less power).

Outputs:
  output/tables/tab_es_2011.tex   (the table that goes in the paper)
  output/csv/event_study_2011_ppml.csv
"""

import os
import re
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # code/archive -> project root
GENERATED = os.path.join(BASE, "data", "generated")
TABLES_DIR = os.path.join(BASE, "output", "tables")
CSV_DIR = os.path.join(BASE, "output", "csv")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

VCOV = {"CRV1": "cell_id"}
REF_YEAR = 2011
EXP2011 = {"dist": "ln_mean_length_2011_20km", "count": "ln_mean_count_2011_20km"}
EXP2009 = {"dist": "ln_mean_length_2009_jan_jun_20km", "count": "ln_mean_count_2009_jan_jun_20km"}


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _year_of(k):
    # anchor on the factor-level bracket "[2005" -- NOT base=2011 in the coef name
    m = re.search(r"\[(20\d{2})", k)
    return int(m.group(1)) if m else None


# ---- Load + 2011 sample (match 3_estim2011.do) ----
df = pd.read_csv(os.path.join(GENERATED, "new_model_kelp_exposure_3.csv"), low_memory=False)
df = df[(df["year"] <= 2016) & (df["latitude"] >= 32) & (df["latitude"] <= 36)].copy()
df["cell_id"] = df["eca_index_official"].astype(str) + "_" + df["year_eca"].astype(str)
df["yq"] = df["year"].astype(str) + "_" + df["quarter"].astype(str)
df = df.drop_duplicates(subset=["cell_id", "yq"])
df["ln_sst"] = df["ln_sst_mean"]
df["ln_precipitation"] = df["ln_precipitation_mean"]
df["ln_swh"] = df["ln_swh_mean"]
df["post2009"] = (df["year"] >= 2009).astype(int)
for t in ("dist", "count"):
    df[f"exp2009post_{t}"] = df[EXP2009[t]] * df["post2009"]

need = (["biomass_sum", "area_sum", "ln_sst", "ln_precipitation", "ln_swh"]
        + list(EXP2011.values()) + list(EXP2009.values()))
df = df.dropna(subset=need)
print(f"[2011 sample] {len(df)} obs, {df['cell_id'].nunique()} cells "
      f"(main-table 2011 spec = 127 cells / 6000 obs)")


def event_study(depvar, t, control_2009):
    ctrl = f" + exp2009post_{t}" if control_2009 else ""
    fml = (f"{depvar} ~ i(year, {EXP2011[t]}, ref={REF_YEAR}){ctrl}"
           f" + ln_sst + ln_precipitation + ln_swh | cell_id + yq")
    return pf.fepois(fml, data=df, vcov=VCOV)


def parse_years(est, exp):
    rows = []
    for k in est.coef().index:
        if exp in k:
            yr = _year_of(k)
            if yr is not None:
                rows.append({"year": yr, "coef": est.coef()[k], "se": est.se()[k],
                             "p": est.pvalue()[k]})
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def pretrend_wald(est, exp, lo, hi):
    names = list(est.coef().index)
    idx = [i for i, k in enumerate(names)
           if exp in k and _year_of(k) is not None and lo <= _year_of(k) <= hi]
    if not idx:
        return np.nan, 0
    beta = est.coef().values[idx]
    V = est._vcov[np.ix_(idx, idx)]
    chi2 = float(beta @ np.linalg.pinv(V) @ beta)
    return 1 - scipy_stats.chi2.cdf(chi2, len(idx)), len(idx)


# ---- Diagnostic printout: all specs, with & without 2009 control (record only) ----
SPECS = [("biomass_sum", "count", "Biomass x count"),
         ("area_sum", "count", "Area x count"),
         ("biomass_sum", "dist", "Biomass x dist"),
         ("area_sum", "dist", "Area x dist")]
ests = {}
all_coefs = []
for depvar, t, lab in SPECS:
    for control_2009 in (False, True):
        est = event_study(depvar, t, control_2009)
        ests[(depvar, t, control_2009)] = est
        yrs = parse_years(est, EXP2011[t])
        p_clean, _ = pretrend_wald(est, EXP2011[t], 2005, 2008)
        p_full, _ = pretrend_wald(est, EXP2011[t], 2005, 2010)
        post = yrs[yrs.year >= 2012]
        pns = int(((post["coef"] < 0) & (post["p"] < 0.10)).sum())
        tag = "with 2009 ctrl" if control_2009 else "no 2009 ctrl "
        print(f"  {lab} [{tag}]: pre05-08 p={p_clean:.3f}, full05-10 p={p_full:.3f}, "
              f"post neg&sig {pns}/5")
        if not control_2009:
            yrs["outcome"], yrs["treatment"] = depvar, t
            all_coefs.append(yrs)
pd.concat(all_coefs, ignore_index=True).to_csv(
    os.path.join(CSV_DIR, "event_study_2011_ppml.csv"), index=False)


# ---- LaTeX table: count spec (no 2009 control), biomass + area, grouped by period ----
bio = parse_years(ests[("biomass_sum", "count", False)], EXP2011["count"]).set_index("year")
area = parse_years(ests[("area_sum", "count", False)], EXP2011["count"]).set_index("year")
p_bio, _ = pretrend_wald(ests[("biomass_sum", "count", False)], EXP2011["count"], 2005, 2008)
p_area, _ = pretrend_wald(ests[("area_sum", "count", False)], EXP2011["count"], 2005, 2008)
N_bio = ests[("biomass_sum", "count", False)]._N
N_area = ests[("area_sum", "count", False)]._N


def row(year):
    rb, ra = bio.loc[year], area.loc[year]
    return (f"{year} &{rb.coef:.3f}\\sym{{{stars(rb.p)}}}&{ra.coef:.3f}\\sym{{{stars(ra.p)}}}\\\\\n"
            f" &({rb.se:.3f})&({ra.se:.3f})\\\\")


L = [r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}",
     r"\begin{tabular}{l*{2}{c}}",
     r"\hline\hline",
     r" &\multicolumn{1}{c}{Biomass}&\multicolumn{1}{c}{Area}\\",
     r"\hline",
     r"\multicolumn{3}{l}{\emph{Pre-2009 window (parallel-trends check)}}\\"]
for y in (2005, 2006, 2007, 2008):
    L.append(row(y))
L.append(f"\\quad Joint 2005--2008 test ($p$-value) &{p_bio:.2f}&{p_area:.2f}\\\\")
L.append(r"[0.4em]")
L.append(r"\multicolumn{3}{l}{\emph{2009 fuel-rule phase-in}}\\")
for y in (2009, 2010):
    L.append(row(y))
L.append(r"\quad 2011 (reference) &---&---\\")
L.append(r"[0.4em]")
L.append(r"\multicolumn{3}{l}{\emph{Post-2011 boundary modification}}\\")
for y in (2012, 2013, 2014, 2015, 2016):
    L.append(row(y))
L.append(r"\hline")
L.append(r"Cell FE &Yes&Yes\\")
L.append(r"Year-Quarter FE &Yes&Yes\\")
L.append(f"Observations &{N_bio}&{N_area}\\\\")
L.append(r"\hline\hline")
L.append(r"\end{tabular}")

out = os.path.join(TABLES_DIR, "tab_es_2011.tex")
with open(out, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"\nSaved {out}")
print(f"  pre-2009 joint test: biomass p={p_bio:.2f}, area p={p_area:.2f} (flat = good)")

# ---- Same estimates, three stacked panels: years in columns, one line per outcome ----
# (used by paper/Kelp.tex; fixed effects and observations go in the note)
NCOL = 5  # year columns per panel (the longest panel, 2012-2016, has five)


def panel(title, years, extra_head=None, extra=None):
    """One panel: title, year header, then coefficient and SE lines for biomass and area."""
    pad = NCOL - len(years) - (1 if extra_head else 0)
    head = [str(y) for y in years] + ([extra_head] if extra_head else []) + [""] * pad
    used = len(years) + (1 if extra_head else 0)
    lines = [rf"\multicolumn{{{NCOL + 1}}}{{l}}{{\emph{{{title}}}}}\\[0.3em]",
             " & " + " & ".join(head) + r"\\",
             rf"\cmidrule(lr){{2-{used + 1}}}"]
    for label, d, p_joint in (("Biomass", bio, p_bio), ("Area", area, p_area)):
        coefs = [f"{d.loc[y].coef:.3f}\\sym{{{stars(d.loc[y].p)}}}" for y in years]
        ses = [f"({d.loc[y].se:.3f})" for y in years]
        if extra_head:
            coefs.append(f"{p_joint:.2f}")
            ses.append("")
        coefs += [""] * pad
        ses += [""] * pad
        lines.append(f"{label} & " + " & ".join(coefs) + r"\\")
        lines.append(" & " + " & ".join(ses) + r"\\")
    return lines


W = [r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}",
     rf"\begin{{tabular}}{{l*{{{NCOL}}}{{c}}}}",
     r"\hline\hline"]
W += panel("Panel A: Before both policies", [2005, 2006, 2007, 2008],
           extra_head=r"Joint test ($p$)")
W.append(r"[0.6em]")
W += panel("Panel B: 2009 fuel rule", [2009, 2010])
W.append(r"[0.6em]")
W += panel("Panel C: After the 2011 modification", [2012, 2013, 2014, 2015, 2016])
W += [r"\hline\hline", r"\end{tabular}"]
out_wide = os.path.join(TABLES_DIR, "tab_es2011_panels.tex")
with open(out_wide, "w") as f:
    f.write("\n".join(W) + "\n")
print(f"Saved {out_wide}  (N = {N_bio}; joint 2005-2008 p = {p_bio:.2f} / {p_area:.2f})")
print("DONE.")
