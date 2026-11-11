"""
Assemble the consolidated robustness table for the 2009 specification.

No estimation is done here. The script collects coefficients that the
pipeline has already produced and writes them as one table, one row per
check:

  output/csv/density_dependence_tests.csv    (robust_density_dependence.py)
  output/csv/robust_prewindow.csv            (robust_spatial_prewindow.py)
  output/tables/crisis_robustness.tex        (5_financial_crisis.py)
  output/tables/weighting_robustness.tex     (2_e_weighting_robustness.py)
  output/tables/placebo_test.tex             (2_a_estim.py)

  -> output/tables/tab_robustness_2009.tex

Columns: Distance x Biomass, Distance x Area, Count x Biomass, Count x Area.
Used by paper/Kelp.tex.
"""
import os
import re
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "output", "csv")
TAB = os.path.join(ROOT, "output", "tables")

DIST = "ln_mean_length_2009_jan_jun_20km"
COUNT = "ln_mean_count_2009_jan_jun_20km"
COLS = [("biomass_sum", DIST), ("area_sum", DIST), ("biomass_sum", COUNT), ("area_sum", COUNT)]


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def cell(coef, p):
    return f"{coef:.3f}\\sym{{{stars(p)}}}"


# ---- rows taken from CSV files ---------------------------------------------
dens = pd.read_csv(os.path.join(CSV, "density_dependence_tests.csv"))


def density_row(test):
    coefs, ses, n = [], [], None
    for outcome, treat in COLS:
        r = dens[(dens.test == test) & (dens.outcome == outcome) & (dens.treatment == treat)].iloc[0]
        coefs.append(cell(r.coef, r.pvalue))
        ses.append(f"({r.se:.3f})")
        n = int(r.nobs)
    return coefs, ses, n


pre = pd.read_csv(os.path.join(CSV, "robust_prewindow.csv"))


def prewindow_row():
    coefs, ses = [], []
    for outcome, treat in [("Biomass", "Distance"), ("Area", "Distance"), ("Biomass", "Count"), ("Area", "Count")]:
        r = pre[(pre.outcome == outcome) & (pre.treatment == treat)].iloc[0]
        coefs.append(cell(r.coef_pre2011, r.p_pre2011))
        ses.append(f"({r.se_pre2011:.3f})")
    return coefs, ses, int(pre.n_pre2011.iloc[0])


# ---- rows parsed from existing LaTeX tables --------------------------------
NUM = r"(-?\d+\.\d+)"


def tex_row(fname, label_start, occurrence=1):
    """Return (coefs, ses, n) for the `occurrence`-th row starting with label_start."""
    with open(os.path.join(TAB, fname), encoding="utf-8") as f:
        lines = f.read().splitlines()
    hits = [i for i, l in enumerate(lines) if l.strip().startswith(label_start)]
    i = hits[occurrence - 1]
    cells = lines[i].split("&")[1:5]
    coefs = []
    for c in cells:
        val = float(re.search(NUM, c).group(1))
        st = re.search(r"\\sym\{(\**)\}", c)
        coefs.append(f"{val:.3f}\\sym{{{st.group(1) if st else ''}}}")
    ses = [f"({float(x):.3f})" for x in re.findall(r"\(" + NUM + r"\)", lines[i + 1])[:4]]
    n = None
    for l in lines[i + 1:]:
        if l.strip().startswith("Observations"):
            n = int(re.search(r"(\d+)", l.split("&")[1]).group(1))
            break
    assert len(coefs) == 4 and len(ses) == 4 and n is not None, (fname, label_start, occurrence)
    return coefs, ses, n


EXPO = r"Post 2009 $\times$ Exposure"
rows = [
    ("panel", r"\emph{Panel A: Main specification}"),
    ("Baseline", density_row("baseline")),
    ("panel", r"\emph{Panel B: Recovery of depleted kelp stands}"),
    (r"+ ln(baseline kelp) $\times$ post", density_row("test1_base_x_post")),
    (r"+ baseline-kelp quartile $\times$ post", density_row("test2_quartile_post")),
    (r"+ ln(baseline kelp) $\times$ trend", density_row("test3a_base_x_trend")),
    (r"+ ln(baseline kelp) $\times$ year effects", density_row("test3b_base_x_yearFE")),
    ("High-baseline cells only", density_row("test4_highbase")),
    ("Low-baseline cells only", density_row("test4_lowbase")),
    ("panel", r"\emph{Panel C: Financial crisis and 2011 boundary modification}"),
    ("+ port activity (NBIC arrivals)", tex_row("crisis_robustness.tex", EXPO, 2)),
    ("Pre-2011 window (2005--2011)", prewindow_row()),
    ("panel", r"\emph{Panel D: Weighting}"),
    ("Weighted by log vessel count", tex_row("weighting_robustness.tex", EXPO, 2)),
    ("Weighted by pre-period kelp biomass", tex_row("weighting_robustness.tex", EXPO, 3)),
    ("panel", r"\emph{Panel E: Placebo}"),
    ("Fake treatment date 2007Q3 (2005--2008)", tex_row("placebo_test.tex", "Fake Post 2007Q3", 1)),
]

out = [
    r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}",
    r"\begin{tabular}{l*{4}{c}c}",
    r"\hline\hline",
    r" &\multicolumn{2}{c}{Distance Traveled}&\multicolumn{2}{c}{Vessel Count}&\\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
    r" &Biomass&Area&Biomass&Area&Obs.\\",
    r"\hline",
]
first = True
for label, content in rows:
    if label == "panel":
        if not first:
            out.append(r"[0.4em]")
        out.append(rf"\multicolumn{{6}}{{l}}{{{content}}}\\")
        first = False
        continue
    coefs, ses, n = content
    out.append(f"{label} & " + " & ".join(coefs) + f" & {n}\\\\")
    out.append(" & " + " & ".join(ses) + r" & \\")
out += [r"\hline\hline", r"\end{tabular}"]

path = os.path.join(TAB, "tab_robustness_2009.tex")
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
print(f"\nSaved: {path}")
