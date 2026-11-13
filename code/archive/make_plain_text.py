"""Plain-text version of paper/Kelp.tex for pasting into a text checker: prose only, one paragraph per line.

Usage, from paper/:
  quarto pandoc Kelp.tex -f latex -t plain --wrap=none --citeproc --bibliography ECA.bib -o raw.txt
  python ../code/archive/make_plain_text.py raw.txt Kelp.tex Kelp_plain.txt
Tables, table and figure notes, equations, footnotes and the reference list are left out.
"""
import io
import re
import sys

src, tex, dst = sys.argv[1], sys.argv[2], sys.argv[3]
L = io.open(src, encoding="utf-8").read().split("\n")

HEADS = ["Introduction", "Background", "The 2009 Ocean-Going Vessel Fuel Rule", "Data", "Specification",
         "Results", "Dynamic effects", "Threats to identification",
         "Boundary Modification of the Regulated Zone in 2011", "Spatial redistribution of traffic after 2011",
         "Empirical strategy", "Conclusion", "Pollution along shipping routes",
         "Robustness of the 2009 Estimates", "Balance Table"]
end = next(i for i, l in enumerate(L) if l.startswith("Abman, Ryan"))   # start of the reference list
L = L[:end]


def is_prose(l):
    s = l.strip()
    if s in HEADS:
        return True
    if len(s) < 200:                     # table rows, captions, equation fragments, short labels
        return False
    if s.startswith(("Notes:", "Note:", "This figure", "Unweighted PPML", "PPML estimates", "Cell-quarter")):
        return False                     # table and figure notes
    return s.count(". ") >= 1


def clean(s):
    s = re.sub(r"\[\d+\]", "", s)                                   # footnote marks
    s = s.replace("); (", "; ")                                     # (A 2020); (B 2021) -> (A 2020; B 2021)
    s = re.sub(r"\(\(([A-Z])", lambda m: "(" + m.group(1), s)       # ((Dayton 1985)) -> (Dayton 1985)
    s = re.sub(r"(\d{4}[a-z]?)\)\)", lambda m: m.group(1) + ")", s)
    s = s.replace("([eq:cdid])", "(1)").replace("([eq:cdid2011])", "(2)")
    s = re.sub(r"\(e\.g\., \(([^()]*)\)", lambda m: "(e.g., " + m.group(1) + ")", s)
    s = re.sub(r"\((see also|see) \(([^()]*)\)", lambda m: "(" + m.group(1) + " " + m.group(2) + ")", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


out = []
# abstract (pandoc stores it as metadata, so take it from the source)
t = io.open(tex, encoding="utf-8").read()
k = t.index("\\begin{abstract}")
a = t[k + len("\\begin{abstract}"):t.index("\\medskip", k)]
a = re.sub(r"\s+", " ", a.replace("\\%", "%").replace("--", "–")).strip()
a = a.replace(" JEL: Q57, R41, F18", "")
out += ["=== ABSTRACT ===", "", a, ""]

for l in L:
    if not is_prose(l):
        continue
    s = clean(l)
    if s in HEADS:
        out += ["", "=== " + s.upper() + " ===", ""]
    else:
        out += [s, ""]

io.open(dst, "w", encoding="utf-8", newline="\r\n").write("\n".join(out))
print("paragraphs:", sum(1 for x in out if len(x) > 200))
