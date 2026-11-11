"""Build paper/Kelp_anonymous.tex from paper/Kelp.tex: same text, no author names, affiliations, thanks or funding."""
import io
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
s = io.open(src, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"

s, n = re.subn(r"[ \t]*pdfauthor=\{[^}]*\},\r?\n", "", s)
assert n == 1, n

i = s.index("\\author{")
j = s.index("\\maketitle")
assert 0 < i < j
s = s[:i] + "\\author{}" + nl + s[j:]

for w in ("Candau", "Lafferrere", "UPPA", "univ-pau", "Aquitaine", "PORTeur", "thank"):
    assert w not in s, w

io.open(dst, "w", encoding="utf-8", newline="").write(s)
print("written", dst)
