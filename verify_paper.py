import sys
sys.stdout.reconfigure(encoding='utf-8')

def check(name, mr, at, isat, tol=0.002):
    calc = (at + isat) / 2
    ok = abs(calc - mr) < tol
    status = "OK" if ok else "FAIL"
    print(f"  {name:15s}: MR={mr}, calc={calc:.4f}  [{status}]")
    return ok

print("=== TABLE 3 (Main Results) ===")
check("Majority", 0.333, 0.333, 0.333)
check("mBERT", 0.427, 0.354, 0.500)
check("hmBERT cal", 0.553, 0.450, 0.655)
check("XLM-R cal", 0.545, 0.447, 0.643)
check("MHIPEX", 0.566, 0.459, 0.672)

print("\n=== TABLE KG ===")
check("Text-only", 0.5527, 0.4555, 0.6499)
check("Text+Wiki", 0.5529, 0.4713, 0.6344)
at_d = (0.4713 - 0.4555) * 100
is_d = (0.6344 - 0.6499) * 100
print(f"  at delta: +{at_d:.2f}%")
print(f"  isAt delta: {is_d:.2f}%")
print(f"  Paper claims +3.5%: actual = +{at_d:.1f}%")

print("\n=== RELATIVE IMPROVEMENT ===")
rel = (0.566 - 0.427) / 0.427 * 100
print(f"  MHIPEX vs mBERT: {rel:.1f}% (paper claims 32.5%)")

print("\n=== CROSSVAL ===")
check("Reference", 0.553, 0.450, 0.655, tol=0.003)
check("C1 Domain", 0.588, 0.482, 0.695, tol=0.003)
check("C2 EN", 0.531, 0.412, 0.649, tol=0.003)
check("C3 FR", 0.490, 0.389, 0.590, tol=0.003)
check("C4 DE", 0.491, 0.426, 0.557, tol=0.003)

print("\n=== ABSTRACT TAG CHECK ===")
tex = open(r"d:\my work\MHIPEX\paper\main.tex", encoding="utf-8").read()
lines = tex.split("\n")
for i, l in enumerate(lines):
    if "abstract" in l.lower() and i < 30:
        print(f"  Line {i+1}: {l.strip()[:100]}")

print("\n=== FIGURE REFERENCES ===")
import re
figs = re.findall(r"\\includegraphics.*?\{(.*?)\}", tex)
refs = re.findall(r"\\ref\{(.*?)\}", tex)
labels = re.findall(r"\\label\{(.*?)\}", tex)
print(f"  Figures used: {figs}")
print(f"  Labels defined: {len(labels)}")
unrefs = [l for l in labels if l not in " ".join(refs)]
if unrefs:
    print(f"  WARNING: unreferenced labels: {unrefs}")
else:
    print(f"  All labels referenced: OK")

print("\n=== CITATION CHECK ===")
cites = set(re.findall(r"\\cite\{(.*?)\}", tex))
bibs = set(re.findall(r"\\bibitem\{(.*?)\}", tex))
all_cited = set()
for c in cites:
    all_cited.update(c.split(","))
uncited = bibs - all_cited
unused_cites = all_cited - bibs
if uncited:
    print(f"  WARNING: defined but never cited: {uncited}")
else:
    print(f"  All bibitems cited: OK")
if unused_cites:
    print(f"  WARNING: cited but not in bibliography: {unused_cites}")
