import requests, sys, os, base64, glob
sys.stdout.reconfigure(encoding="utf-8")

TEX_FILE = r"d:\my work\MHIPEX\paper\main.tex"
FIG_DIR  = r"d:\my work\MHIPEX\paper\figures"
OUT_PDF  = r"d:\my work\MHIPEX\paper\MHIPEX_Paper_Draft.pdf"

print("Compiling LaTeX to PDF (with ALL images)...")

with open(TEX_FILE, "r", encoding="utf-8") as f:
    tex_content = f.read()

# Build resources list
resources = [{"main": True, "content": tex_content}]

# Add ALL images from figures/ — exclude stale architecture.png (superseded by architecture1.png)
EXCLUDE_FIGURES = {"architecture.png", "fig5_ablation.png"}
for img_path in glob.glob(os.path.join(FIG_DIR, "*.png")):
    fname = os.path.basename(img_path)
    if fname in EXCLUDE_FIGURES:
        print(f"  Skipped (stale): figures/{fname}")
        continue
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("ascii")
    resources.append({"path": f"figures/{fname}", "file": img_b64})
    print(f"  Added: figures/{fname} ({os.path.getsize(img_path)//1024} KB)")

API_URL = "https://latex.ytotech.com/builds/sync"
payload = {"compiler": "pdflatex", "resources": resources}

print(f"  Sending {len(resources)} files to compilation service...")
resp = requests.post(API_URL, json=payload, timeout=300)
print(f"  Status: {resp.status_code}")
ct = resp.headers.get("Content-Type", "")
print(f"  Content-Type: {ct}")

if resp.content[:5] == b'%PDF-':
    with open(OUT_PDF, "wb") as f:
        f.write(resp.content)
    size_kb = os.path.getsize(OUT_PDF) / 1024
    print(f"\nPDF generated successfully!")
    print(f"  File: {OUT_PDF}")
    print(f"  Size: {size_kb:.0f} KB")
else:
    print(f"\nCompilation failed. Response preview:")
    print(resp.text[:2000])
