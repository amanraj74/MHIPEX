import sys, json
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('hipe-2026-dr-sarika-jain-nit-new.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']
print(f"Total cells: {len(cells)}")

for i, c in enumerate(cells):
    ct = c['cell_type']
    src = ''.join(c['source'])
    
    # For code cells, also grab outputs
    outputs_text = ''
    if ct == 'code' and c.get('outputs'):
        for out in c['outputs']:
            if out.get('text'):
                outputs_text += ''.join(out['text'])
            elif out.get('data', {}).get('text/plain'):
                outputs_text += ''.join(out['data']['text/plain'])
    
    # Print cell header
    print(f"\n{'='*60}")
    print(f"CELL {i+1}/{len(cells)} [{ct.upper()}]")
    print(f"{'='*60}")
    
    # Print source (truncated)
    if len(src) > 2000:
        print(src[:2000])
        print(f"\n... [TRUNCATED - {len(src)} chars total]")
    else:
        print(src)
    
    # Print outputs (truncated)
    if outputs_text:
        print(f"\n--- OUTPUT ---")
        if len(outputs_text) > 1500:
            print(outputs_text[:1500])
            print(f"\n... [OUTPUT TRUNCATED - {len(outputs_text)} chars total]")
        else:
            print(outputs_text)
