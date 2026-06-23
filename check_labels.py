import json, sys
sys.stdout.reconfigure(encoding='utf-8')

at_c = {'FALSE':0, 'PROBABLE':0, 'TRUE':0}
isat_c = {'FALSE':0, 'TRUE':0}

for lang in ['en','fr','de']:
    for split in ['train','dev']:
        path = f'data/raw/hipe2026/data/sandbox/{lang}-{split}.jsonl'
        for line in open(path, 'r', encoding='utf-8'):
            doc = json.loads(line)
            for p in doc.get('sampled_pairs', []):
                at_c[p['at']] = at_c.get(p['at'], 0) + 1
                isat_c[p['isAt']] = isat_c.get(p['isAt'], 0) + 1

print("=== LABEL DISTRIBUTION (ALL LANGUAGES, TRAIN+DEV) ===")
total_at = sum(at_c.values())
total_isat = sum(isat_c.values())
print(f"\nAT labels: {at_c}")
print(f"  FALSE:    {at_c['FALSE']:>5} ({at_c['FALSE']/total_at*100:.1f}%)")
print(f"  PROBABLE: {at_c['PROBABLE']:>5} ({at_c['PROBABLE']/total_at*100:.1f}%)")
print(f"  TRUE:     {at_c['TRUE']:>5} ({at_c['TRUE']/total_at*100:.1f}%)")
print(f"\nisAt labels: {isat_c}")
print(f"  FALSE:    {isat_c['FALSE']:>5} ({isat_c['FALSE']/total_isat*100:.1f}%)")
print(f"  TRUE:     {isat_c['TRUE']:>5} ({isat_c['TRUE']/total_isat*100:.1f}%)")
