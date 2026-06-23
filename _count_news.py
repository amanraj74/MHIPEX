import json, sys
sys.stdout.reconfigure(encoding='utf-8')
print("=== SANDBOX ===")
for split in ['train','dev']:
    for lang in ['en','fr','de']:
        f = f'data/raw/hipe2026/data/sandbox/{lang}-{split}.jsonl'
        pairs = sum(len(json.loads(l)['sampled_pairs']) for l in open(f, encoding='utf-8'))
        print(f'  {lang}-{split}: {pairs} pairs')

print("\n=== NEWSPAPER v1.0 ===")
for lang in ['en','fr','de']:
    f = f'data/raw/hipe2026/data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-{lang}.jsonl'
    pairs = sum(len(json.loads(l)['sampled_pairs']) for l in open(f, encoding='utf-8'))
    at_dist = {'FALSE':0,'PROBABLE':0,'TRUE':0}
    isat_dist = {'FALSE':0,'TRUE':0}
    for line in open(f, encoding='utf-8'):
        for p in json.loads(line)['sampled_pairs']:
            at_dist[p['at']] += 1
            isat_dist[p['isAt']] += 1
    print(f'  {lang}: {pairs} pairs | at={at_dist} | isAt={isat_dist}')
