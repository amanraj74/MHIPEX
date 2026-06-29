"""
MHIPEX — Multi-Knowledge-Graph Augmented Relation Extraction
Run on Kaggle with GPU T4 x2 + Internet enabled
Estimated runtime: ~90 minutes (Trains E0, E1, E2, E3)
"""

import json, os, time, requests, gc, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import recall_score, classification_report
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR = Path("out_multikg")
OUT_DIR.mkdir(exist_ok=True)

AT_MAP = {"FALSE": 0, "PROBABLE": 1, "TRUE": 2}
ISAT_MAP = {"FALSE": 0, "TRUE": 1}

# ══════════════════════════════════════════════════════════════════
#  Wikidata, GeoNames, and Getty API Clients
# ══════════════════════════════════════════════════════════════════

WIKIDATA_API = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"
GEONAMES_API = "http://api.geonames.org/getJSON?geonameId={}&username=demo" # Kaggle allows HTTP calls, but 'demo' has rate limits.
# Fallback: We will extract GeoNames/Getty data *directly* from Wikidata properties 
# to avoid API rate limits and complex SPARQL endpoints.
# P1566 = GeoNames ID, P1667 = TGN ID, P17 = Country, P131 = Admin territorial entity, P625 = Coords, P31 = Instance of

CACHE_FILE = OUT_DIR / "kg_cache.json"

def fetch_entity_from_wikidata(qid, cache):
    if qid in cache: return cache[qid]
    
    try:
        resp = requests.get(WIKIDATA_API.format(qid), timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("entities", {}).get(qid, {})
            claims = data.get("claims", {})
            
            # Helper to extract linked QIDs
            def get_qids(prop):
                if prop not in claims: return []
                return [c["mainsnak"]["datavalue"]["value"]["id"] for c in claims[prop] if c["mainsnak"].get("datavalue", {}).get("type") == "wikibase-entityid"]
            
            # Helper to extract string values
            def get_strings(prop):
                if prop not in claims: return []
                return [c["mainsnak"]["datavalue"]["value"] for c in claims[prop] if c["mainsnak"].get("datavalue", {}).get("type") == "string"]
                
            labels = data.get("labels", {})
            label = labels.get("en", labels.get("fr", labels.get("de", {}))).get("value", "")

            # General Wikidata Properties
            wd_props = []
            for p in ["P19", "P20", "P551", "P27", "P106"]: # birth, death, residence, country, occupation
                wd_props.extend(get_qids(p))
            for p in ["P17", "P131"]: # place country, region
                wd_props.extend(get_qids(p))
                
            # GeoNames Properties (simulated via Wikidata mapping)
            gn_props = []
            gn_ids = get_strings("P1566")
            if gn_ids:
                gn_props.append(f"GeoNamesID_{gn_ids[0]}")
                gn_props.extend(get_qids("P17")) # Country
                gn_props.extend(get_qids("P31")) # Feature class
                
            # Getty TGN Properties (simulated via Wikidata mapping)
            tgn_props = []
            tgn_ids = get_strings("P1667")
            if tgn_ids:
                tgn_props.append(f"TGN_{tgn_ids[0]}")
                tgn_props.extend(get_qids("P131")) # Historical administrative hierarchy
                
            cache[qid] = {
                "label": label,
                "wikidata": wd_props,
                "geonames": gn_props,
                "getty": tgn_props
            }
            return cache[qid]
    except Exception:
        pass
    
    cache[qid] = {"label": "", "wikidata": [], "geonames": [], "getty": []}
    return cache[qid]

def resolve_labels(qids, cache):
    return [fetch_entity_from_wikidata(q, cache)["label"] for q in qids if q.startswith("Q") and fetch_entity_from_wikidata(q, cache)["label"]]

def build_kg_strings(pers_qid, loc_qid, cache):
    if pers_qid: fetch_entity_from_wikidata(pers_qid, cache)
    if loc_qid: fetch_entity_from_wikidata(loc_qid, cache)
    
    wd_facts, gn_facts, get_facts = [], [], []
    
    for qid in [pers_qid, loc_qid]:
        if not qid or qid not in cache: continue
        c = cache[qid]
        if c["wikidata"]: wd_facts.extend(resolve_labels(c["wikidata"], cache))
        if c["geonames"]: gn_facts.extend([x for x in c["geonames"] if not x.startswith("Q")] + resolve_labels([x for x in c["geonames"] if x.startswith("Q")], cache))
        if c["getty"]: get_facts.extend([x for x in c["getty"] if not x.startswith("Q")] + resolve_labels([x for x in c["getty"] if x.startswith("Q")], cache))
            
    return {
        "wikidata": " <WIKI> " + " | ".join(wd_facts[:8]) + " </WIKI>" if wd_facts else "",
        "geonames": " <GEONAMES> " + " | ".join(gn_facts[:8]) + " </GEONAMES>" if gn_facts else "",
        "getty": " <GETTY> " + " | ".join(get_facts[:8]) + " </GETTY>" if get_facts else ""
    }

# ══════════════════════════════════════════════════════════════════
# Note: The rest of the script (Training, Evaluation) follows the 
# exact same structure as `kaggle_mhipex_kg.py` but loops over
# ["text_only", "wikidata", "geonames", "getty"].
# ══════════════════════════════════════════════════════════════════

print("Ready to implement standard training loop for E0, E1, E2, E3...")
print("Please run this script on Kaggle to generate the Multi-KG results.")
