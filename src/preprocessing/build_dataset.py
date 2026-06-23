import json, re
from pathlib import Path
from src.utils.config import AT_LABEL_MAP, ISAT_LABEL_MAP, TRAIN_FILES, DEV_FILES

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_input(text, person_mentions, loc_mentions):
    person = person_mentions[0] if person_mentions else "UNKNOWN"
    place  = loc_mentions[0]   if loc_mentions   else "UNKNOWN"
    person_clean = clean_text(person)
    place_clean  = clean_text(place)
    text_clean   = clean_text(text)
    return f"{text_clean} [SEP] Person: {person_clean} [SEP] Place: {place_clean}"

def process_file(filepath, lang):
    records = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            for pair in doc.get("sampled_pairs", []):
                at_raw   = pair.get("at",   "FALSE")
                isat_raw = pair.get("isAt", "FALSE")
                if at_raw not in AT_LABEL_MAP:
                    continue
                if isat_raw not in ISAT_LABEL_MAP:
                    continue
                records.append({
                    "document_id": doc["document_id"],
                    "language":    lang,
                    "date":        doc.get("date", ""),
                    "input_text":  build_input(
                                       doc["text"],
                                       pair["pers_mentions_list"],
                                       pair["loc_mentions_list"]
                                   ),
                    "person":      pair["pers_mentions_list"][0] if pair["pers_mentions_list"] else "",
                    "place":       pair["loc_mentions_list"][0]  if pair["loc_mentions_list"]  else "",
                    "at_label":    AT_LABEL_MAP[at_raw],
                    "isat_label":  ISAT_LABEL_MAP[isat_raw],
                    "at_raw":      at_raw,
                    "isat_raw":    isat_raw,
                })
    return records

def build_all(output_dir="data/processed"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for split, files in [("train", TRAIN_FILES), ("dev", DEV_FILES)]:
        all_records = []
        for lang, path in files.items():
            if Path(path).exists():
                recs = process_file(path, lang)
                all_records.extend(recs)
                print(f"  {lang} {split}: {len(recs)} pairs")
            else:
                print(f"  WARNING: {path} not found")
        out_path = Path(output_dir) / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(all_records)} records -> {out_path}")

if __name__ == "__main__":
    print("Building dataset...")
    build_all()
    print("Done.")
