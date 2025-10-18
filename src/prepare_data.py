
import os
import argparse
import pandas as pd

def infer_label_from_path(path):
    p = path.lower()
    if "fake" in p:
        return "fake"
    if "real" in p:
        return "real"
    return None

def read_all_txt(root_dir):
    rows = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.lower().endswith(".txt"):
                continue
            full = os.path.join(dirpath, fn)
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read().strip()
            except Exception as e:
                print(f"skip {full}: {e}")
                continue
            label = infer_label_from_path(dirpath)
            if label is None:
                print(f"skip (no label inferred): {full}")
                continue
            if not txt:
                continue
            rows.append({"text": txt, "label": label, "source_file": full})
    return pd.DataFrame(rows)

def main(args):
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = read_all_txt(args.raw_dir)
    print(f"found {len(df)} documents")
    # drop duplicates (exact)
    df["text_norm"] = df["text"].str.strip().str.lower()
    df = df.drop_duplicates("text_norm").drop(columns=["text_norm"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(df)} rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw", help="root directory with fake/real txt files")
    ap.add_argument("--out-csv", default="data/dataset.csv")
    args = ap.parse_args()
    main(args)
