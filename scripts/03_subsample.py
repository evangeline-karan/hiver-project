"""
Step 3: Take a big pairs file (100k+ rows) and cut it down to a workable
subsample for building the retrieval index / doing EDA, PLUS dump a plain-text
batch of customer messages for you to read by hand (this reading is how you
build the intent taxonomy in Day 2 -- there's no shortcut around actually
reading real messages).

Usage:
    python 03_subsample.py --input data/pairs_amazonhelp.csv \
        --n-working 5000 --n-read 250 \
        --out-working data/pairs_amazonhelp_subsample.csv \
        --out-read data/messages_to_read.txt
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n-working", type=int, default=5000,
                     help="size of the working subsample used for retrieval/eda in later steps")
    ap.add_argument("--n-read", type=int, default=250,
                     help="how many customer messages to dump as plain text for manual reading")
    ap.add_argument("--out-working", required=True)
    ap.add_argument("--out-read", required=True)
    ap.add_argument("--seed", type=int, default=42, help="fixed seed so this is reproducible")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"[info] loaded {len(df)} pairs")

    n_working = min(args.n_working, len(df))
    working = df.sample(n=n_working, random_state=args.seed).reset_index(drop=True)
    working.to_csv(args.out_working, index=False)
    print(f"[info] wrote working subsample: {len(working)} pairs -> {args.out_working}")

    n_read = min(args.n_read, len(working))
    to_read = working.sample(n=n_read, random_state=args.seed + 1)

    with open(args.out_read, "w", encoding="utf-8") as f:
        for i, row in enumerate(to_read.itertuples(), 1):
            f.write(f"--- {i} (thread_id={row.thread_id}) ---\n")
            f.write(f"CUSTOMER: {row.customer_text}\n")
            f.write(f"BRAND:    {row.brand_text}\n\n")

    print(f"[info] wrote {n_read} messages to read -> {args.out_read}")
    print("\n[next step] Open messages_to_read.txt and read through it. As you read, keep a")
    print("running scratch list of the ~6-10 recurring issue *types* you're seeing. That")
    print("list becomes your intent taxonomy in Day 2 -- don't try to be exhaustive on the")
    print("first pass, just start naming buckets and merge/split as patterns repeat.")


if __name__ == "__main__":
    main()
