"""
Step 1: Load the raw Customer Support on Twitter dataset and reconstruct
(customer_message -> brand_reply) pairs for a single brand.

Input : data/twcs.csv   (download from Kaggle: thoughtvector/customer-support-on-twitter,
                          file is usually named twcs.csv, ~3M rows)
Output: data/pairs_<brand>.csv  -- one row per (customer msg, brand reply) pair,
                                    plus lightweight thread context.

Usage:
    python 01_load_and_thread.py --input data/twcs.csv --brand AmazonHelp --out data/pairs_amazonhelp.csv
"""

import argparse
import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"tweet_id": str, "in_response_to_tweet_id": str,
                                   "response_tweet_id": str}, low_memory=False)
    df["text"] = df["text"].astype(str)
    return df


def build_pairs(df: pd.DataFrame, brand_handle: str) -> pd.DataFrame:
    """
    A 'pair' is: an inbound (customer) tweet, and the brand's direct reply to it
    (in_response_to_tweet_id on the brand tweet == the customer tweet_id).

    We keep it to single-hop pairs for v1 -- multi-turn thread stitching is a
    reasonable v2 extension, worth a line in the decision log either way.
    """
    by_id = df.set_index("tweet_id", drop=False)

    brand_replies = df[(df["inbound"] == False) & (df["author_id"] == brand_handle)].copy()
    brand_replies = brand_replies[brand_replies["in_response_to_tweet_id"].notna()]

    records = []
    missing_parent = 0
    for _, reply in brand_replies.iterrows():
        parent_id = reply["in_response_to_tweet_id"]
        if parent_id not in by_id.index:
            missing_parent += 1
            continue
        parent = by_id.loc[parent_id]
        if isinstance(parent, pd.DataFrame):  # duplicate ids, guard against it
            parent = parent.iloc[0]
        if not parent["inbound"]:
            # brand replying to itself / another brand tweet -- skip, not a customer msg
            continue
        records.append({
            "thread_id": parent_id,
            "customer_tweet_id": parent["tweet_id"],
            "customer_text": parent["text"],
            "customer_created_at": parent["created_at"],
            "brand_tweet_id": reply["tweet_id"],
            "brand_text": reply["text"],
            "brand_created_at": reply["created_at"],
        })

    pairs = pd.DataFrame(records)
    print(f"[info] built {len(pairs)} pairs for brand={brand_handle} "
          f"({missing_parent} replies had no resolvable parent tweet)")
    return pairs


def basic_clean(pairs: pd.DataFrame) -> pd.DataFrame:
    before = len(pairs)
    # drop obviously empty / near-empty text
    pairs = pairs[pairs["customer_text"].str.strip().str.len() > 3]
    pairs = pairs[pairs["brand_text"].str.strip().str.len() > 3]
    # drop exact-duplicate customer messages (bots / retries) keeping first
    pairs = pairs.drop_duplicates(subset=["customer_text"])
    pairs = pairs.reset_index(drop=True)
    print(f"[info] cleaning: {before} -> {len(pairs)} pairs")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to raw twcs.csv")
    ap.add_argument("--brand", required=True, help="brand author_id, e.g. AmazonHelp")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    df = load_raw(args.input)
    pairs = build_pairs(df, args.brand)
    pairs = basic_clean(pairs)
    pairs.to_csv(args.out, index=False)
    print(f"[done] wrote {len(pairs)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
