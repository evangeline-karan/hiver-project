"""
Step 2: Survey brands in the raw dataset BEFORE picking one, so the choice is
evidence-based rather than a guess. Answers: which brands have enough volume,
enough single-hop reply pairs, and reasonable message length (a proxy for
"repeatable, templatable issues" vs "one-off essays").

Usage:
    python 02_brand_survey.py --input data/twcs.csv --top 20
"""

import argparse
import pandas as pd


def survey(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    brands = df[df["inbound"] == False]["author_id"].value_counts().head(top_n)

    rows = []
    for brand, total_replies in brands.items():
        brand_df = df[(df["inbound"] == False) & (df["author_id"] == brand)]
        replies_to_customers = brand_df["in_response_to_tweet_id"].notna().sum()

        customer_msgs = df[(df["inbound"] == True)]
        # crude proxy: customer tweets that @-mention this brand
        mentions = customer_msgs["text"].str.contains(f"@{brand}", case=False, na=False).sum()

        avg_len = brand_df["text"].str.len().mean()

        rows.append({
            "brand": brand,
            "total_brand_tweets": total_replies,
            "replies_with_parent": replies_to_customers,
            "customer_mentions": mentions,
            "avg_reply_char_len": round(avg_len, 1) if pd.notna(avg_len) else None,
        })

    out = pd.DataFrame(rows).sort_values("replies_with_parent", ascending=False)
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--top", type=int, default=20, help="how many top brands (by tweet volume) to survey")
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype={"tweet_id": str, "in_response_to_tweet_id": str},
                      low_memory=False)
    df["inbound"] = df["inbound"].astype(str).str.strip().str.lower() == "true"

    result = survey(df, args.top)
    pd.set_option("display.width", 120)
    print(result.to_string(index=False))
    result.to_csv("data/brand_survey.csv", index=False)
    print("\n[done] wrote data/brand_survey.csv")
    print("\nLook for: high replies_with_parent (lots of resolvable pairs) and a brand")
    print("whose product surface you understand well enough to judge reply quality.")


if __name__ == "__main__":
    main()
