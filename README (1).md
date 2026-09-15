# AI Support Agent for AmazonHelp

An AI support agent built on the Customer Support on Twitter dataset (AmazonHelp brand):
classifies customer message intent, drafts a reply grounded in historically-resolved
similar cases, and decides auto-handle vs. escalate with a stated reason.

See `REPORT.md` for problem framing, results vs. baselines, failure analysis, and the
"what's misleading about my headline number" section. See `DECISION_LOG.md` for the
non-obvious decisions made along the way.

## Reproduce the headline results (~15 min, excluding your own dataset download)

### 0. Setup
```bash
pip install pandas requests scikit-learn
```
Set your Groq API key as an environment variable (`GROQ_API_KEY`) — see
`scripts/check_groq_models.py` if you need to verify which models are available to
your key.

### 1. Get the data
Download `twcs.csv` from Kaggle (`thoughtvector/customer-support-on-twitter`) and place
it at `data/twcs.csv`.

### 2. Run the pipeline in order
```bash
# Survey brands, confirm AmazonHelp is the right pick (optional -- already decided)
python scripts/02_brand_survey.py --input data/twcs.csv --top 20

# Build (customer, brand-reply) pairs for AmazonHelp
python scripts/01_load_and_thread.py --input data/twcs.csv --brand AmazonHelp --out data/pairs_amazonhelp.csv

# Subsample to a workable size
python scripts/03_subsample.py --input data/pairs_amazonhelp.csv --n-working 5000 --n-read 250 --out-working data/pairs_amazonhelp_subsample.csv --out-read data/messages_to_read.txt

# (Taxonomy already finalized in data/final_taxonomy.md -- scripts/04 shows how it was derived)

# Classify a sample of messages with the main system
python scripts/05_classify_intents.py --input data/pairs_amazonhelp_subsample.csv --n-classify 300 --out data/classified.csv

# Build the stratified golden-set review sheet, then hand-label it (see golden_review.csv)
python scripts/06_build_golden_set.py --input data/classified.csv --target-n 180 --out data/golden_review.csv

# Run the full agent pipeline (classify + retrieve + draft + escalate) over the golden set
python scripts/07_agent_pipeline.py --corpus data/pairs_amazonhelp_subsample.csv --golden data/golden_review.csv --out data/agent_predictions.csv --top-k 3

# Score: main system vs. both baselines, plus LLM-judge reply quality
python scripts/08_eval_harness.py score --golden data/golden_review.csv --predictions data/agent_predictions.csv --out data/eval_results.csv

# Human-score a sample for judge validation, then check agreement
python scripts/quick_human_score.py --input data/eval_results.csv --out data/eval_results_human_scored.csv --n 35
python scripts/08_eval_harness.py judge-agreement --scored data/eval_results_human_scored.csv
```

## Headline results

| System | Intent accuracy | Intent macro F1 | Escalation F1 |
|---|---|---|---|
| Main system | 0.700 | 0.714 | 0.836 |
| Trivial baseline | 0.250 | — | 0.000 |
| Simple baseline (TF-IDF+LogReg) | 0.400 | 0.195 | — |

Reply quality (LLM judge, 1-5, n=80): mean 4.45 — **see REPORT.md Section 4 for why this
number is misleading on its own.** Evaluated on an 80-example genuinely hand-reviewed
subset of the golden set — see REPORT.md Section 6 for why not the full 180.

## Repo structure
```
data/                  -- intermediate + final data artifacts (not all committed -- see .gitignore)
scripts/
  01_load_and_thread.py       -- reconstruct (customer, brand-reply) pairs
  02_brand_survey.py          -- data-driven brand selection
  03_subsample.py              -- cut corpus to a workable size, sample for reading
  04_propose_taxonomy.py       -- LLM-assisted intent taxonomy proposal
  05_classify_intents.py       -- main system: LLM few-shot intent classifier
  06_build_golden_set.py       -- stratified, pre-filled golden-set review sheet
  07_agent_pipeline.py         -- the actual agent: classify + retrieve + draft + escalate
  08_eval_harness.py           -- metrics, baselines, LLM judge, judge-human agreement
  quick_human_score.py         -- terminal tool for fast human scoring
  check_groq_models.py         -- diagnostic: list available models for your API key
REPORT.md              -- full report
DECISION_LOG.md         -- non-obvious decisions and why
data/final_taxonomy.md  -- the 9-intent taxonomy with definitions
data/golden_review.csv  -- the hand-labeled golden evaluation set (150-250 examples)
```

## Notes on borrowed code / tools
- Dataset: Kaggle `thoughtvector/customer-support-on-twitter`.
- LLM API: Groq (`openai/gpt-oss-120b` for drafting/judging, `openai/gpt-oss-20b` for
  classification).
- Libraries: pandas, scikit-learn (TF-IDF + Logistic Regression baseline), requests.
- No external code copied beyond standard library usage of the above.
