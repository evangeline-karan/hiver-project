# AI Support Agent for AmazonHelp — Report

## 1. Problem framing

**What "good" means for this brand.** AmazonHelp handles a huge volume of routine,
repetitive issues (delivery status, refunds, account access) alongside emotionally
charged complaints and occasional fraud reports. For this brand, "good" means: (a)
never confidently answer a message that isn't actually a resolvable support issue,
(b) escalate anything with real financial, security, or reputational risk rather than
auto-resolve it, and (c) for the large routine-issue majority, draft a reply consistent
with how the brand has actually responded before, not a generic template.

**What I chose not to build:**
- Multi-turn thread context (built single-hop customer→reply pairs only, given the
  time budget — see decision log).
- Non-English handling. A meaningful share of AmazonHelp messages are non-English
  (French, Japanese, Spanish, Portuguese observed in the sample). These are routed to
  `other_uncategorized` → escalate rather than attempting generation quality I can't
  evaluate.
- Dense embedding retrieval. Used TF-IDF cosine similarity instead — faster, no model
  download, fully deterministic, at some cost to semantic match quality on paraphrased
  issues (see failure analysis).
- Multi-label intent classification. Some real messages arguably span two intents
  (e.g. a refund request framed as a scam complaint); the system forces a single label.

## 2. Results vs. baselines

| System | Intent accuracy | Intent macro F1 | Escalation F1 |
|---|---|---|---|
| **Main system** (few-shot LLM + TF-IDF retrieval) | **0.700** | **0.714** | **0.836** |
| Trivial baseline (majority class, never escalate) | 0.250 | — | 0.000 |
| Simple baseline (TF-IDF + LogReg, 5-fold CV) | 0.400 | 0.195 | — |

Reply quality (LLM-as-judge, 1-5 scale, n=80): mean **4.45**, median **5.0**.

Evaluated on an 80-example genuinely hand-reviewed subset of the golden set (see Section
6 and the golden-set methodology note for why 80, not the full 180). The main system
clearly beats both baselines: roughly 2.8x the trivial baseline's accuracy and 1.75x the
simple baseline's, while also producing macro F1 well above either (the simple
baseline's low macro F1 relative to its accuracy reflects it collapsing onto the
majority class for several of the rarer intents, given only 80 training labels split
across 9 categories).

## 6. Golden-set methodology note

180 candidate examples were stratified-sampled across the LLM's own predicted intent
buckets (to guarantee coverage of rare intents like `security_scam_report`) from a
300-message classified sample of the 5,000-message AmazonHelp working corpus. Due to
time constraints, **80 of the 180** were genuinely hand-reviewed and are the basis for
all reported metrics; the remaining 100 are unreviewed LLM pre-fills and were excluded
from evaluation rather than counted as ground truth.

Labeling process: each of the 80 messages was read in full, an initial intent reading
was proposed (by the author working with an AI assistant, Claude, to accelerate reading
through messages), and the author confirmed or corrected each one individually before
it counted as final. 26 of 80 (32.5%) were corrected from the LLM classifier's original
guess — a plausible, defensible disagreement rate that gives confidence the labels
reflect real judgment rather than either rubber-stamping or noise (an earlier attempt at
self-labeling via a terminal tool produced a 92.5% correction rate that, on inspection,
consisted of essentially random category swaps — that data was discarded rather than
reported, precisely to avoid submitting a fabricated-looking result).

## 3. Failure analysis (top failure modes, with real examples)

**1. Non-actionable messages get treated as support requests.**
The system doesn't reliably distinguish "this is a support issue" from "this mentions
the brand but isn't asking for anything." Real examples pulled during human review:
- Customer: *"Eu te amo, @117086!"* ("I love you") → system drafted a reply offering
  help, when no help was requested.
- Customer: *"I'm watching Mr. Robot ... #PrimeVideo #Hype"* (a fan tweet) → system
  replied "Glad you're enjoying it! DM us your account details" — inventing a reason
  to ask for account details on a message that wasn't a support contact at all.
- Customer: *"Thank God ... filled the box with extra packing"* (a thank-you, not a
  complaint) → system replied offering to "double-check your order's packing."
Hypothesis: the classifier's `other_uncategorized` bucket isn't being selected often
enough for borderline non-requests — it's pattern-matching on brand mentions and
support-adjacent vocabulary rather than confirming there's an actual ask.

**2. LLM-as-judge inflates reply quality relative to human judgment.**
Judge mean was 4.45/5 (n=80, same subset as the accuracy numbers above). On a 10-row human-scored sample: exact agreement with the judge
was only 20%, though 70% of scores were within one point, and the judge ran
**1.2 points higher than the human score on average**. This is a real, measured gap,
not a hedge — see Section 4.

**3. TF-IDF retrieval misses paraphrased or semantically-similar-but-lexically-different
precedents.** Two of the lowest-similarity retrievals in the predictions
(`top_retrieval_similarity` = 0.00) show the failure clearly: a Japanese message about
a missing payment-confirmation email retrieved an unrelated Spanish message about
confirming a bank charge — same rough topic (payment confirmation) but zero lexical
overlap, so TF-IDF found nothing. A dense embedding retriever would likely have matched
on semantic similarity across languages/phrasings where TF-IDF cannot.

**4. Escalation sometimes fires on tone alone even for low-stakes issues.**
The keyword-based anger detector (`ANGRY_MARKERS`) can over-trigger on hyperbolic but
low-stakes language (e.g. "worst" used about a minor annoyance), while under-triggering
on calmly-worded but genuinely serious complaints that don't use flagged words.

**5. Non-English messages are consistently routed to escalate, which is correct by
design but means the system provides zero automation value for a meaningful fraction
of real traffic.** A heuristic scan of the 180-example golden pool found roughly
**13.9% (25/180)** likely non-English messages (German, Portuguese, Japanese, Spanish,
French observed), all routed to `other_uncategorized` → escalate per the documented
scope decision. That's a real cost, not just a caveat — over an eighth of traffic gets
zero automation benefit under the current design.

## 4. What's misleading about my headline number

The 4.45/5 reply-quality score is the single most misleading number in this report if
taken at face value. Three specific reasons:
1. **Judge-model self-preferencing.** The judge (`gpt-oss-120b`) and the reply drafter
   (`gpt-oss-120b`) are the same model family — a known failure mode where LLM judges
   rate outputs from similar models more favorably. The measured 1.2-point gap between
   judge and human scores on a 10-row sample is direct evidence of this, not just a
   theoretical concern.
2. **The human-agreement check itself is small** (n=10, below the 20-30 the assignment
   suggests) due to time constraints — the 1.2-point gap estimate has real uncertainty
   and would benefit from more hand-scored rows.
3. **The eval set's message pool was selected via stratified sampling from the LLM's
   OWN predictions** (from `classified.csv`, which the LLM itself labeled). While the
   final labels themselves are now genuinely hand-corrected (see Section 6), the
   *selection* of which messages appear in the eval set still reflects what the LLM
   tends to predict — an intent the LLM never guesses at all for any message would
   never enter the eval pool to begin with, so recall on such a "blind spot" category
   is structurally unmeasurable here.
4. The golden set (180 candidate examples, 80 genuinely hand-reviewed, see Section 6)
   and predictions were all drawn from a 5,000-message subsample of one brand's traffic
   — not the full 153,004-pair corpus, let alone the full Twitter support dataset.
   Headline numbers should be read as "performance on this subsample," not "performance
   on AmazonHelp support broadly."
5. **The final eval sample size (n=80) is below the 150-250 the assignment asked for**,
   and the judge-human agreement check is smaller still (n=10). Both are disclosed
   honestly here rather than padded: given the time available, a smaller, genuinely
   human-reviewed sample was judged more trustworthy than a larger one that wasn't
   actually reviewed (see Section 6 for what happened when a first labeling attempt
   produced unreliable data, and why it was discarded rather than used).

## 5. What I'd do next with one more week

- Hand-label a properly-sized golden set (300+) sampled from raw messages directly,
  not from the LLM's own predictions, to remove the circularity in issue #3 above.
- Replace TF-IDF retrieval with sentence-transformer embeddings for better semantic
  matching on paraphrased issues.
- Use a different, ideally stronger/independent model family as the LLM judge to
  reduce self-preferencing risk, and get the human-scored sample to 30-40 rows.
- Build an explicit "is this even a support request?" pre-filter ahead of intent
  classification, directly targeting failure mode #1.
- Reconstruct full multi-turn threads rather than single-hop pairs, since some issues
  clearly span multiple exchanges in the raw data.
- Quantify the non-English traffic share and decide whether it's worth building
  language-specific handling vs. keeping the escalate-everything policy.

## Reproducing these results
See README.md — full pipeline runs via scripts/01 through scripts/08 in order,
in well under 15 minutes excluding LLM API call latency.
