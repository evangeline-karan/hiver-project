"""
Streamlit demo for the AI support agent. Type any customer message, see the full
pipeline run live: intent classification -> retrieval grounding -> drafted reply ->
escalation decision with reason. Built for demoing the project, not for grading --
the graded pipeline is scripts/01 through scripts/08.

Run with:
    streamlit run app.py

Requires GROQ_API_KEY set as an environment variable, and data/pairs_amazonhelp_subsample.csv
to exist (run scripts/01-03 first if it doesn't).
"""

import os
import sys

import pandas as pd
import streamlit as st

# module names starting with a digit ("07_agent_pipeline") can't be imported with a
# normal `import` statement, so load it explicitly by file path instead.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "agent_pipeline",
    os.path.join(os.path.dirname(__file__), "scripts", "07_agent_pipeline.py"),
)
agent_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent_pipeline)


st.set_page_config(page_title="AmazonHelp AI Support Agent", page_icon="\U0001F4E6", layout="wide")

st.title("AI Support Agent — AmazonHelp")
st.caption(
    "Classifies intent, retrieves grounding from real historically-resolved cases, "
    "drafts a reply, and decides auto-handle vs. escalate — with a stated reason."
)

# --- headline results, shown honestly (including the caveat) ---
with st.expander("Headline eval results (click to expand)", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("Intent accuracy", "0.761", help="vs. trivial baseline 0.244, simple baseline 0.372")
    col2.metric("Escalation F1", "0.819")
    col3.metric("Reply quality (LLM judge)", "4.48 / 5", help="See caveat below")
    st.warning(
        "The 4.48 judge score is likely inflated: a small human-agreement check (n=10) "
        "found the judge scores ~1.2 points higher on average than a human reviewer, "
        "consistent with judge/drafter model self-preferencing (same model family for both). "
        "See REPORT.md Section 4 for the full breakdown."
    )


@st.cache_resource
def load_retrieval_index():
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "pairs_amazonhelp_subsample.csv")
    corpus_df = pd.read_csv(corpus_path)
    vec, matrix = agent_pipeline.build_retrieval_index(corpus_df)
    return corpus_df, vec, matrix


try:
    corpus_df, vec, matrix = load_retrieval_index()
    corpus_loaded = True
except FileNotFoundError:
    corpus_loaded = False
    st.error(
        "data/pairs_amazonhelp_subsample.csv not found. Run scripts/01_load_and_thread.py and "
        "scripts/03_subsample.py first to build the retrieval corpus."
    )

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY environment variable is not set. Set it and restart this app.")

st.subheader("Try it")
example_messages = [
    "Type your own message, or pick an example below...",
    "@AmazonHelp my package was supposed to arrive yesterday and it's still not here!",
    "@AmazonHelp I want a refund, the item arrived broken",
    "@AmazonHelp is 02245430129 a real Amazon number or a scam?",
    "@AmazonHelp when does the new Kindle release?",
]
choice = st.selectbox("Example messages", example_messages)
default_text = "" if choice == example_messages[0] else choice
user_text = st.text_area("Customer message", value=default_text, height=100)

run = st.button("Run agent", type="primary", disabled=not (corpus_loaded and api_key))

if run and user_text.strip():
    with st.spinner("Classifying intent..."):
        intent = agent_pipeline.classify_intent(user_text, api_key)

    with st.spinner("Retrieving similar historical cases..."):
        retrieved = agent_pipeline.retrieve_similar(user_text, vec, matrix, corpus_df, top_k=3)

    with st.spinner("Drafting reply..."):
        reply = agent_pipeline.draft_reply(user_text, retrieved, api_key)

    decision, reason = agent_pipeline.decide_escalation(intent, user_text, retrieved)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Predicted intent")
        st.info(intent)
    with c2:
        st.markdown("### Escalation decision")
        if decision == "escalate":
            st.error(f"**ESCALATE**\n\n{reason}")
        else:
            st.success(f"**AUTO-HANDLE**\n\n{reason}")

    st.markdown("### Drafted reply")
    st.write(reply)

    st.markdown("### Grounded on these historical precedents")
    for r in retrieved:
        with st.container(border=True):
            st.caption(f"Similarity: {r['similarity']:.2f}")
            st.write(f"**Past customer message:** {r['customer_text']}")
            st.write(f"**How it was resolved:** {r['brand_text']}")
elif run:
    st.warning("Type a message first.")

st.divider()
st.caption(
    "Built on the Customer Support on Twitter dataset (Kaggle, thoughtvector/customer-support-on-twitter). "
    "See REPORT.md and DECISION_LOG.md for full methodology, results, and limitations."
)