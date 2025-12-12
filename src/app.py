# Main Streamlit app for RAG

import streamlit as st
import time
import re
from typing import List, Dict, Any

from rag import retrieve_and_prepare
from gen_client import generate_answer  # your working Gemini client

# pandas for table display
try:
    import pandas as pd
except Exception:
    pd = None

# Streamlit page config
st.set_page_config(page_title="RAG QA - Qatar IMF Document", layout="wide")

st.title("Qatar IMF Document — RAG Question Answering System")
st.write(
    "Enter a question and the system will retrieve relevant excerpts from the Qatar IMF PDF and "
    "generate an answer grounded in those excerpts."
)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Top-k retrieved chunks", 1, 10, 5)
    max_tokens = st.number_input("Max output tokens", 64, 2048, 512)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    st.markdown("---")
    st.markdown(
        "Notes:\n"
        "- Make sure `data/meta.jsonl` and `data/faiss.index` exist.\n"
        "- Ensure your `.env` contains `GOOGLE_API_KEY` (used by google-generativeai) if you want live generation.\n"
        "- Recommended: set Temperature = 0.0 for deterministic concise answers during demo."
    )

# Session state for last query
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

question = st.text_input("Ask your question here:", value=st.session_state.get("last_query", ""))

# Button remains for users who prefer clicking
run_button = st.button("Retrieve & Generate")

# Determine whether to run:
# - Run if user clicked button
# - OR run if question is non-empty and different from last_query (i.e. Enter pressed / changed)
should_run = False
if run_button:
    should_run = True
elif question and question.strip() and question.strip() != st.session_state.get("last_query", "").strip():
    should_run = True

# Heuristic table parser
def parse_table_from_chunks(retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Heuristic parser that looks for IMF-style table chunks and tries to extract rows -> numeric sequences.
    Returns a dict of {title: pandas.DataFrame} if pandas is available, otherwise returns structured dicts.
    """
    tables = {}
    num_re = re.compile(r"[-+]?\d+(?:\.\d+)?")
    #Row labels commonly found in IMF Qatar document tables
    row_labels = [
        "Revenue", "Expenditure", "Real GDP", "Nominal GDP", "Oil", "LNG",
        "Investment income from public enterprises", "Corporate tax revenue",
        "Other revenue", "Expense", "Compensation of employees", "Goods and services",
        "Nonhydrocarbon", "Non-hydrocarbon", "Nominal nonhydrocarbon GDP"
    ]

    for r in retrieved:
        text = (r.get("text") or "").replace("\n", " ").replace("\r", " ")
        # quick heuristic: must contain either 'Table' or multiple year-like numbers
        if "Table" not in text and not re.search(r"\b20\d{2}\b", text):
            continue

        # try to collect rows
        rows = {}
        # attempt simple split by semicolons, "  " multiple spaces, or " 202" boundaries
        candidates = re.split(r"\s{2,}|\s*;\s*", text)
        # fallback to sentence split if too few candidates
        if len(candidates) < 2:
            candidates = re.split(r"\. ", text)

        for seg in candidates:
            seg = seg.strip()
            if not seg:
                continue
            for label in row_labels:
                # case-insensitive label presence
                if re.search(rf"\b{re.escape(label)}\b", seg, flags=re.IGNORECASE):
                    nums = num_re.findall(seg)
                    if nums:
                        rows[label] = [float(x) for x in nums]
                    else:
                        # try to find numbers elsewhere in the text after the label
                        m = re.search(rf"{re.escape(label)}(.*?)(?=[A-Z][a-z]|\Z)", text, flags=re.IGNORECASE)
                        if m:
                            nums2 = num_re.findall(m.group(1))
                            if nums2:
                                rows[label] = [float(x) for x in nums2]
                    break  

        # attempt to discover years 
        years = []
        year_matches = re.findall(r"\b(20\d{2})\b", text)
        if year_matches:
            # deduplicate years while preserving order
            seen = set()
            for yy in year_matches:
                y = int(yy)
                if y not in seen:
                    years.append(y)
                    seen.add(y)

        # Build DataFrame if pandas available, else a dict
        if rows:
            maxlen = max(len(v) for v in rows.values())
            if years and len(years) == maxlen:
                # align rows to years
                if pd:
                    df = pd.DataFrame(index=years)
                    for k, v in rows.items():
                        vals = v[:len(years)] + [None] * max(0, len(years) - len(v))
                        df[k] = vals
                    title = f"Table_from_page_{r.get('page')}"
                    tables[title] = df
                else:
                    # return dict with years key
                    table_dict = {"years": years}
                    for k, v in rows.items():
                        vals = v[:len(years)] + [None] * max(0, len(years) - len(v))
                        table_dict[k] = vals
                    tables[f"Table_from_page_{r.get('page')}"] = table_dict
            else:
                # fallback to row-major dict / df with generic columns
                if maxlen > 0:
                    cols = [f"C{i+1}" for i in range(maxlen)]
                    if pd:
                        df = pd.DataFrame(columns=cols)
                        for k, v in rows.items():
                            vals = v[:maxlen] + [None] * (maxlen - len(v))
                            df.loc[k] = vals
                        tables[f"Table_from_page_{r.get('page')}"] = df
                    else:
                        td = {}
                        for k, v in rows.items():
                            vals = v[:maxlen] + [None] * (maxlen - len(v))
                            td[k] = vals
                        tables[f"Table_from_page_{r.get('page')}"] = td

    return tables

#Main execution
if should_run:
    # update last_query immediately to avoid double-run on rerun
    st.session_state["last_query"] = question.strip()

    t0 = time.time()

    # Retrieval step
    try:
        with st.spinner("Retrieving relevant chunks..."):
            out = retrieve_and_prepare(question, k=k)
    except Exception as e:
        st.error(f"Retrieval failed: {e}")
        raise

    retrieved = out.get("retrieved", [])
    prompt = out.get("prompt", "")

    # Trying parsing tables from retrieved chunks
    tables = parse_table_from_chunks(retrieved) if retrieved else {}

    # Inform user if no relevant chunks found
    if not retrieved:
        st.warning("No relevant chunks found for this query. The model will respond 'I don't know.' per instructions.")
    else:
        st.subheader("📄 Retrieved Chunks")
        for r in retrieved:
            st.markdown(f"**Page {r.get('page')} — Chunk {r.get('chunk_id')}**")
            st.write(r.get("text")[:800])
            st.caption(f"Score: {r.get('_score')}")
            st.write("---")

    # Display parsed tables 
    if tables:
        st.subheader("Extracted Table(s)")
        for title, table in tables.items():
            st.markdown(f"**{title}**")
            if pd and isinstance(table, pd.DataFrame):
                st.dataframe(table)
            else:
                # show fallback dict
                st.write(table)

    # Show the prompt 
    st.subheader("Prompt Sent to the LLM (truncated)")
    st.code(prompt[:3000], language="text")

    # Generation step
    with st.spinner("Generating answer..."):
        try:
            # use the slider's temperature and max_tokens (temperature default 0.0 )
            answer = generate_answer(prompt, max_tokens=int(max_tokens), temperature=float(temperature))
        except Exception as e:
            st.error(f"Generation failed: {e}")
            answer = None

    # Post-process & display concise answer first, then full answer
    st.subheader("Final Answer (concise)")
    if answer:
        # attempt 1: first non-empty line
        first_line = ""
        for line in answer.splitlines():
            line = line.strip()
            if line:
                first_line = line
                break

        # attempt 2: try to extract numeric projection 
        percent_match = re.search(r"(\d+(?:\.\d+)?\s?%|\d+¾|\d+½|\d+¼|\d+(?:\.\d+)?\s?percent|\b\d+(?:\.\d+)?\s?percent\b)", answer, flags=re.IGNORECASE)
        gdp_phrase_match = re.search(r"(real\s+GDP\s+growth[^\.]*\d+(?:\.\d+)?\s?%|\bprojected\s+to\s+\d+(?:\.\d+)?\s?%|\bprojected\s+GDP\s+growth[^\.]*)", answer, flags=re.IGNORECASE)

        concise_display = None
        if percent_match and gdp_phrase_match:
            # if both present, prefer a full descriptive phrase
            concise_display = gdp_phrase_match.group(0).strip()
        elif percent_match:
            concise_display = percent_match.group(0).strip()
            concise_display = f"Projected GDP growth: {concise_display}"
        elif first_line:
            concise_display = first_line
        else:
            concise_display = answer.splitlines()[0][:200]

        st.markdown(f"**{concise_display}**")

        # Then show full generated answer below
        st.subheader("Full generated answer")
        st.write(answer)
    else:
        st.info("No answer generated.")

    # Sources
    st.subheader("📚 Sources (retrieved chunks)")
    if not retrieved:
        st.write("No sources — query appears outside the document.")
    else:
        for r in retrieved:
            st.write(f"Page {r.get('page')} — Chunk {r.get('chunk_id')}")

    st.success(f"Done in {time.time() - t0:.2f} seconds")
