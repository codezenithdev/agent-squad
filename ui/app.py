"""Streamlit UI (v2.3) — watch the pipeline run live.

Consumes the API's Server-Sent Events stream (POST /run/stream) so each agent's
step appears as it completes, then renders the final document.

Run the API first:   uvicorn api.main:app --port 8000
Then the UI:         streamlit run ui/app.py
"""
from __future__ import annotations

import json

import httpx
import streamlit as st

st.set_page_config(page_title="Multi-Agent Architect", page_icon="🛠️", layout="wide")
st.title("🛠️ Multi-Agent Architect")
st.caption("Plan → design → code → audit → test → review → compile, live.")

with st.sidebar:
    api_base = st.text_input("API base URL", "http://localhost:8000")
    thread_id = st.text_input("Thread ID", "ui-run-1")
    try:
        health = httpx.get(f"{api_base}/", timeout=5).json()
        st.success(f"API up · mock_mode={health.get('mock_mode')}")
    except Exception:
        st.error("API not reachable — start `uvicorn api.main:app --port 8000`")

requirement = st.text_area(
    "Requirement",
    "Design a scalable job board with Next.js frontend, FastAPI backend, "
    "PostgreSQL, user auth, job listings with search, and an employer dashboard",
    height=110,
)

if st.button("Run pipeline", type="primary"):
    timeline = st.container()
    timeline.subheader("Live agent timeline")
    final_doc = ""

    try:
        with httpx.stream(
            "POST",
            f"{api_base}/run/stream",
            json={"requirement": requirement, "thread_id": thread_id},
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                evt = json.loads(line[6:])
                if evt.get("type") == "step":
                    timeline.write(
                        f"**{evt['step']:>2}. `{evt['node']}`** — {evt['summary']}"
                    )
                elif evt.get("type") == "done":
                    fw = evt.get("frameworks_detected", {})
                    st.success(
                        f"Finished in {evt['steps']} steps · "
                        f"stack: {fw.get('frontend')}/{fw.get('backend')} · "
                        f"review: {evt.get('review_decision')}"
                    )
                    final_doc = evt.get("final_document", "")
    except Exception as e:  # noqa: BLE001
        st.error(f"Stream failed: {e}")

    if final_doc:
        st.subheader("Final document")
        st.markdown(final_doc)
        st.download_button("Download output.md", final_doc, file_name="output.md")
