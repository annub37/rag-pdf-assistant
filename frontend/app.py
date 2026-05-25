"""
RAG PDF Assistant — Streamlit Frontend
=======================================
This is the ENTIRE frontend. One file. Pure Python.

HOW STREAMLIT WORKS:
- Every time a user clicks a button or types something, Streamlit
  re-runs THIS ENTIRE FILE from top to bottom.
- st.session_state is a dictionary that SURVIVES re-runs.
  Without it, variables would reset every time the user interacts.
- We use the `requests` library to call our FastAPI backend API.
"""

import streamlit as st
import requests

# ─── CONFIGURATION ───────────────────────────────────────────────
# WHY: The frontend needs to know WHERE the backend API lives.
# In production, this would be a real URL. Locally, FastAPI runs on port 8000.
API_BASE = "http://127.0.0.1:8000"

# ─── PAGE SETUP ──────────────────────────────────────────────────
# WHY: This configures the browser tab (title, icon, layout).
# Must be the FIRST Streamlit command in the script.
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="wide",  # use full browser width
)

st.title("📄 RAG PDF Assistant")
st.caption("Upload a PDF, then ask questions about it. Powered by RAG + Azure OpenAI.")

# ─── SIDEBAR: PDF UPLOAD ─────────────────────────────────────────
# WHY sidebar? The upload control should always be visible but not block
# the main chat area. Sidebar = persistent left panel.
with st.sidebar:
    st.header("📁 Upload PDFs")

    # st.file_uploader returns None if nothing uploaded, or a list of file-like objects.
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],              # only allow PDFs
        accept_multiple_files=True,
    )

    # This button triggers the upload to our FastAPI backend
    if uploaded_files and st.button("🚀 Upload & Process", use_container_width=True):
        # WHY st.spinner? It shows a loading animation while we wait.
        # Without it, the user sees nothing and thinks the app is frozen.
        with st.spinner(f"Uploading and processing {len(uploaded_files)} PDF(s)..."):
            try:
                # Send all PDFs to our FastAPI POST /documents/upload-multiple endpoint
                files_payload = [
                    ("files", (f.name, f.getvalue(), "application/pdf"))
                    for f in uploaded_files
                ]
                response = requests.post(
                    f"{API_BASE}/documents/upload-multiple",
                    files=files_payload,
                    timeout=300,
                )

                if response.status_code == 200:
                    data = response.json()
                    st.success(
                        f"✅ {data['successful']}/{data['total_files']} files processed successfully!"
                    )
                    if data["errors"]:
                        for err in data["errors"]:
                            if "already been uploaded" in err["error"]:
                                st.info(f"📄 {err['filename']}: Already uploaded — skipped duplicate.")
                            else:
                                st.warning(f"⚠️ {err['filename']}: {err['error']}")
                    # Save to session_state so we remember the upload even after re-runs
                    st.session_state["last_upload"] = data
                else:
                    st.error(f"❌ Upload failed: {response.json().get('detail', response.text)}")
            except requests.ConnectionError:
                st.error("❌ Cannot connect to backend. Is the FastAPI server running on port 8000?")

    # Show last upload info if available
    if "last_upload" in st.session_state:
        st.divider()
        st.subheader("Last Upload")
        info = st.session_state["last_upload"]
        if "results" in info:
            # Multi-file upload response
            st.write(f"**Files processed:** {info['successful']}/{info['total_files']}")
            for r in info["results"]:
                st.write(f"- **{r['original_name']}**: {r['total_pages']} pages, {r['total_chunks']} chunks")
        else:
            # Single-file upload response (backward compat)
            st.write(f"**File:** {info['original_name']}")
            st.write(f"**Pages:** {info['total_pages']}")
            st.write(f"**Chunks:** {info['total_chunks']}")

    # ─── SUMMARIZE BUTTON ────────────────────────────────────────
    # WHY always visible? The PDF might have been uploaded via API/terminal,
    # not just through the UI. So we always show this button.
    st.divider()
    if st.button("📝 Summarize PDF", use_container_width=True):
        st.session_state["_summarize"] = True

# ─── HANDLE SUMMARIZE (outside sidebar) ─────────────────────────
# WHY outside sidebar? We want the answer to appear in the main chat area,
# not inside the sidebar. Streamlit buttons set a flag, then we act on it here.
_pending_question = None
if st.session_state.get("_summarize"):
    st.session_state["_summarize"] = False
    _pending_question = "Provide a detailed summary of the entire document. Cover all key topics and main points from every page."
    st.session_state["messages"].append({"role": "user", "content": "📝 Summarize this PDF"})

# ─── MAIN AREA: CHAT INTERFACE ───────────────────────────────────
# WHY session_state for messages? Streamlit re-runs the script on every
# interaction. Without session_state, chat history would disappear
# every time you send a message.
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display all previous messages (chat history)
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box at the bottom of the page
user_question = st.chat_input("Ask a question about your PDF...")

# If user typed a question, add it to chat history
if user_question:
    st.session_state["messages"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)
    _pending_question = user_question

# ─── CALL BACKEND (works for both chat and summarize) ────────────
# WHY one shared block? Both "ask a question" and "summarize" do the same
# thing — send a question to /chat/ and show the answer. DRY principle.
if _pending_question:
    # Use top_k=30 for summaries/chapters (grab more chunks), 10 for normal questions
    q_lower = _pending_question.lower()
    needs_more_context = any(word in q_lower for word in ["summary", "summarize", "chapter", "section", "overview", "explain all", "entire", "whole"])
    top_k = 30 if needs_more_context else 10
    # Use balanced retrieval for broad queries so all documents are covered
    balanced = needs_more_context

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_BASE}/chat/",
                    json={"question": _pending_question, "top_k": top_k, "balanced": balanced},
                    timeout=120,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]

                    st.markdown(answer)

                    if data.get("sources"):
                        with st.expander("📚 Sources"):
                            for src in data["sources"]:
                                st.write(f"- Page {src['page_number']} (relevance: {1 - src['distance']:.0%})")
                else:
                    try:
                        detail = response.json().get('detail', response.text)
                    except Exception:
                        detail = response.text
                    answer = f"❌ Error: {detail}"
                    st.error(answer)

            except requests.ConnectionError:
                answer = "❌ Cannot connect to backend. Is the FastAPI server running?"
                st.error(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
