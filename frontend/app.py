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
    st.header("📁 Upload PDF")

    # st.file_uploader returns None if nothing uploaded, or a file-like object.
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],              # only allow PDFs
        accept_multiple_files=False,
    )

    # This button triggers the upload to our FastAPI backend
    if uploaded_file and st.button("🚀 Upload & Process", use_container_width=True):
        # WHY st.spinner? It shows a loading animation while we wait.
        # Without it, the user sees nothing and thinks the app is frozen.
        with st.spinner("Uploading and processing PDF..."):
            try:
                # Send the PDF to our FastAPI POST /documents/upload endpoint
                # 'files' is how you send multipart/form-data in requests
                response = requests.post(
                    f"{API_BASE}/documents/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    timeout=120,
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Processed! {result['total_pages']} pages → {result['total_chunks']} chunks")
                    # Save to session_state so we remember the upload even after re-runs
                    st.session_state["last_upload"] = result
                else:
                    st.error(f"❌ Upload failed: {response.json().get('detail', response.text)}")
            except requests.ConnectionError:
                st.error("❌ Cannot connect to backend. Is the FastAPI server running on port 8000?")

    # Show last upload info if available
    if "last_upload" in st.session_state:
        st.divider()
        st.subheader("Last Upload")
        info = st.session_state["last_upload"]
        st.write(f"**File:** {info['original_name']}")
        st.write(f"**Pages:** {info['total_pages']}")
        st.write(f"**Chunks:** {info['total_chunks']}")

# ─── MAIN AREA: CHAT INTERFACE ───────────────────────────────────
# WHY session_state for messages? Streamlit re-runs the script on every
# interaction. Without session_state, chat history would disappear
# every time you send a message.
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display all previous messages (chat history)
# WHY: After each re-run, we need to re-draw the entire chat history.
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):  # "user" or "assistant"
        st.markdown(msg["content"])

# Chat input box at the bottom of the page
# WHY st.chat_input? It gives a nice input bar pinned to the bottom,
# just like ChatGPT. Returns None if empty, or the text if user pressed Enter.
user_question = st.chat_input("Ask a question about your PDF...")

if user_question:
    # 1. Show the user's message immediately
    st.session_state["messages"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 2. Call our FastAPI backend and show the response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_BASE}/chat/",
                    json={"question": user_question, "top_k": 5},
                    timeout=120,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]

                    # Display the answer
                    st.markdown(answer)

                    # Show sources in a collapsible section
                    # WHY expander? Sources are useful but secondary info.
                    # Hiding them keeps the chat clean.
                    if data.get("sources"):
                        with st.expander("📚 Sources"):
                            for src in data["sources"]:
                                st.write(f"- Page {src['page_number']} (relevance: {1 - src['distance']:.0%})")
                else:
                    answer = f"❌ Error: {response.json().get('detail', response.text)}"
                    st.error(answer)

            except requests.ConnectionError:
                answer = "❌ Cannot connect to backend. Is the FastAPI server running?"
                st.error(answer)

    # 3. Save assistant response to chat history
    st.session_state["messages"].append({"role": "assistant", "content": answer})
