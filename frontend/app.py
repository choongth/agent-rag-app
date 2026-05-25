import json
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="SmartDoc Assistant", page_icon="📚", layout="wide")
st.title("📚 SmartDoc Assistant")
st.caption("Upload documents · Ask questions · Get cited answers — powered by DeepSeek + RAG")


# ── Sidebar: document management ────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Documents")

    uploaded = st.file_uploader(
        "Upload a document", type=["pdf", "txt", "md"], label_visibility="collapsed"
    )
    if uploaded:
        if st.button("Index document", use_container_width=True):
            with st.spinner(f"Indexing {uploaded.name}…"):
                resp = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                )
            if resp.ok:
                st.success(resp.json()["message"])
                st.rerun()
            else:
                st.error(f"Upload failed: {resp.text}")

    st.divider()
    st.subheader("Indexed documents")

    try:
        docs: list[str] = requests.get(f"{API_URL}/documents", timeout=5).json()[
            "documents"
        ]
    except Exception as e:
        docs = []
        st.warning(f"Cannot reach backend: {e}")

    if docs:
        for doc in docs:
            col_name, col_btn = st.columns([5, 1])
            col_name.markdown(f"📄 `{doc}`")
            if col_btn.button("🗑", key=f"del_{doc}", help=f"Remove {doc}"):
                r = requests.delete(f"{API_URL}/documents/{doc}")
                if r.ok:
                    st.success(r.json()["message"])
                    st.rerun()
                else:
                    st.error(r.text)
    else:
        st.info("No documents indexed yet.\nUpload a PDF, TXT, or MD file to get started.")

    if st.button("🔄 Refresh list", use_container_width=True):
        st.rerun()


# ── Chat area ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New user input
def _stream_tokens(url: str, payload: dict, history_out: list):
    """SSE generator that yields text tokens and captures history via history_out."""
    with requests.post(url, json=payload, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event["type"] == "token":
                    yield event["content"]
                elif event["type"] == "done":
                    history_out.append(event["history"])


if prompt := st.chat_input("Ask anything about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            history_out: list = []
            payload = {"message": prompt, "history": st.session_state.history}
            gen = _stream_tokens(f"{API_URL}/chat/stream", payload, history_out)
            answer = st.write_stream(gen)  # streams tokens live, returns full text
            if history_out:
                st.session_state.history = history_out[0]
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the backend. Is it running?")
        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.messages:
    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()
