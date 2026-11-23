"""Streamlit front-end for the Technical Manual & Datasheet Assistant.

Local:   streamlit run apps/streamlit_app.py
Cloud:   Streamlit Community Cloud, main file = apps/streamlit_app.py (see docs/deployment.md).

Secrets / env (all optional):
  ANTHROPIC_API_KEY, ANTHROPIC_MODEL  -> Claude backend (the only LLM option on Streamlit Cloud)
  OLLAMA_HOST, OLLAMA_MODEL           -> local Ollama (only works when Ollama is reachable)
  APP_PASSWORD                        -> if set, visitors must enter it (protects your API quota)
Without any LLM the app falls back to the extractive backend (verbatim passages, no synthesis).
"""
from __future__ import annotations

import os

import streamlit as st

# Streamlit exposes top-level secrets as env vars on Cloud, but copy them explicitly for local secrets.toml too.
try:
    for _k in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "OLLAMA_HOST", "OLLAMA_MODEL", "APP_PASSWORD",
               "DATASHEET_ASSISTANT_ROOT"):
        if _k in st.secrets and _k not in os.environ:
            os.environ[_k] = str(st.secrets[_k])
except Exception:  # no secrets file configured
    pass

from assistant.generate import make_backend  # noqa: E402
from assistant.pipeline import DATA_DIR, INDEX, Assistant  # noqa: E402
from assistant.retrieval import Retriever  # noqa: E402
from assistant import ingest as ing  # noqa: E402

st.set_page_config(page_title="Datasheet Assistant", page_icon="📟", layout="centered")

EXAMPLES = [
    "What is the I2C address of the BMP280 when SDO is connected to VDDIO?",
    "What temperature resolution does the MAX31855 provide?",
    "Can the DS18B20 be powered from the data line?",
    "What supply voltage does the AM2303 need?",
    "What is the capital of France?",
]


@st.cache_resource(show_spinner="Loading documentation index…")
def load_retriever() -> Retriever:
    if not INDEX.exists():  # needs poppler-utils + the PDFs; normally the index is committed
        ing.save(ing.ingest(DATA_DIR), INDEX)
    return Retriever(ing.load(INDEX))


@st.cache_resource
def load_backend(name: str):
    return make_backend(name)


def password_gate() -> None:
    pw = os.environ.get("APP_PASSWORD")
    if not pw or st.session_state.get("authed"):
        return
    entered = st.text_input("Password", type="password")
    if entered and entered == pw:
        st.session_state.authed = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    st.stop()


def main() -> None:
    st.title("📟 Technical Manual & Datasheet Assistant")
    st.caption("Answers from the sensor/instrument datasheets in the knowledge base, with document and page citations.")
    password_gate()

    with st.sidebar:
        choices = ["auto", "anthropic", "ollama", "extractive"]
        name = st.selectbox("Backend", choices, help="auto = Claude if ANTHROPIC_API_KEY is set, else Ollama, else extractive")
        rag = st.toggle("Use retrieval (RAG)", value=True,
                        help="Off = baseline: the model answers from memory with no excerpts (expect confident wrong specs).")
        if st.button("Clear conversation"):
            st.session_state.pop("assistant", None)
            st.session_state.pop("messages", None)
            st.rerun()
        st.markdown("**Try:**")
        for ex in EXAMPLES:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state.pending = ex

    try:
        backend = load_backend(name)
    except KeyError:
        st.error(f"Backend '{name}' needs an environment variable that is not set (e.g. ANTHROPIC_API_KEY).")
        st.stop()
    if not rag and backend.name == "extractive":
        st.warning("The extractive backend has no model, so it can't run without retrieval. Turn RAG back on.")
        st.stop()
    st.sidebar.caption(f"Active backend: **{backend.name}**")
    if backend.name == "extractive":
        st.sidebar.info("No LLM configured: showing verbatim passages only.")

    # One Assistant per browser session: it holds that visitor's chat history; the index/backend are shared.
    sess = st.session_state.get("assistant")
    if sess is None or sess.backend is not backend:
        sess = st.session_state.assistant = Assistant(load_retriever(), backend)
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("Sources"):
                    st.markdown("\n".join(f"- {s}" for s in m["sources"]))

    q = st.session_state.pop("pending", None) or st.chat_input("Ask about a sensor or instrument…")
    if not q:
        return
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documentation…"):
                ans = sess.ask(q, remember=True, rag=rag)
        except Exception as e:  # network / quota / model errors from the backend
            st.error(f"The {backend.name} backend failed: {e}")
            st.session_state.messages.pop()
            return
        st.markdown(ans.text)
        if ans.sources:
            with st.expander("Sources"):
                st.markdown("\n".join(f"- {s}" for s in ans.sources))
    st.session_state.messages.append({"role": "assistant", "content": ans.text, "sources": ans.sources})


main()
