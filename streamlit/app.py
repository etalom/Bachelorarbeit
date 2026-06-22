import os
import json
from datetime import datetime

import httpx
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://rag-api:8000")
API_BASE = f"{API_URL}/api"

st.set_page_config(
    page_title="Chatbot-Vergabe",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
   html, body, [data-testid="stAppViewContainer"] { background: #ffffff !important; }



    .premium-header {
        background: linear-gradient(135deg, #1f4788 0%, #66f4ff  100%);
        color: white;
        padding: 28px 32px;
        border-radius: 0px;
        margin: -16px -16px 28px -16px;
        box-shadow: 0 4px 20px #b8e3e9 ;
    }

    .premium-header h1 {
        font-size: 2.5em;
        font-weight: 800;
        margin: 0 0 8px 0;
        letter-spacing: -1px;
        color: #ffffff !important;
    }

    .premium-header p {
        font-size: 1.05em;
        opacity: 0.95;
        margin: 0;
        font-weight: 300;
        letter-spacing: 0.3px;
    }

    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] { background: #b8e3e9 !important; }


    .api-status-box {
        background: white;
        border: 1px solid #e0e5ee;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .api-status-online { background: #e8f5e9; border-color: #2e7d32; }
    .api-status-offline { background: #ffebee; border-color: #c62828; }

    .api-status-label { font-size: 0.9em; font-weight: 600; color: #1a1a1a; }
    .api-status-badge { font-size: 1.2em; }

    .stButton > button {
        background: #1f4788 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        font-size: 0.9em !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 8px rgba(31, 71, 136, 0.15) !important;
    }
    .stButton > button:hover {
        background: #2d5aa8 !important;
        box-shadow: 0 4px 12px rgba(31, 71, 136, 0.25) !important;
        transform: translateY(-1px) !important;
    }

    .stTextInput > div > div > input,
    .stChatInput input {
        background: white !important;
        border: 1.5px solid #d8dfe6 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        font-size: 0.95em !important;
        transition: all 0.25s ease !important;
        color: #1a1a1a !important;
    }
    .stTextInput > div > div > input:focus,
    .stChatInput input:focus {
        border-color: #1f4788 !important;
        box-shadow: 0 0 0 3px rgba(31, 71, 136, 0.08) !important;
        outline: none !important;
    }
    

    .stChatMessage { background: transparent !important; }

    .metric-container {
        background: white;
        border: 1px solid #e0e5ee;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        transition: all 0.25s ease;
    }
    .metric-container:hover {
        border-color: #1f4788;
        box-shadow: 0 2px 8px rgba(31, 71, 136, 0.1);
    }
    .metric-number { font-size: 2.2em; font-weight: 700; color: #1f4788; margin: 8px 0; }
    .metric-label { font-size: 0.85em; color: #666; font-weight: 500; }

    .compare-header {
        background: linear-gradient(135deg, #f9a825 0%, #fa961a 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 16px;
        font-weight: 600;
        font-size: 0.95em;
    }
    .compare-header.hybrid { background: linear-gradient(135deg, #1f4788 0%, #2d5aa8 100%); }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="premium-header">
  <h1>⚖️ Chatbot</h1>
  <p>Ihr KI-gestützter Assistent</p>
</div>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "compare_history" not in st.session_state:
    st.session_state.compare_history = []

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def check_api_health() -> tuple[bool, dict]:
    """Prüft Health-Endpoint der API."""
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{API_BASE}/health")
            if r.status_code == 200:
                return True, r.json()
    except Exception:
        pass
    return False, {}


def render_meta(meta: dict, model: str = "") -> None:
    """Optionale Meta-Informationen (ohne Quellen)."""
    if not meta:
        return

    chips = []
    if meta.get("latency_ms") is not None:
        chips.append(f"🕐 {meta['latency_ms']}ms")
    if meta.get("score") is not None:
        chips.append(f"📊 Score: {meta['score']}")
    if meta.get("hits") is not None:
        chips.append(f"🔍 {meta['hits']} Treffer")
    if model:
        chips.append(f"🤖 {model}")

    if not chips:
        return

    chip_html = " ".join([f"<span class='metric-chip'>{c}</span>" for c in chips])
    st.markdown(f"<div class='meta-info'>{chip_html}</div>", unsafe_allow_html=True)


def export_chat() -> str:
    export_data = {
        "export_zeit": datetime.now().isoformat(),
        "nachrichten": st.session_state.messages,
    }
    return json.dumps(export_data, ensure_ascii=False, indent=2)


def call_ask_api(query: str, model: str, timeout: float = 300.0) -> dict:
    """Ruft /ask auf und liefert JSON."""
    with httpx.Client(timeout=timeout) as client:
        response = client.get(
            f"{API_BASE}/ask",
            params={"query": query, "model": model},
        )
        response.raise_for_status()
        return response.json()


def call_compare_api(query: str, model: str, timeout: float = 300.0) -> dict:
    """Ruft /compare auf und liefert JSON."""
    with httpx.Client(timeout=timeout) as client:
        response = client.get(
            f"{API_BASE}/compare",
            params={"query": query, "model": model},
        )
        response.raise_for_status()
        return response.json()


def render_chat_history(show_meta: bool) -> None:
    """Render den Chatverlauf."""
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "⚖️"

        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

            if message["role"] == "assistant" and show_meta:
                render_meta(message.get("meta", {}), message.get("model", ""))


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    is_healthy, health_data = check_api_health()
    status_class = "api-status-online" if is_healthy else "api-status-offline"
    status_text = "Online" if is_healthy else "Offline"
    status_icon = "✅" if is_healthy else "❌"

    st.markdown(
        f"""
<div class="api-status-box {status_class}">
  <span class="api-status-label"> API Status</span>
  <span class="api-status-badge">{status_icon} {status_text}</span>
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### 🤖 Modell")
    available_models = health_data.get("modelle", ["gpt-oss:20b", "mistral:latest"])
    selected_model = st.selectbox(
        "Auswählen:",
        options=available_models,
        index=0,
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### 🔧 Modus")
    app_mode = st.radio(
        "Ansicht:",
        options=["💬 Chat", "🔬 Modell-Vergleich"],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### ⚡ Einstellungen")
    show_meta = st.checkbox("Meta-Infos anzeigen", value=False)

    st.divider()

    st.markdown("### 🎯 Aktionen")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Verlauf", use_container_width=True):
            st.session_state.messages = []
            st.session_state.compare_history = []
            st.rerun()
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    if st.session_state.get("messages"):
        chat_pairs = []
        last_user = None
        for msg in st.session_state.messages:
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
            elif msg.get("role") == "assistant" and last_user is not None:
                chat_pairs.append({
                    "query": last_user,
                    "answer": msg.get("content", ""),
                    "model": msg.get("model", selected_model),
                    "quellen": msg.get("quellen", []),
                    "export_zeit": datetime.now().isoformat(),
                })
                last_user = None
        if chat_pairs:
            st.download_button(
                label="📥 Export (best_chunks_and_answer.json)",
                data=json.dumps({
                    "export_zeit": datetime.now().isoformat(),
                    "app": "Chatbot-Vergabe",
                    "run_name": "best_chunks_and_answer",
                    "entries": chat_pairs,
                }, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="best_chunks_and_answer.json",
                mime="application/json",
                use_container_width=True,
            )

    st.divider()

# ──────────────────────────────────────────────
# MAIN: CHAT
# ──────────────────────────────────────────────
if app_mode == "💬 Chat":
    st.divider()

    render_chat_history(show_meta=show_meta)

    user_query = st.chat_input("Stellen Sie Ihre Frage")
    if not user_query:
        st.stop()

    # User message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    # Assistant response
    with st.chat_message("assistant", avatar="⚖️"):
        placeholder = st.empty()
        placeholder.markdown("⏳ *Durchsuche Dokumente...*")

        try:
            data = call_ask_api(user_query, selected_model)

            answer = data.get("antwort", "Keine Antwort erhalten.")
            meta = data.get("meta", {})
            model = data.get("model", selected_model)

            # ✅ Quellen werden bewusst NICHT gerendert
            placeholder.markdown(answer)

            if show_meta:
                render_meta(meta, model)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "quellen": data.get("quellen", []),  # gespeichert, aber nicht gerendert
                    "meta": meta,
                    "model": model,
                }
            )

        except httpx.TimeoutException:
            placeholder.warning(
                "⏱️ **Timeout**: Das LLM wird möglicherweise noch geladen. "
                "Bitte warten Sie 30 Sekunden und versuchen Sie es erneut."
            )
        except httpx.ConnectError:
            placeholder.error(
                "❌ **Verbindung fehlgeschlagen**: API nicht erreichbar.\n\n"
                f"API URL: `{API_URL}`\n\n"
                "Prüfen Sie ob der rag-api Container läuft:\n"
                "`docker compose logs rag-api`"
            )
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else "?"
            body = e.response.text if e.response is not None else ""
            placeholder.error(f"❌ API Fehler: Status {status}\n\n{body}")
        except Exception as e:
            placeholder.error(f"❌ Unerwarteter Fehler: {type(e).__name__}: {str(e)}")

# ──────────────────────────────────────────────
# MAIN: MODEL COMPARISON
# ──────────────────────────────────────────────
elif app_mode == "🔬 Modell-Vergleich":
    st.title("BM25 vs. Vector Search")
    st.info("Vergleicht Keyword-Suche (BM25) und semantische Vektorsuche direkt nebeneinander.")

    compare_query = st.text_input(
        "Vergleichsfrage:",
        placeholder="z.B. Vergleichsangebote Schwellenwert",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        run_compare = st.button("🔍 Vergleichen", type="primary", use_container_width=True)

    if run_compare and compare_query:
        with st.spinner("⏳ Vergleiche BM25 vs. Vector..."):
            try:
                data = call_compare_api(compare_query, selected_model)

                bm25_data = data.get("bm25", {})
                vector_data = data.get("vector", {})
                analyse = data.get("analyse", {})

                st.markdown("### 📊 Analyse")
                m1, m2, m3, m4 = st.columns(4)

                with m1:
                    st.markdown(
                        "<div class='metric-container'>"
                        "<div class='metric-label'>Gemeinsam</div>"
                        f"<div class='metric-number'>{analyse.get('gemeinsame_dokumente', '?')}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        "<div class='metric-container'>"
                        "<div class='metric-label'>Nur BM25</div>"
                        f"<div class='metric-number'>{analyse.get('nur_in_bm25', '?')}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                with m3:
                    st.markdown(
                        "<div class='metric-container'>"
                        "<div class='metric-label'>Nur Vector</div>"
                        f"<div class='metric-number'>{analyse.get('nur_in_vector', '?')}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                with m4:
                    st.markdown(
                        "<div class='metric-container'>"
                        "<div class='metric-label'>Overlap %</div>"
                        f"<div class='metric-number'>{analyse.get('overlap_prozent', 0):.0f}%</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                st.divider()

                col_bm25, col_vector = st.columns(2, gap="medium")

                with col_bm25:
                    st.markdown(
                        "<div class='compare-header'>📝 BM25 (Keyword)</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"⏱️ {bm25_data.get('latency_ms','?')}ms | "
                        f"📊 {bm25_data.get('max_score','?')} | "
                        f"🔍 {bm25_data.get('hits','?')} Treffer"
                    )
                    st.markdown(bm25_data.get("answer", "Keine Antwort"))

                with col_vector:
                    st.markdown(
                        "<div class='compare-header hybrid'>🔀 Vector Search (Semantisch)</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"⏱️ {vector_data.get('latency_ms','?')}ms | "
                        f"📊 {vector_data.get('max_score','?')} | "
                        f"🔍 {vector_data.get('hits','?')} Treffer"
                    )
                    st.markdown(vector_data.get("answer", "Keine Antwort"))

                st.session_state.compare_history.append(
                    {
                        "zeit": datetime.now().strftime("%H:%M:%S"),
                        "query": compare_query,
                        "model": selected_model,
                        "analyse": analyse,
                        "bm25_score": bm25_data.get("max_score"),
                        "vector_score": vector_data.get("max_score"),
                    }
                )

            except httpx.TimeoutException:
                st.warning("⏱️ Timeout - API antwortet zu langsam")
            except httpx.ConnectError:
                st.error(f"❌ API nicht erreichbar: {API_URL}")
            except httpx.HTTPStatusError as e:
                st.error(f"❌ API Fehler: {e.response.status_code}\n\n{e.response.text}")
            except Exception as e:
                st.error(f"❌ Fehler: {type(e).__name__}: {str(e)}")

    if st.session_state.compare_history:
        st.divider()
        st.markdown("### 📈 Historie")

        df = pd.DataFrame(st.session_state.compare_history)
        display_df = df[["zeit", "query", "bm25_score", "vector_score"]].copy()
        display_df.columns = ["Zeit", "Frage", "BM25", "Vector"]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.divider()
        st.download_button(
            label="📥 Vergleiche exportieren",
            data=json.dumps(st.session_state.compare_history, ensure_ascii=False, indent=2),
            file_name=f"vergleich_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )

