import streamlit as st
import requests
import base64

API_URL = "http://127.0.0.1:8000/ask"
BLANK_AVATAR = (
    "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
)

st.set_page_config(
    page_title="NSB-AI Assistant", page_icon="./media/nsb-logo.png", layout="centered"
)


def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


logo = get_base64_image("./media/nsb-logo.png")

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #1f2937; }

.nsb-header {
    background: linear-gradient(135deg, #0b1f3a, #0d47a1);
    padding: 20px 24px;
    border-radius: 14px;
    margin-bottom: 24px;
    border: 1px solid #1e3a8a;
}
.nsb-header h1 { color: white; font-size: 2rem; margin: 0; font-weight: 700; }
.nsb-header p { color: #b3d4fc; font-size: 1rem; margin-top: 6px; }

[data-testid="stChatMessage"] {
    background-color: #161b22;
    border-radius: 14px;
    padding: 10px 14px;
    border: 1px solid #1f2937;
    margin-bottom: 10px;
}
[data-testid="stChatInput"] {
    background-color: #161b22 !important;
    border: 1px solid #2a5298 !important;
    border-radius: 18px !important;
    padding: 6px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: black !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea:focus { outline: none !important; box-shadow: none !important; }
[data-testid="stChatInputSubmitButton"] { background-color: #2a5298 !important; border-radius: 12px !important; }
[data-testid="stChatInputSubmitButton"]:hover { background-color: #1d4b8f !important; }

.stMarkdown, .stText, p, li, label { color: #f3f4f6 !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #2a5298; border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="nsb-header">
    <div style="display:flex; align-items:center; gap:16px;">
        <img src="data:image/png;base64,{logo}" width="56"/>
        <div>
            <h1>NSB-AI Assistant</h1>
            <p>National Signals Bureau</p>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    if st.button("Clear Chat", use_container_width=True, type="primary"):
        st.session_state.chat = []
        st.rerun()

    st.divider()

    st.markdown("""
**NSB-AI** answers strictly from official NSB documents.

- No external knowledge
- No guessing
- Internal use only
""")

    st.divider()

    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=2)

        if r.status_code == 200:
            st.success("API Online 🟢")
        else:
            st.error("API Offline 🔴")

    except Exception:
        st.error("API Offline 🔴")

if "chat" not in st.session_state:
    st.session_state.chat = []

if not st.session_state.chat:
    with st.chat_message("assistant", avatar=BLANK_AVATAR):
        st.markdown("Hello, I'm **NSB-AI**. Ask me anything.")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"], avatar=BLANK_AVATAR):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.chat.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=BLANK_AVATAR):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=BLANK_AVATAR):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")
        answer = ""
        first_chunk = True

        try:
            with requests.post(
                API_URL, json={"question": user_input}, stream=True, timeout=600
            ) as r:
                for chunk in r.iter_content(chunk_size=None):
                    if chunk:
                        if first_chunk:
                            placeholder.empty()
                            first_chunk = False
                        answer += chunk.decode("utf-8")
                        placeholder.markdown(answer)
        except requests.exceptions.Timeout:
            answer = "Request timed out. Try again."
            placeholder.markdown(answer)
        except Exception as e:
            answer = f"Connection error: {str(e)}"
            placeholder.markdown(answer)

    st.session_state.chat.append({"role": "assistant", "content": answer})
