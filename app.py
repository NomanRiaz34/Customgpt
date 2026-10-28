import streamlit as st
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

st.set_page_config(page_title="NOMAN'S ChatGPT", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
:root { color-scheme: dark; }
html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; }
.stApp { background: radial-gradient(ellipse at 75% -20%,rgba(109,77,221,.13),transparent 42%),#0B0E16; color:#F4F4FC; }
.block-container { max-width:990px; padding-top:2rem; padding-bottom:7rem; }
h1,h2,h3 { font-family:'Outfit',sans-serif; letter-spacing:-.035em; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#111522; border-right:1px solid #252C40; }
[data-testid="stSidebar"] .block-container { padding-top:1.5rem; }
[data-testid="stSidebar"] h3 { font-size:.85rem; color:#9EAAC4; letter-spacing:.06em; text-transform:uppercase; }
[data-testid="stSidebar"] button { background:#1A2132; border:1px solid #30394F; color:#EDEFFC; border-radius:12px; transition:background .2s,transform .2s,border-color .2s; }
[data-testid="stSidebar"] button:hover { background:#27314A; border-color:#8670E9; transform:translateY(-1px); color:white; }
button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] { background:linear-gradient(110deg,#7758F5,#516CF3)!important; border:0!important; color:white!important; box-shadow:0 6px 26px #654be042; transition: transform .2s,filter .2s; }
button[kind="primary"]:hover { filter:brightness(1.12); transform:translateY(-2px); }
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea { background:#1C2334!important; color:#F5F6FF!important; border:1px solid #303A52!important; border-radius:12px!important; }
[data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus { border-color:#8973FF!important; }
[data-testid="stChatInput"] { background:#1D2535; border:1px solid #3B4560; border-radius:18px; box-shadow:0 12px 35px #0004; }
[data-testid="stChatInput"]:focus-within { border-color:#927DFF; box-shadow:0 0 0 3px #8066ee20; }
[data-testid="stChatInput"] textarea { color:#F2F3FA; }
[data-testid="stChatMessage"] { background:#151B29; border:1px solid #283149; border-radius:18px; padding:1rem; margin-bottom:.85rem; animation:slideUp .35s ease both; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background:#20223B; border-color:#34355E; }
[data-testid="stChatMessage"] p { line-height:1.7; }
[data-testid="stForm"] { background:rgba(23,29,46,.85); border:1px solid #333D57; border-radius:22px; padding:1.8rem; box-shadow:0 20px 70px #0005; }
hr { border-color:#293248!important; }
.brand { font-family:'Outfit',sans-serif; font-size:1.4rem; font-weight:800; letter-spacing:-.035em; color:#F8F8FF; }
.brand span { background:linear-gradient(120deg,#B9A9FF,#76C6FF); -webkit-background-clip:text; background-clip:text; color:transparent; }
.eyebrow { color:#A698FF; letter-spacing:.16em; text-transform:uppercase; font-size:.72rem; font-weight:700; }
.hero { text-align:center; padding:9vh 0 2.6rem; animation:slideUp .7s ease both; }
.hero-icon { display:inline-flex; width:76px; height:76px; align-items:center; justify-content:center; border-radius:24px; font-size:2.4rem; background:linear-gradient(140deg,#413272,#252B54); border:1px solid #6153AB; box-shadow:0 0 48px #8868f34b; animation:float 5s ease-in-out infinite; }
.hero h1 { font-size:clamp(2.1rem,4.5vw,3.2rem); margin:1.3rem 0 .35rem; color:#F9F8FF; }
.hero p { color:#9FAAC2; font-size:1.02rem; }
.subtle { color:#93A0BB; font-size:.9rem; }
.status-pill { display:inline-block; background:#163128; border:1px solid #245946; color:#86E8C3; padding:.35rem .75rem; border-radius:100px; font-size:.76rem; font-weight:600; }
.login-header { text-align:center; margin:1rem 0 2rem; animation:slideUp .7s ease both; }
.login-header h1 { font-size:2.4rem; margin:.7rem 0; }
.login-header p { color:#A4AEC6; }
.login-logo { display:inline-flex; justify-content:center; align-items:center; height:78px; width:78px; background:linear-gradient(135deg,#795BF0,#3B6DF0); box-shadow:0 0 50px #785cf370; border-radius:23px; font-size:2.3rem; animation:float 4s ease-in-out infinite; }
.footer-text { text-align:center; color:#687794; font-size:.75rem; margin-top:1rem; }
@keyframes float { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-8px); } }
@keyframes slideUp { from { opacity:0; transform:translateY(18px); } to { opacity:1; transform:translateY(0); } }
@media (prefers-reduced-motion:reduce) { *, *::before, *::after { animation:none!important; transition:none!important; } }
@media (max-width:600px) { .block-container {padding-top:1rem;} .hero {padding-top:4vh;} .login-header h1 {font-size:1.8rem;} }
</style>
""", unsafe_allow_html=True)

# Session-only memory. A browser session reset can clear conversations.
def initialize():
    defaults = {
        "api_key": None,
        "authenticated": False,
        "chats": {},
        "current_chat": None,
        "chat_counter": 0,
        "system_role": "You are a helpful, clear, and friendly AI assistant.",
        "pending_prompt": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def new_chat():
    st.session_state.chat_counter += 1
    chat_id = str(st.session_state.chat_counter)
    st.session_state.chats[chat_id] = {"title": "New conversation", "messages": []}
    st.session_state.current_chat = chat_id


def logout():
    # Discard all session-held API credentials and conversations.
    for key in ("api_key", "authenticated", "chats", "current_chat", "chat_counter", "pending_prompt"):
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


initialize()

# --------------------------------------------------
# Login view
# --------------------------------------------------
if not st.session_state.authenticated:
    st.markdown("""
    <style>
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display:none!important; }
    .block-container { max-width:510px; padding-top:9vh; }
    </style>
    <div class="login-header">
        <div class="login-logo">✨</div>
        <h1>Welcome to <span style="color:#A99BFF">NOMAN'S AI</span></h1>
        <p>Your creative workspace, powered by intelligence.</p>
        <span class="status-pill">🔐 Private API connection</span>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        st.markdown("#### 🚀 Get started")
        st.caption("Enter your OpenAI API key to unlock your assistant.")
        key = st.text_input("🔑 OpenAI API key", type="password", placeholder="sk-...", autocomplete="off")
        submit = st.form_submit_button("✨ Connect to your assistant", use_container_width=True, type="primary")

    st.markdown('<p class="footer-text">Your key is kept in this Streamlit session, not written to a file.</p>', unsafe_allow_html=True)

    if submit:
        key = key.strip()
        if not key:
            st.warning("Please enter your API key.")
        else:
            with st.spinner("🔄 Checking your connection..."):
                try:
                    OpenAI(api_key=key).models.list()
                    st.session_state.api_key = key
                    st.session_state.authenticated = True
                    st.rerun()
                except AuthenticationError:
                    st.error("This key was rejected by OpenAI. Check or regenerate it.")
                except APIConnectionError:
                    st.error("Connection failed. Check your internet connection.")
                except Exception:
                    st.error("Could not validate the key right now. Please try again.")
    st.stop()

# --------------------------------------------------
# Sidebar and conversation picker
# --------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand">✨ NOMAN\'S <span>AI</span></div><div class="subtle">Your personal AI workspace</div>', unsafe_allow_html=True)
    st.write("")
    if st.button("➕  New conversation", type="primary", use_container_width=True):
        new_chat()
        st.rerun()

    st.markdown("### 💬 Conversations")
    for chat_id in reversed(list(st.session_state.chats)):
        title = st.session_state.chats[chat_id]["title"]
        prefix = "🟣" if chat_id == st.session_state.current_chat else "💬"
        if st.button(f"{prefix}  {title}", key=f"chat_{chat_id}", use_container_width=True):
            st.session_state.current_chat = chat_id
            st.rerun()

    st.divider()
    with st.expander("⚙️ Assistant settings"):
        st.text_area("System role", key="system_role", height=120)
        st.caption("Applied to your next message.")
        if st.button("🗑️ Delete current chat", use_container_width=True):
            active = st.session_state.current_chat
            if active in st.session_state.chats:
                del st.session_state.chats[active]
            st.session_state.current_chat = next(reversed(st.session_state.chats), None) if st.session_state.chats else None
            st.rerun()

    st.markdown('<span class="status-pill">● API connected</span>', unsafe_allow_html=True)
    if st.button("🚪 Log out", use_container_width=True):
        logout()

if st.session_state.current_chat is None or st.session_state.current_chat not in st.session_state.chats:
    new_chat()

active_id = st.session_state.current_chat
messages = st.session_state.chats[active_id]["messages"]

# --------------------------------------------------
# Chat header and welcome screen
# --------------------------------------------------
header_left, header_right = st.columns([4, 1], vertical_alignment="center")
with header_left:
    st.markdown('<div class="brand">✨ NOMAN\'S <span>ChatGPT</span></div>', unsafe_allow_html=True)
with header_right:
    st.caption("⚡ GPT-5.2")
st.divider()

if not messages:
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">🧠</div>
        <div class="eyebrow" style="margin-top:1.3rem">YOUR PERSONAL AI ASSISTANT</div>
        <h1>What can we create today?</h1>
        <p>Code, brainstorm, learn, and turn ideas into reality. ✨</p>
    </div>
    """, unsafe_allow_html=True)
    prompts = [
        ("💻  Write code", "Help me write a clean Python program."),
        ("💡  Brainstorm", "Give me five creative project ideas."),
        ("📚  Explain a topic", "Explain Python decorators with simple examples."),
        ("✍️  Improve writing", "Help me write a clear professional email."),
    ]
    for row_start in (0, 2):
        cols = st.columns(2, gap="small")
        for col, (label, prompt_text) in zip(cols, prompts[row_start:row_start + 2]):
            with col:
                if st.button(label, key=f"suggest_{row_start}_{label}", use_container_width=True):
                    st.session_state.pending_prompt = prompt_text
                    st.rerun()
else:
    for message in messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="✨"):
                st.markdown(message.content)

# --------------------------------------------------
# Chat composer and streamed response
# --------------------------------------------------
entered_prompt = st.chat_input("✨ Ask NOMAN'S AI anything...")
prompt = st.session_state.pending_prompt or entered_prompt
st.session_state.pending_prompt = None

if prompt and prompt.strip():
    prompt = prompt.strip()
    messages.append(HumanMessage(content=prompt))
    if len(messages) == 1:
        st.session_state.chats[active_id]["title"] = (prompt[:27] + "…") if len(prompt) > 27 else prompt

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="✨"):
        try:
            model = ChatOpenAI(model="gpt-5.2", api_key=st.session_state.api_key)
            conversation = [SystemMessage(content=st.session_state.system_role or "You are a helpful assistant.")] + messages

            def token_stream():
                for chunk in model.stream(conversation):
                    if isinstance(chunk.content, str):
                        yield chunk.content
                    elif isinstance(chunk.content, list):
                        for part in chunk.content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                yield part.get("text", "")

            with st.spinner("✨ Thinking..."):
                answer = st.write_stream(token_stream())
            messages.append(AIMessage(content=answer))
        except AuthenticationError:
            st.error("Your API key is no longer valid. Please log out and reconnect.")
        except RateLimitError:
            st.error("Your OpenAI API account hit a quota or rate limit. Check billing and usage.")
        except APIConnectionError:
            st.error("Network connection failed. Please retry.")
        except Exception:
            st.error("Couldn't generate a response. Check model access or your API account.")

st.markdown('<p class="footer-text">✨ NOMAN\'S AI can make mistakes. Verify important information.</p>', unsafe_allow_html=True)
