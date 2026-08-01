"""
Streamlit Chatbot UI — ChatGPT/Claude/Perplexity-inspired frontend.

IMPORTANT: All backend logic below (model init, message list, model.invoke call,
memory handling via `messages`) is UNCHANGED from the original script.
Only the Streamlit presentation layer has been added/redesigned.
"""

import time
import streamlit as st
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# =========================================================================
# BACKEND CONFIG (UNCHANGED) — model, system prompt, memory structure
# =========================================================================

MODEL_NAME = "mistral-small-2506"
TEMPERATURE = 0.9

SYSTEM_PROMPT = "You are a funny AI agent"


@st.cache_resource(show_spinner=False)
def get_model():
    """Instantiate the LLM exactly as in the original script (cached across reruns)."""
    return ChatMistralAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )


model = get_model()


# =========================================================================
# PAGE CONFIG
# =========================================================================

st.set_page_config(
    page_title="FunnyBot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================================
# GLOBAL STYLES — gradients, rounded cards, bubbles, glassmorphism
# =========================================================================

def inject_custom_css() -> None:
    """Injects all custom CSS for the modern chatbot dark theme."""
    is_dark = True
    app_bg = "#020617"
    sidebar_bg = "rgba(15,23,42,0.92)"
    card_bg = "rgba(15,23,42,0.88)"
    card_border = "rgba(148,163,184,0.16)"
    text_color = "#e2e8f0"
    subtext_color = "#94a3b8"
    soft_shadow = "0 24px 64px rgba(15,23,42,0.15)"

    st.markdown(
        f"""
        <style>
        :root {{
            color-scheme: {'dark' if is_dark else 'light'};
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        body {{
            background: radial-gradient(circle at top left, rgba(99,102,241,0.18), transparent 18%),
                        radial-gradient(circle at bottom right, rgba(168,85,247,0.16), transparent 20%),
                        linear-gradient(180deg, {app_bg} 0%, {'#020617' if is_dark else '#eff6ff'} 100%);
            color: {text_color};
        }}

        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 4rem;
            max-width: 1400px;
        }}

        .main {{
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }}

        section[data-testid="stSidebar"] {{
            background: {sidebar_bg};
            box-shadow: 0 24px 80px rgba(15,23,42,0.18);
            border-left: 1px solid rgba(255,255,255,0.08);
            color: {text_color};
        }}

        .sidebar-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 18px;
            padding: 1.2rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: {soft_shadow};
            color: {text_color};
        }}

        .sidebar-logo {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .sidebar-logo .icon {{
            display: grid;
            place-items: center;
            width: 3rem;
            height: 3rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #6366f1, #a855f7);
            color: white;
            font-size: 1.5rem;
        }}

        .sidebar-title {{
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
            color: {text_color};
        }}

        .sidebar-subtitle {{
            font-size: 0.82rem;
            color: {subtext_color};
            margin: 0;
        }}

        .sidebar-note {{
            font-size: 0.82rem;
            color: {subtext_color};
            line-height: 1.55;
        }}

        .sidebar-section-title {{
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
            color: {text_color};
        }}

        .sidebar-metric {{
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: center;
            margin-bottom: 0.8rem;
            padding: 0.85rem 0;
            border-bottom: 1px solid rgba(148,163,184,0.12);
            color: {text_color};
        }}

        .sidebar-metric:last-child {{
            border-bottom: none;
            margin-bottom: 0;
        }}

        .sidebar-metric-key {{ font-size: 0.85rem; color: {subtext_color}; }}
        .sidebar-metric-value {{ font-size: 0.95rem; font-weight: 700; color: {text_color}; }}

        .hero-panel {{
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(15,23,42,0.08);
            backdrop-filter: blur(18px);
            border-radius: 24px;
            padding: 1.75rem 2rem;
            box-shadow: {soft_shadow};
            margin-bottom: 1.5rem;
        }}

        .hero-panel.dark {{
            background: rgba(255,255,255,0.92);
            border-color: rgba(148,163,184,0.16);
        }}

        .hero-grid {{
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 1.5rem;
            align-items: center;
        }}

        .hero-copy h1 {{
            margin: 0;
            font-size: clamp(2rem, 2.4vw, 3rem);
            line-height: 1.05;
            letter-spacing: -0.03em;
            color: #0f172a;
        }}

        .hero-copy p {{
            margin: 1rem 0 0;
            font-size: 1rem;
            line-height: 1.75;
            color: #475569;
        }}

        .hero-pill-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-top: 1.5rem;
        }}

        .hero-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.7rem 1rem;
            border-radius: 999px;
            background: #e0e7ff;
            color: #3730a3;
            font-weight: 700;
            font-size: 0.88rem;
        }}

        .hero-pill.status {{
            background: #dcfce7;
            color: #166534;
        }}

        .hero-pill.time {{
            background: #fef3c7;
            color: #92400e;
        }}

        .hero-visual {{
            display: grid;
            place-items: center;
            min-height: 240px;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,250,252,0.95));
            border: 1px solid rgba(148,163,184,0.16);
        }}

        .hero-visual-inner {{
            display: grid;
            place-items: center;
            width: 100%;
            height: 100%;
            text-align: center;
            color: #0f172a;
        }}

        .hero-visual-icon {{
            font-size: 3.2rem;
            margin-bottom: 0.85rem;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}

        .metric-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 20px;
            padding: 1.15rem 1.25rem;
            box-shadow: {soft_shadow};
            min-height: 124px;
        }}

        .metric-card h3 {{
            margin: 0;
            font-size: 0.95rem;
            color: {subtext_color};
            font-weight: 600;
        }}

        .metric-card p {{
            margin: 0.8rem 0 0;
            font-size: 1.45rem;
            font-weight: 700;
            color: {text_color};
        }}

        .metric-icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.5rem;
            height: 2.5rem;
            border-radius: 16px;
            background: rgba(99,102,241,0.14);
            color: #4338ca;
            margin-bottom: 0.85rem;
            font-size: 1.1rem;
        }}

        .chat-panel {{
            background: transparent;
            padding: 0;
        }}

        .chat-message-wrapper {{
            border-radius: 24px;
            border: 1px solid rgba(15,23,42,0.08);
            background: {card_bg};
            padding: 1rem;
            box-shadow: {soft_shadow};
            margin-bottom: 1rem;
        }}

        .chat-meta {{
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
            color: {subtext_color};
            font-size: 0.85rem;
            margin-bottom: 0.85rem;
        }}

        .chat-meta strong {{ color: {text_color}; }}

        .message-actions {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.95rem;
        }}

        .message-actions button {{
            border-radius: 999px;
            border: 1px solid rgba(99,102,241,0.16);
            background: rgba(99,102,241,0.08);
            color: #4338ca;
            padding: 0.55rem 0.9rem;
            cursor: pointer;
            transition: all 0.18s ease;
            font-size: 0.9rem;
        }}

        .message-actions button:hover {{
            transform: translateY(-1px);
            background: rgba(99,102,241,0.14);
        }}

        .suggestion-card {{
            border-radius: 20px;
            background: {card_bg};
            border: 1px solid {card_border};
            padding: 1.1rem 1.15rem;
            box-shadow: {soft_shadow};
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}

        .suggestion-card:hover {{
            transform: translateY(-3px);
            border-color: rgba(99,102,241,0.28);
        }}

        .suggestion-card h4 {{
            margin: 0;
            font-size: 1rem;
            color: {text_color};
        }}

        .suggestion-card p {{
            margin: 0.65rem 0 0;
            color: {subtext_color};
            font-size: 0.92rem;
            line-height: 1.6;
        }}

        .suggestion-button {{
            width: 100%;
            border-radius: 16px;
            background: linear-gradient(135deg, #6366f1, #a855f7);
            border: none;
            color: white;
            padding: 0.85rem 0;
            font-weight: 700;
            letter-spacing: 0.01em;
            transition: transform 0.2s ease;
        }}

        .suggestion-button:hover {{
            transform: translateY(-1px);
        }}

        .footer-note {{
            font-size: 0.82rem;
            color: {subtext_color};
            text-align: center;
            margin-top: 1rem;
        }}

        .stButton > button {{
            border-radius: 14px;
            box-shadow: none;
            transition: all 0.2s ease;
            color: {text_color} !important;
            background: {'rgba(255,255,255,0.08)' if is_dark else '#eef2ff'} !important;
            border: 1px solid {'rgba(255,255,255,0.12)' if is_dark else 'rgba(99,102,241,0.16)'} !important;
        }}

        .stButton > button:hover {{
            transform: translateY(-1px);
            background: {'rgba(255,255,255,0.12)' if is_dark else '#dbeafe'} !important;
        }}

        .stMarkdown {{
            color: {text_color} !important;
        }}

        .stTextArea > div > textarea {{
            border-radius: 18px !important;
            padding: 0.95rem !important;
            min-height: 90px;
            color: {text_color} !important;
            background: {'#111827' if is_dark else '#ffffff'} !important;
        }}

        .stChatMessage, .stChatMessage * {{
            color: {text_color} !important;
        }}

        .stChatMessage {{
            border-radius: 22px !important;
            padding: 1rem 1rem !important;
            margin-bottom: 1rem !important;
        }}

        .typing-dots span {{
            animation: blink 1.4s infinite both;
            font-size: 1.3rem;
        }}
        .typing-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
        .typing-dots span:nth-child(3) {{ animation-delay: 0.4s; }}
        @keyframes blink {{
            0%, 80%, 100% {{ opacity: 0.18; }}
            40% {{ opacity: 1; }}
        }}

        .scroll-container {{
            scroll-behavior: smooth;
        }}

        @media (max-width: 900px) {{
            .hero-grid {{ grid-template-columns: 1fr; }}
            .metrics-grid {{ grid-template-columns: 1fr; }}
        }}
        @media (max-width: 700px) {{
            section[data-testid="stSidebar"] {{ display: none; }}
            .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# SESSION STATE INITIALIZATION (backend memory structure UNCHANGED)
# =========================================================================

def init_session_state() -> None:
    """Sets up st.session_state.messages exactly like the original `messages` list."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=SYSTEM_PROMPT),
        ]
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Dark"
    if "session_start" not in st.session_state:
        st.session_state.session_start = time.time()
    if "last_response_time" not in st.session_state:
        st.session_state.last_response_time = 0.0
    if "message_timestamps" not in st.session_state:
        st.session_state.message_timestamps = [time.time()]


def clear_chat() -> None:
    """Resets conversation back to just the system prompt."""
    st.session_state.messages = [
        SystemMessage(content=SYSTEM_PROMPT),
    ]
    st.session_state.message_timestamps = [time.time()]
    st.session_state.session_start = time.time()
    st.success("Conversation cleared! ✨")


def count_exchanges() -> tuple[int, int]:
    """Returns (user_message_count, ai_message_count) for sidebar stats."""
    user_count = sum(1 for m in st.session_state.messages if isinstance(m, HumanMessage))
    ai_count = sum(1 for m in st.session_state.messages if isinstance(m, AIMessage))
    return user_count, ai_count


def estimate_tokens() -> int:
    """Estimate token usage based on current conversation text."""
    words = sum(len(m.content.split()) for m in st.session_state.messages)
    return max(0, int(words * 0.85))


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    hours = int(minutes // 60)
    if hours:
        return f"{hours}h {minutes % 60}m"
    return f"{minutes}m {int(seconds % 60)}s"


def format_timestamp(index: int) -> str:
    if index < len(st.session_state.message_timestamps):
        ts = st.session_state.message_timestamps[index]
        return datetime.fromtimestamp(ts).strftime("%b %d • %I:%M %p")
    return datetime.now().strftime("%b %d • %I:%M %p")


# =========================================================================
# SIDEBAR
# =========================================================================

def render_sidebar() -> None:
    """Renders the premium sidebar with controls, stats, and export actions."""
    with st.sidebar:
        st.markdown(
            """
            <div class='sidebar-card'>
                <div class='sidebar-logo'>
                    <div class='icon'>🤖</div>
                    <div>
                        <div class='sidebar-title'>FunnyBot AI</div>
                        <p class='sidebar-subtitle'>Premium conversational assistant</p>
                    </div>
                </div>
                <p class='sidebar-note'>A polished AI companion built for professional conversations, knowledge work, and creative brainstorming.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='sidebar-card'>
                <div class='sidebar-section-title'>Profile</div>
                <div class='sidebar-metric'><span class='sidebar-metric-key'>User</span><span class='sidebar-metric-value'>Guest</span></div>
                <div class='sidebar-metric'><span class='sidebar-metric-key'>Role</span><span class='sidebar-metric-value'>AI Assistant</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='sidebar-card'>
                <div class='sidebar-section-title'>Session Overview</div>
            """,
            unsafe_allow_html=True,
        )
        user_count, ai_count = count_exchanges()
        total_tokens = estimate_tokens()
        session_duration = format_duration(time.time() - st.session_state.session_start)
        st.markdown(
            f"""
            <div class='sidebar-metric'><span class='sidebar-metric-key'>Model</span><span class='sidebar-metric-value'>{MODEL_NAME}</span></div>
            <div class='sidebar-metric'><span class='sidebar-metric-key'>Temperature</span><span class='sidebar-metric-value'>{TEMPERATURE}</span></div>
            <div class='sidebar-metric'><span class='sidebar-metric-key'>User messages</span><span class='sidebar-metric-value'>{user_count}</span></div>
            <div class='sidebar-metric'><span class='sidebar-metric-key'>Assistant replies</span><span class='sidebar-metric-value'>{ai_count}</span></div>
            <div class='sidebar-metric'><span class='sidebar-metric-key'>Estimated tokens</span><span class='sidebar-metric-value'>{total_tokens}</span></div>
            <div class='sidebar-metric'><span class='sidebar-metric-key'>Session duration</span><span class='sidebar-metric-value'>{session_duration}</span></div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class='sidebar-card'>
                <div class='sidebar-section-title'>Controls</div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("New Chat", use_container_width=True, disabled=st.session_state.is_generating):
            clear_chat()
            st.rerun()
        if st.button("Clear Chat", use_container_width=True, disabled=st.session_state.is_generating):
            clear_chat()
            st.rerun()

        export_content = "\n\n".join(
            f"[{format_timestamp(idx)}] {'You' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for idx, msg in enumerate(st.session_state.messages)
            if not isinstance(msg, SystemMessage)
        )
        st.download_button(
            "Export Chat",
            data=export_content or "",
            file_name="funnybot_chat.txt",
            disabled=st.session_state.is_generating,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class='sidebar-card'>
                <div class='sidebar-section-title'>Theme</div>
                <div class='sidebar-metric'><span class='sidebar-metric-key'>Mode</span><span class='sidebar-metric-value'>Dark</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='sidebar-card'>
                <div class='sidebar-section-title'>About</div>
                <p class='sidebar-note'>Version 1.0.0 · Streamlit UI redesign for a premium AI assistant experience.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================================
# HEADER
# =========================================================================

def render_header() -> None:
    """Renders the premium hero header with model status, date, and branding."""
    current_time = datetime.now().strftime("%A, %b %d • %I:%M %p")
    hero_style = "hero-panel dark" if st.session_state.theme_mode == "Dark" else "hero-panel"
    subtext_color = "#94a3b8" if st.session_state.theme_mode == "Dark" else "#475569"
    st.markdown(
        f"""
        <div class='{hero_style}'>
            <div class='hero-grid'>
                <div class='hero-copy'>
                    <h1>FunnyBot AI — Your premium AI assistant</h1>
                    <p>Fast, polished conversational intelligence for brainstorming, coding, summarizing, and creative work.</p>
                    <div class='hero-pill-row'>
                        <span class='hero-pill'>Model: {MODEL_NAME}</span>
                        <span class='hero-pill status'>🟢 Online</span>
                        <span class='hero-pill time'>{current_time}</span>
                    </div>
                </div>
                <div class='hero-visual'>
                    <div class='hero-visual-inner'>
                        <div class='hero-visual-icon'>🤖</div>
                        <div style='font-size:1rem; color: #4338ca; font-weight:700;'>AI assistant</div>
                        <div style='margin-top:0.75rem; color: {subtext_color};'>Built for modern work and premium conversations.</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# DASHBOARD CARDS
# =========================================================================

def render_dashboard_cards() -> None:
    """Renders a set of premium dashboard metric cards."""
    user_count, ai_count = count_exchanges()
    total_tokens = estimate_tokens()
    session_duration = format_duration(time.time() - st.session_state.session_start)
    last_response = f"{st.session_state.last_response_time}s" if st.session_state.last_response_time else "—"

    st.markdown(
        f"""
        <div class='metrics-grid'>
            <div class='metric-card'>
                <div class='metric-icon'>🤖</div>
                <h3>Active Model</h3>
                <p>{MODEL_NAME}</p>
            </div>
            <div class='metric-card'>
                <div class='metric-icon'>💬</div>
                <h3>Conversation Count</h3>
                <p>{user_count + ai_count}</p>
            </div>
            <div class='metric-card'>
                <div class='metric-icon'>⚡</div>
                <h3>Estimated Tokens</h3>
                <p>{total_tokens}</p>
            </div>
            <div class='metric-card'>
                <div class='metric-icon'>⏱️</div>
                <h3>Session Duration</h3>
                <p>{session_duration}</p>
            </div>
            <div class='metric-card'>
                <div class='metric-icon'>🟢</div>
                <h3>AI Status</h3>
                <p>Online</p>
            </div>
            <div class='metric-card'>
                <div class='metric-icon'>⏳</div>
                <h3>Response Time</h3>
                <p>{last_response}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# CHAT AREA
# =========================================================================

def render_empty_state() -> None:
    """Friendly welcome screen shown before the first message is sent."""
    st.markdown(
        """
        <div class='chat-panel'>
            <div class='chat-message-wrapper'>
                <h2>Welcome to FunnyBot AI</h2>
                <p class='chat-meta'>Start a new conversation with a suggestion below or type your own prompt.</p>
                <div class='metrics-grid'>
                    <div class='suggestion-card'>
                        <h4>Explain Quantum Computing</h4>
                        <p>Ask FunnyBot to simplify complex ideas with clarity.</p>
                    </div>
                    <div class='suggestion-card'>
                        <h4>Write Python Code</h4>
                        <p>Generate ready-to-run Python snippets for your task.</p>
                    </div>
                    <div class='suggestion-card'>
                        <h4>Summarize an Article</h4>
                        <p>Condense content into a clear, easy-to-read summary.</p>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestion_columns = st.columns(4, gap="small")
    prompts = [
        "Explain Quantum Computing",
        "Write Python Code",
        "Summarize an Article",
        "Debug My Program",
        "Generate SQL Query",
        "Create a Resume",
        "Translate Text",
        "Brainstorm Startup Ideas",
    ]
    for idx, prompt_text in enumerate(prompts):
        with suggestion_columns[idx % 4]:
            if st.button(prompt_text, key=f"suggest_{idx}"):
                handle_user_input(prompt_text)
                st.rerun()


def render_chat_history() -> None:
    """Renders all HumanMessage/AIMessage turns using st.chat_message, with markdown support."""
    visible_messages = [m for m in st.session_state.messages if not isinstance(m, SystemMessage)]

    if not visible_messages:
        return

    for idx, msg in enumerate(st.session_state.messages):
        if isinstance(msg, SystemMessage):
            continue
        timestamp = format_timestamp(idx)
        if isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="🧑"):
                st.markdown(msg.content)
                st.markdown(f"<div class='chat-meta'><strong>You</strong><span>{timestamp}</span></div>", unsafe_allow_html=True)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg.content)
                st.markdown(f"<div class='chat-meta'><strong>FunnyBot</strong><span>{timestamp}</span></div>", unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                with col1:
                    if st.button("Copy", key=f"copy_{idx}"):
                        st.toast("Copied response! 🤖")
                with col2:
                    if st.button("Regenerate", key=f"regen_{idx}"):
                        last_prompt = next((m.content for m in reversed(st.session_state.messages) if isinstance(m, HumanMessage)), None)
                        if last_prompt:
                            handle_user_input(last_prompt)
                            st.rerun()
                with col3:
                    if st.button("👍", key=f"like_{idx}"):
                        st.toast("Thanks for the feedback!")
                with col4:
                    if st.button("👎", key=f"dislike_{idx}"):
                        st.toast("Feedback noted — thanks!")


def render_typing_indicator() -> None:
    """Shows a small animated 'typing' bubble while the bot is generating a reply."""
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            """
            <div class='typing-dots'>
                <span>●</span><span>●</span><span>●</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================================
# MAIN CHAT LOGIC (BACKEND UNCHANGED — same invoke call & memory append)
# =========================================================================

def handle_user_input(prompt: str) -> None:
    """
    Mirrors the original loop body exactly:
        messages.append(HumanMessage(content=prompt))
        response = model.invoke(messages)
        messages.append(AIMessage(content=response.content))
    """
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.message_timestamps.append(time.time())

    # Render the user's message immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Generate response with a visible typing/loading state
    st.session_state.is_generating = True
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown(
            '<div class="typing-dots"><span>●</span><span>●</span><span>●</span></div>',
            unsafe_allow_html=True,
        )
        try:
            start_time = time.monotonic()
            response = model.invoke(st.session_state.messages)
            st.session_state.last_response_time = round(time.monotonic() - start_time, 2)
            st.session_state.messages.append(AIMessage(content=response.content))
            st.session_state.message_timestamps.append(time.time())
            placeholder.markdown(response.content)
        except Exception as e:
            placeholder.empty()
            st.error(f"⚠️ Something went wrong: {e}")
            st.session_state.is_generating = False
            return

    st.session_state.is_generating = False
    st.toast("Response ready! 🎉", icon="✅")


# APP ENTRY POINT

def main() -> None:
    inject_custom_css()
    init_session_state()

    render_sidebar()
    render_header()
    render_dashboard_cards()

    visible_messages = [m for m in st.session_state.messages if not isinstance(m, SystemMessage)]
    if not visible_messages:
        render_empty_state()

    render_chat_history()

    prompt = st.chat_input(
        "Type your message here...",
        disabled=st.session_state.is_generating,
        key="chat_input",
    )

    if prompt:
        handle_user_input(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
