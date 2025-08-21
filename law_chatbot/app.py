# Add this at the very top
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from legal_advisor_bot import LegalAdvisorBot
import os

# --- Page Config ---
st.set_page_config(
    page_title="Legal Advisor Bot",
    page_icon="⚖️",
    layout="wide",
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput input {
        border: 2px solid #4CAF50;
        border-radius: 10px;
    }
    .stButton button {
        background: linear-gradient(to right, #4CAF50, #2e7d32);
        color: white;
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: bold;
    }
    .chat-bubble {
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 80%;
    }
    .user {
        background-color: #DCF8C6;
        align-self: flex-end;
    }
    .bot {
        background-color: #EAEAEA;
        align-self: flex-start;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Bot ---
@st.cache_resource
def load_bot():
    try:
        return LegalAdvisorBot(pdf_path="law_chatbot/law1.pdf")
    except Exception as e:
        st.error(f"Error initializing bot: {str(e)}")
        return None

bot = load_bot()

# --- Title ---
st.title("⚖️ Legal Advisor Chatbot")
st.markdown("Ask any **law-related question** based on uploaded legal documents. The bot will provide clear, easy-to-understand answers with legal references.")

# --- Chat History Session ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- User Input ---
user_input = st.text_input("💬 Enter your legal question:", key="input")

# --- Chat Handling ---
if st.button("Ask") and user_input.strip() != "":
    if bot is None:
        st.error("Bot is not initialized properly. Please check your API key and try again.")
    else:
        with st.spinner("Analyzing your query... ⏳"):
            try:
                answer = bot.ask_question(user_input)
                # Save history
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("bot", answer))
            except Exception as e:
                st.error(f"Error getting answer: {str(e)}")

# --- Display Chat ---
if st.session_state.chat_history:
    st.subheader("📜 Conversation")
    for role, text in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"<div class='chat-bubble user'>👤 **You:** {text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble bot'>🤖 **Bot:** {text}</div>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
st.sidebar.info("This chatbot is powered by **Groq LLaMA 3.3 70B** + **LangChain**.")
st.sidebar.markdown("Developed with ❤️ using Streamlit.")

# Show warning if bot not initialized
if bot is None:
    st.sidebar.error("⚠️ Bot initialization failed. Please check your API key in Streamlit secrets.")
