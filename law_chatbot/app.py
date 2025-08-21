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


st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* All text white */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Headers white */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Text input styling */
    .stTextInput input {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 2px solid #4CAF50;
        border-radius: 10px;
    }
    
    .stTextInput label {
        color: #ffffff !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(to right, #4CAF50, #2e7d32);
        color: white;
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: bold;
        border: none;
    }
    
    .stButton button:hover {
        background: linear-gradient(to right, #2e7d32, #4CAF50);
        color: white;
    }
    
    /* Chat bubbles */
    .chat-bubble {
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 80%;
    }
    
    .user {
        background-color: #2c5282;
        color: #ffffff;
        align-self: flex-end;
    }
    
    .bot {
        background-color: #2d3748;
        color: #ffffff;
        align-self: flex-start;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a1a;
    }
    
    .css-1d391kg p, .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #ffffff !important;
    }
    
    /* Markdown text white */
    .stMarkdown {
        color: #ffffff;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333333;
    }
    
    /* Error messages */
    .stError {
        background-color: #2d1a1a;
        color: #ff6b6b;
        border: 1px solid #ff6b6b;
    }
    
    /* Warning messages */
    .stWarning {
        background-color: #2d2a1a;
        color: #ffe066;
        border: 1px solid #ffe066;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #1a2d1a;
        color: #51cf66;
        border: 1px solid #51cf66;
    }
    
    /* Spinner color */
    .stSpinner > div {
        border-color: #4CAF50 transparent transparent transparent !important;
    }
    
    /* Text area */
    .stTextArea textarea {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333333;
    }
    
    /* Select boxes */
    .stSelectbox select {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333333;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #ffffff !important;
    }
    
    /* Checkboxes */
    .stCheckbox label {
        color: #ffffff !important;
    }
    
    /* File uploader */
    .stFileUploader label {
        color: #ffffff !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        color: #ffffff !important;
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
