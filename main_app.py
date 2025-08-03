# main_app.py

import streamlit as st

st.set_page_config(
    page_title="AI-driven Insights Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 AI-driven Text and Call Insights Platform")

st.markdown("""
Welcome to the AI-driven Insights Platform. This application combines two powerful use cases:

1.  **Interactive Text Analysis Platform**: Analyze text data from various sources to uncover insights, sentiments, and topics.
2.  **AI-driven Call Insights**: Transcribe and analyze call recordings to understand customer sentiment, generate summaries, and extract key information.

Please select a use case from the sidebar to get started.
""")
