# pages/2_Call_Insights.py

import streamlit as st
from utils.audio_processing import transcribe_audio, summarize_text, calculate_talk_to_listen_ratio, extract_pain_points
from utils.text_processing import get_sentiment_vader, create_wordcloud
import pandas as pd
import plotly.express as px
import os
import tempfile

st.set_page_config(
    page_title="AI-driven Call Insights",
    page_icon="📞",
    layout="wide"
)

st.title("📞 AI-driven Call Insights")

uploaded_file = st.file_uploader("Upload an audio file (.mp3, .wav)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type)

    if st.button("Analyze Call"):
        temp_file_path = None
        try:
            # **FIX:** Use tempfile for robust, cross-platform temporary file handling.
            # This creates a temporary file and gives us its path.
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_file_path = tmp_file.name

            with st.spinner("Transcribing audio... This may take a while."):
                # Pass the secure temporary file path to the transcription function
                transcription = transcribe_audio(temp_file_path)
            
            if transcription:
                st.success("Transcription complete.")
                
                st.subheader("Transcription")
                st.write(transcription)

                with st.spinner("Analyzing transcription..."):
                    sentiment = get_sentiment_vader(transcription)
                    summary = summarize_text(transcription)
                    talk_listen_ratio = calculate_talk_to_listen_ratio(transcription)
                    pain_points = extract_pain_points(transcription)

                st.subheader("Call Analytics Dashboard")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Overall Sentiment", value=f"{sentiment['compound']:.2f} (VADER Compound Score)")
                    st.markdown("#### Sentiment Breakdown")
                    sentiment_df = pd.DataFrame([sentiment])
                    st.bar_chart(sentiment_df[['neg', 'neu', 'pos']])

                with col2:
                    talk_ratio = max(0, min(1, talk_listen_ratio))
                    listen_ratio = 1 - talk_ratio
                    st.metric(label="Talk-to-Listen Ratio", value=f"{talk_ratio:.2f}")
                    st.markdown("#### Talk-to-Listen Ratio")
                    ratio_fig = px.pie(values=[talk_ratio, listen_ratio], names=['Talk', 'Listen'], title='Talk-to-Listen Ratio')
                    st.plotly_chart(ratio_fig, use_container_width=True)

                st.subheader("Call Summary")
                st.write(summary)

                st.subheader("Customer Pain Points")
                if pain_points:
                    for point in pain_points:
                        st.write(f"- {point}")
                else:
                    st.write("No specific pain points identified.")

                st.subheader("Keyword Cloud")
                with st.spinner("Generating keyword cloud..."):
                    wordcloud_fig = create_wordcloud(pd.Series([transcription]))
                    st.pyplot(wordcloud_fig)
                
            else:
                st.error("Could not transcribe the audio. Please try another file.")
        
        finally:
            # **FIX:** Ensure the temporary file is cleaned up, even if errors occur.
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
