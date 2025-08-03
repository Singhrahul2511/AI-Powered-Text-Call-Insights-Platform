# utils/audio_processing.py

import whisper
import streamlit as st
from transformers import pipeline
import re
import shutil # Import the shutil library to check for executables

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def transcribe_audio(file_path):
    """
    Transcribes an audio file using OpenAI's Whisper model.
    Includes a definitive check for FFmpeg's existence.
    """
    # **FIX:** Add a definitive check to see if ffmpeg is in the system's PATH.
    # This will confirm if the installation via packages.txt was successful and made available to the app.
    if not shutil.which("ffmpeg"):
        st.error(
            "FATAL: FFmpeg is not installed or not in the system's PATH. "
            "Even though packages.txt exists, the Streamlit Cloud environment has failed to make FFmpeg available. "
            "Please try rebooting the app one more time. If the error persists, please contact Streamlit support as this indicates a platform-level issue."
        )
        return None

    try:
        model = load_whisper_model()
        result = model.transcribe(file_path, fp16=False) # Set fp16=False for CPU
        return result["text"]
    except Exception as e:
        # This will now catch other potential errors from whisper itself.
        st.error(f"An unexpected error occurred during transcription: {e}")
        return None

def summarize_text(text):
    """
    Generates a summary of the given text using a transformers pipeline.
    """
    try:
        summarizer = load_summarizer()
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return "Could not generate summary."

def calculate_talk_to_listen_ratio(transcription):
    """
    A simplified implementation to calculate the talk-to-listen ratio.
    """
    agent_keywords = ['i', 'me', 'my', 'we', 'our', 'us', 'company']
    words = re.findall(r'\b\w+\b', transcription.lower())
    if not words:
        return 0.0
    
    agent_word_count = sum(1 for word in words if word in agent_keywords)
    total_words = len(words)
    
    return agent_word_count / total_words if total_words > 0 else 0.0

def extract_pain_points(transcription):
    """
    A rule-based approach to extract potential customer pain points.
    """
    pain_point_keywords = ['problem', 'issue', 'not working', 'broken', 'frustrated', 'disappointed', 'error', 'fail', 'slow', 'confusing']
    sentences = re.split(r'[.!?]', transcription)
    pain_points = [sentence.strip() for sentence in sentences if any(keyword in sentence.lower() for keyword in pain_point_keywords)]
    return pain_points
