# utils/audio_processing.py

import whisper
import streamlit as st
from transformers import pipeline
import re

@st.cache_resource
def load_whisper_model():
    # Using .cache_resource to ensure the model is loaded only once
    return whisper.load_model("base")

@st.cache_resource
def load_summarizer():
    # Using .cache_resource for the summarization pipeline
    return pipeline("summarization", model="facebook/bart-large-cnn")

def transcribe_audio(file_path):
    """
    Transcribes an audio file using OpenAI's Whisper model.
    Accepts a file path string.
    """
    try:
        model = load_whisper_model()
        # The model's transcribe function expects a file path string.
        result = model.transcribe(file_path, fp16=False) # Set fp16=False for CPU
        return result["text"]
    except FileNotFoundError:
        # **FIX:** Catch the specific FileNotFoundError and provide a helpful message.
        # This is a common issue on local Windows machines if FFmpeg is not installed.
        st.error(
            "Transcription Error: FFmpeg not found. "
            "Whisper requires FFmpeg to process audio files. Please ensure FFmpeg is installed on your system and accessible in your PATH. "
            "You can download it from https://ffmpeg.org/download.html"
        )
        return None
    except Exception as e:
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
    A real-world scenario would require speaker diarization.
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
