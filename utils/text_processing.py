# utils/text_processing.py

import spacy
from collections import Counter
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import subprocess
import sys

def check_and_download_spacy_model(model="en_core_web_sm"):
    if not spacy.util.is_package(model):
        with st.spinner(f"Downloading spaCy model '{model}'..."):
            try:
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
                st.success(f"Successfully downloaded '{model}'. Please refresh the page.")
                st.stop()
            except Exception as e:
                st.error(f"Failed to download spaCy model: {e}")
                st.stop()

check_and_download_spacy_model()

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Could not load spaCy model. Please ensure it is installed.")
    st.stop()

@st.cache_data
def preprocess_text(text):
    if not isinstance(text, str):
        return []
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]

def get_sentiment_textblob(text):
    if not isinstance(text, str):
        return 0.0, 0.0
    return TextBlob(text).sentiment

def get_sentiment_vader(text):
    if not isinstance(text, str):
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    return SentimentIntensityAnalyzer().polarity_scores(text)

def get_keywords(tokens, top_n=10):
    return Counter(tokens).most_common(top_n)

def get_tfidf_keywords(series, top_n=10):
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(series.astype(str))
    feature_names = tfidf.get_feature_names_out()
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    top_keywords = []
    for i in range(len(df_tfidf)):
        top_n_words = df_tfidf.iloc[i].nlargest(top_n)
        formatted_keywords = ", ".join([f"{word} ({score:.2f})" for word, score in zip(top_n_words.index, top_n_words.values) if score > 0])
        top_keywords.append(formatted_keywords)
    return pd.DataFrame({'Top TF-IDF Keywords': top_keywords})

def perform_topic_modeling(series, method='LDA', num_topics=5, num_words=5):
    vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2, max_features=1000)
    term_matrix = vectorizer.fit_transform(series.astype(str))
    feature_names = vectorizer.get_feature_names_out()

    if method == 'LDA':
        model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    else: # NMF
        model = NMF(n_components=num_topics, random_state=42, init='nndsvda')

    doc_topic_matrix = model.fit_transform(term_matrix)
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append((f"Topic {topic_idx + 1}", ", ".join(top_words)))

    df_topics = pd.DataFrame(topics, columns=['Topic', 'Top Words'])
    df_doc_topics = pd.DataFrame(doc_topic_matrix, columns=[f"Topic {i+1}" for i in range(num_topics)])
    return df_topics, df_doc_topics

def plot_sentiment_distribution(df, column):
    fig = px.histogram(df, x=column, nbins=30, title=f'Distribution of Sentiment ({column})')
    return fig

def plot_top_keywords(keywords):
    df_keywords = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
    fig = px.bar(df_keywords.sort_values(by='Frequency', ascending=True), 
                 x='Frequency', y='Keyword', orientation='h', title='Top Keywords by Frequency')
    return fig

def plot_topic_distribution(df_doc_topics):
    dominant_topic = df_doc_topics.idxmax(axis=1)
    topic_counts = dominant_topic.value_counts().sort_index()
    fig = px.bar(topic_counts, x=topic_counts.index, y=topic_counts.values,
                 title='Document Distribution Across Topics')
    return fig

def create_wordcloud(series):
    text = " ".join(review for review in series)
    wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig
