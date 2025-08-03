# pages/1_Text_Analysis.py

import streamlit as st
import pandas as pd
from utils.helpers import load_data, get_table_download_link
from utils.text_processing import (
    preprocess_text, get_sentiment_textblob, get_sentiment_vader,
    get_keywords, get_tfidf_keywords, perform_topic_modeling,
    plot_sentiment_distribution, plot_top_keywords, plot_topic_distribution,
    create_wordcloud
)

st.set_page_config(
    page_title="Interactive Text Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Interactive Text Analysis Platform")

with st.sidebar:
    st.header("1. Input Data")
    input_method = st.radio("Choose input method:", ("Upload a file", "Paste text"))

    text_column = None
    df = None

    if input_method == "Upload a file":
        uploaded_file = st.file_uploader("Choose a file (CSV, Excel, JSON)", type=["csv", "xlsx", "xls", "json"])
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                st.success("File loaded successfully!")
                text_column_options = [col for col in df.columns if df[col].dtype == 'object']
                if not text_column_options:
                    st.warning("No text columns found in the uploaded file.")
                else:
                    text_column = st.selectbox("Select the column with text data:", text_column_options)
    else:
        text_input = st.text_area("Enter text here:", height=200)
        if text_input:
            df = pd.DataFrame([text_input], columns=['text'])
            text_column = 'text'

    if df is not None and text_column:
        st.header("2. Analysis Options")
        analysis_tasks = st.multiselect(
            "Select analysis tasks:",
            ["Data Preview", "Text Preprocessing", "Sentiment Analysis", "Keyword Extraction", "Topic Modeling", "Word Cloud"],
            default=["Data Preview", "Sentiment Analysis"]
        )
        
        if "Sentiment Analysis" in analysis_tasks:
            st.subheader("Sentiment Analysis Settings")
            sentiment_method = st.radio("Choose sentiment analysis method:", ("TextBlob", "VADER"))

        if "Topic Modeling" in analysis_tasks:
            st.subheader("Topic Modeling Settings")
            topic_method = st.radio("Choose topic modeling method:", ("LDA", "NMF"))
            num_topics = st.slider("Number of topics:", min_value=2, max_value=15, value=5)

if df is not None and text_column:
    st.header("Results")
    df[text_column] = df[text_column].astype(str).fillna('')
    analysis_df = df.copy()

    if "Data Preview" in analysis_tasks:
        st.subheader("Data Preview")
        st.dataframe(df.head())
        st.write(f"**Shape of the dataset:** {df.shape[0]} rows, {df.shape[1]} columns")

    if "Text Preprocessing" in analysis_tasks:
        with st.spinner("Preprocessing text..."):
            analysis_df['processed_tokens'] = analysis_df[text_column].apply(preprocess_text)
            analysis_df['processed_text'] = analysis_df['processed_tokens'].apply(lambda tokens: ' '.join(tokens))
        st.subheader("Preprocessed Text")
        st.dataframe(analysis_df[['processed_text', 'processed_tokens']].head())
        st.success("Text preprocessing complete.")

    if "Sentiment Analysis" in analysis_tasks:
        st.subheader("Sentiment Analysis")
        with st.spinner(f"Performing sentiment analysis with {sentiment_method}..."):
            if sentiment_method == "TextBlob":
                sentiments = analysis_df[text_column].apply(get_sentiment_textblob)
                analysis_df['polarity'] = sentiments.apply(lambda x: x[0])
                analysis_df['subjectivity'] = sentiments.apply(lambda x: x[1])
                st.plotly_chart(plot_sentiment_distribution(analysis_df, 'polarity'), use_container_width=True)
            
            elif sentiment_method == "VADER":
                sentiments = analysis_df[text_column].apply(get_sentiment_vader)
                vader_df = pd.DataFrame(sentiments.tolist())
                analysis_df = pd.concat([analysis_df, vader_df], axis=1)
                st.plotly_chart(plot_sentiment_distribution(analysis_df, 'compound'), use_container_width=True)
        st.dataframe(analysis_df.head())
        st.success("Sentiment analysis complete.")
    
    if "Keyword Extraction" in analysis_tasks:
        st.subheader("Keyword Extraction")
        if 'processed_tokens' not in analysis_df.columns:
             with st.spinner("Preprocessing text for keyword extraction..."):
                analysis_df['processed_tokens'] = analysis_df[text_column].apply(preprocess_text)
        
        with st.spinner("Extracting keywords..."):
            all_tokens = [token for sublist in analysis_df['processed_tokens'] for token in sublist]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Top Keywords by Frequency")
                top_freq_keywords = get_keywords(all_tokens, top_n=15)
                st.plotly_chart(plot_top_keywords(top_freq_keywords), use_container_width=True)
            
            with col2:
                st.markdown("#### Top Keywords by TF-IDF")
                tfidf_keywords_df = get_tfidf_keywords(analysis_df[text_column], top_n=5)
                st.dataframe(tfidf_keywords_df)
        st.success("Keyword extraction complete.")

    if "Topic Modeling" in analysis_tasks:
        st.subheader(f"Topic Modeling with {topic_method}")
        if 'processed_text' not in analysis_df.columns:
             with st.spinner("Preprocessing text for topic modeling..."):
                analysis_df['processed_tokens'] = analysis_df[text_column].apply(preprocess_text)
                analysis_df['processed_text'] = analysis_df['processed_tokens'].apply(lambda tokens: ' '.join(tokens))

        with st.spinner(f"Running {topic_method} for {num_topics} topics..."):
            valid_docs = analysis_df[analysis_df['processed_text'].str.strip().astype(bool)]
            if len(valid_docs) < num_topics:
                st.warning(f"Not enough documents ({len(valid_docs)}) to perform topic modeling for {num_topics} topics.")
            else:
                df_topics, df_doc_topics = perform_topic_modeling(valid_docs['processed_text'], method=topic_method, num_topics=num_topics)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Top Words per Topic")
                    st.dataframe(df_topics)
                with col2:
                    st.markdown("#### Document Distribution Across Topics")
                    st.plotly_chart(plot_topic_distribution(df_doc_topics), use_container_width=True)
        st.success("Topic modeling complete.")

    if "Word Cloud" in analysis_tasks:
        st.subheader("Word Cloud")
        if 'processed_text' not in analysis_df.columns:
            with st.spinner("Preprocessing text for word cloud..."):
                analysis_df['processed_tokens'] = analysis_df[text_column].apply(preprocess_text)
                analysis_df['processed_text'] = analysis_df['processed_tokens'].apply(lambda tokens: ' '.join(tokens))
        
        with st.spinner("Generating word cloud..."):
            wordcloud_fig = create_wordcloud(analysis_df['processed_text'])
            st.pyplot(wordcloud_fig)
        st.success("Word cloud generated.")

    with st.sidebar:
        st.header("3. Export Results")
        if st.button("Prepare Download"):
            st.markdown(get_table_download_link(analysis_df.drop(columns=['processed_tokens'], errors='ignore')), unsafe_allow_html=True)

else:
    st.info("Please upload a file or paste text in the sidebar to begin analysis.")
