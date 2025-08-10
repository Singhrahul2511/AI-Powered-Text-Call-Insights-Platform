# AI-Powered Text & Call Insights Platform
```bash
<p align="center">
<img src="https://www.google.com/search?q=https://i.imgur.com/your-banner-image.png" alt="Project Banner">
<!-- You can create a banner image and upload it to a service like imgur.com -->
</p>

<p align="center">
<a href="https://www.google.com/search?q=https://your-streamlit-app-url.streamlit.app/">
<img src="https://www.google.com/search?q=https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
</a>
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</p>
```
#### An advanced, multi-page Streamlit application designed to extract actionable insights from both unstructured text and audio call recordings. This platform integrates a suite of NLP and Speech-to-Text models to provide a comprehensive analytics dashboard for business intelligence, customer feedback analysis, and call center optimization.

🚀 Live Demo
Experience the live application here: https://rahul-insights-platform.streamlit.app/

(Note: The initial loading of AI models on the free tier may take a moment.)

```bash
✨ Key Features
This platform is composed of two primary modules, each with a rich set of features:
```

📊 1. **Interactive Text Analysis Platform**
```bash
Versatile Data Loading: Upload and analyze text data from .csv, .xls, .xlsx, and .json files.

Advanced Preprocessing: Utilizes spaCy for robust tokenization, lemmatization, and stopword removal.

Dual Sentiment Analysis: Choose between TextBlob (polarity/subjectivity) and VADER (compound sentiment scoring) for nuanced analysis.

Topic Modeling: Employs LDA and NMF to uncover latent topics and themes within your text data.

Intelligent Keyword Extraction: Identifies key terms using both raw frequency counts and TF-IDF relevance scores.

Rich Visualizations: Interactive charts for sentiment distribution, topic modeling results, and keyword frequency.

Dynamic Word Clouds: Generate visually appealing word clouds to highlight the most prominent terms.

Data Export: Download the full analysis results, including processed text and sentiment scores, as an Excel file.
```

📞 2. **AI-driven Call Insights**

```bash
Audio Transcription: High-accuracy speech-to-text transcription for .mp3 and .wav files using OpenAI-Whisper.

Abstractive Summarization: Generates concise, human-readable summaries of call transcripts using a Transformers-based model.

Sentiment Arc: Performs sentiment analysis on the transcribed text to gauge the emotional tone of the conversation.

Talk-to-Listen Ratio: Calculates an estimated ratio to analyze agent and customer interaction dynamics.

Automated Pain Point Detection: A rule-based system to automatically flag sentences indicating customer issues or frustration.

Comprehensive Dashboard: A unified view displaying the call summary, sentiment metrics, talk ratio, and keyword clouds for quick insights.
```

### 🛠️ Tech Stack & Architecture
This application is built with a modern, modular Python stack, ensuring scalability and maintainability.
```bash
Frontend: Streamlit

Core NLP: spaCy, NLTK

Sentiment Analysis: TextBlob, VADER

Topic Modeling: scikit-learn, Gensim

Speech-to-Text: OpenAI-Whisper

Summarization: Hugging Face Transformers (BART)

Data Handling: pandas

Visualizations: Plotly, Matplotlib, WordCloud
```
### System Architecture
The project is organized into a multi-page Streamlit app structure for clean separation of concerns:

```bash
insights_platform/
│
├── pages/
│   ├── 1_Text_Analysis.py
│   └── 2_Call_Insights.py
│
├── utils/
│   ├── helpers.py
│   ├── text_processing.py
│   └── audio_processing.py
│
├── main_app.py
├── requirements.txt
└── packages.txt
```
### ⚙️ Installation & Local Setup
To run this project on your local machine, please follow these steps.

#### Prerequisites
```bash
Python 3.11
Git

FFmpeg (for the Call Insights module)
```

1. **Clone the Repository**
```bash
git clone [https://github.com/Singhrahul2511/AI-Powered-Text-Call-Insights-Platform.git](https://github.com/Singhrahul2511/AI-Powered-Text-Call-Insights-Platform.git)
cd AI-Powered-Text-Call-Insights-Platform
```
2. **Create and Activate a Virtual Environment**
It's highly recommended to use a virtual environment to manage dependencies.
```bash
Create the environment
python -m venv venv
Activate it
1. On Windows:
```bash
venv\Scripts\activate
```
2. On macOS/Linux:
```bash
source venv/bin/activate
```
3. **Install FFmpeg**
```bash
The Whisper library requires FFmpeg.

On macOS (using Homebrew):

brew install ffmpeg

On Linux (using apt):

sudo apt update && sudo apt install ffmpeg
```

#### On Windows:
This requires a manual installation. Please follow the detailed guide here to download the program and add it to your system's PATH.

4. **Install Python Dependencies**
The requirements.txt file contains pinned versions for a stable setup.
```bash
pip install -r requirements.txt
```

5. **Run the Streamlit App**
```bash
streamlit run main_app.py
```

```bash
The application should now be running and accessible in your web browser.

☁️ Deployment
This application is configured for seamless deployment on Streamlit Cloud.

Push to GitHub: Ensure your repository is up-to-date.

Create a Streamlit Cloud Account: Sign up using your GitHub account.

Deploy: Link your GitHub repository and deploy. The deployment will automatically use the requirements.txt and packages.txt files to set up the environment correctly, including the installation of FFmpeg.
```

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

📧 Contact
Rahul Singh - Your LinkedIn Profile URL - rahulkumarpatelbr.@gmail.com

Project Link: https://github.com/Singhrahul2511/AI-Powered-Text-Call-Insights-Platform
