# topic_modelling_reviews
This script analyses anonymised customer review data from two sources (Google and Trustpilot) to extract insights using NLP techniques including: - Preprocessing and word frequency analysis - WordClouds and visualisation - BERTopic and LDA topic modelling - Emotion detection using a BERT model - Text generation with a large language model

# 🧠 Topic Modelling & Emotion Analysis on Customer Reviews

This project showcases the use of **Natural Language Processing (NLP)** techniques to uncover insights from anonymised customer review data. By applying both **topic modelling** and **emotion analysis**, the project identifies key themes and customer sentiments that can help guide business improvement decisions.

> ⚠️ This project uses **fully anonymised** review data from two public-facing platforms and is designed for educational and demonstrative purposes only.

---

## 🚀 Project Objectives

- Preprocess and clean raw text reviews.
- Identify frequent keywords and visualise them using **WordClouds** and **bar charts**.
- Apply **BERTopic** and **LDA (Latent Dirichlet Allocation)** to uncover hidden topics within negative reviews.
- Detect emotional sentiment using a **BERT-based emotion classifier**.
- Leverage a **Large Language Model (LLM)** to extract top-level review topics and generate actionable insights.

---

## 📁 Project Structure

- `topic_modelling_reviews.py` – Main Python script for data processing, topic modelling, and emotion analysis  
- `README.md` – Project overview and usage guide  
- `data/` – Folder containing anonymised review datasets  
  - `google_reviews.csv`  
  - `trustpilot_reviews.csv`

---

## 🛠️ Tools & Libraries

- `pandas`, `matplotlib`, `seaborn`, `nltk`
- `wordcloud`, `gensim`, `langdetect`
- [`BERTopic`](https://github.com/MaartenGr/BERTopic)
- [`transformers`](https://huggingface.co/transformers/)
  - Emotion Model: `bhadresh-savani/bert-base-uncased-emotion`
  - LLM: `microsoft/Phi-3.5-mini-instruct`
- `sentence-transformers`, `scikit-learn`

---

## 🧪 Key Features

### ✅ Preprocessing & Exploratory Analysis
- Tokenization, stopword removal, lowercase conversion
- Word frequency distribution
- Visualisation of most common terms

### 🧵 Topic Modelling
- **BERTopic**: High-performance topic clustering with transformer embeddings
- **LDA (Gensim)**: Classic probabilistic model for topic discovery
- Topic visualisation (bar charts, distance maps, heatmaps)

### 😠 Emotion Analysis
- Detects emotions in reviews (e.g. anger, joy, sadness)
- Filters out reviews with negative sentiment for deep topic analysis

### 🤖 Large Language Model (LLM) Integration
- Extracts high-level topics using a GPT-style model
- Generates actionable business insights from review text

---

## 📈 Sample Output

- Top Negative Topics: `['membership access', 'cleanliness', 'staff behavior']`
- Common Emotions: `anger`, `disgust`, `sadness`
- Suggestions: Improve front-desk support, streamline membership access, maintain gym cleanliness.

---
