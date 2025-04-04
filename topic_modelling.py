
# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import string
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# 2. Setup

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

# 3. Load Data

# Replace with anonymised paths
google_path = 'data/google_reviews.csv'
trustpilot_path = 'data/trustpilot_reviews.csv'

g_reviews = pd.read_csv(google_path)
t_reviews = pd.read_csv(trustpilot_path)

g_reviews = g_reviews.dropna(subset=['Comment'])
t_reviews = t_reviews.dropna(subset=['Review Content'])

# 4. Preprocessing

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

g_reviews['Clean_Comment'] = g_reviews['Comment'].apply(preprocess_text)
t_reviews['Clean_Review_Content'] = t_reviews['Review Content'].apply(preprocess_text)

# 5. Word Frequency and WordClouds

def top_words(freq_dist, title):
    top = freq_dist.most_common(10)
    words, counts = zip(*top)
    plt.bar(words, counts)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

def wordcloud_plot(words, title):
    wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

g_words = [word for tokens in g_reviews['Clean_Comment'] for word in tokens]
t_words = [word for tokens in t_reviews['Clean_Review_Content'] for word in tokens]
g_freq_dist = FreqDist(g_words)
t_freq_dist = FreqDist(t_words)

top_words(g_freq_dist, "Top Words in Google Reviews")
top_words(t_freq_dist, "Top Words in Trustpilot Reviews")
wordcloud_plot(g_words, "Google Wordcloud")
wordcloud_plot(t_words, "Trustpilot Wordcloud")

# 6. Filter Negative Reviews

g_negative = g_reviews[g_reviews['Overall Score'] < 3]
t_negative = t_reviews[t_reviews['Review Stars'] < 3]

# 7. Emotion Classification

emotion_model = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

def classify_top_emotion(text):
    try:
        result = emotion_model(text[:512])
        return max(result, key=lambda x: x['score'])['label']
    except:
        return "unknown"

g_reviews['Top Emotion'] = g_reviews['Clean_Comment'].apply(lambda x: classify_top_emotion(' '.join(x)))
t_reviews['Top Emotion'] = t_reviews['Clean_Review_Content'].apply(lambda x: classify_top_emotion(' '.join(x)))

# 8. Filter Angry Reviews

g_angry = g_negative[g_reviews['Top Emotion'] == 'anger']
t_angry = t_negative[t_reviews['Top Emotion'] == 'anger']

angry_combined = pd.concat([
    g_angry['Clean_Comment'].apply(' '.join),
    t_angry['Clean_Review_Content'].apply(' '.join)
]).tolist()

def filter_english(texts):
    english = []
    for t in texts:
        try:
            if detect(t) == 'en':
                english.append(t)
        except LangDetectException:
            continue
    return english

angry_english = filter_english(angry_combined)

# 9. BERTopic on Angry Reviews

from bertopic import BERTopic

embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
topic_model = BERTopic(embedding_model=embedding_model, nr_topics=5)
topics, probs = topic_model.fit_transform(angry_english)

topic_model.visualize_topics()
topic_model.visualize_barchart(top_n_topics=5, n_words=5)

# 10. Gensim LDA on Angry Reviews

def clean(doc):
    stop_free = " ".join([w for w in doc.lower().split() if w not in stop_words])
    punc_free = ''.join(ch for ch in stop_free if ch not in string.punctuation)
    normalized = " ".join(lemmatizer.lemmatize(word) for word in punc_free.split())
    return normalized

tokenized_docs = [clean(doc).split() for doc in angry_english]
dictionary = corpora.Dictionary(tokenized_docs)
dictionary.filter_extremes(no_below=1, no_above=0.8)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

lda_model = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary, passes=20)

for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")

# 11. Use Large Language Model (LLM)

phi_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", device_map="cpu", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

llm_pipe = pipeline("text-generation", model=phi_model, tokenizer=tokenizer)

prompt = (
    "You work as a data analyst for a gym chain. In the following customer review, "
    "identify up to 5 main topics and return them as a list: "
)

sample_review = "The gym equipment was broken and nobody helped me when I asked for assistance."
response = llm_pipe(prompt + sample_review, max_new_tokens=1000)
print(response[0]['generated_text'])
