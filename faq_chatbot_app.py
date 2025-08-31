import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download("punkt")
nltk.download("stopwords")

faq_tfidf = None
vectorizer = None
faq_answers = [
    "AI stands for Artificial Intelligence. It is the simulation of human intelligence in machines.",
    "Machine Learning is a subset of AI which enables machines to learn from data.",
    "Deep Learning is a subset of Machine Learning that uses neural networks with many layers.",
    "Streamlit is an open-source app framework for Machine Learning and Data Science projects.",
    "NLP stands for Natural Language Processing, which enables computers to understand human language."
]

def load_faq_model():
    global faq_tfidf, vectorizer
    vectorizer = TfidfVectorizer()
    faq_tfidf = vectorizer.fit_transform(faq_answers)

def get_answer(user_query):
    global faq_tfidf, vectorizer
    if faq_tfidf is None or vectorizer is None:
        load_faq_model()
    user_tfidf = vectorizer.transform([user_query])
    similarity = cosine_similarity(user_tfidf, faq_tfidf)
    idx = similarity.argmax()
    return faq_answers[idx]

st.title("🤖 FAQ Chatbot")

st.write("Ask questions about AI, ML, Deep Learning, NLP, or Streamlit.")

user_question = st.text_input("Type your question:")

if st.button("Get Answer"):
    if user_question.strip() != "":
        answer = get_answer(user_question)
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.warning("Please type a question.")
