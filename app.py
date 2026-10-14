import json
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "faqs.json"

st.set_page_config(
    page_title="CampusConnect | Student Support",
    page_icon="🎓",
    layout="centered",
)

@st.cache_data
def load_faqs():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

faqs = load_faqs()
questions = [item["question"] for item in faqs]
# Include alternate phrasings so the chatbot can recognize common variations.
training_texts = []
training_answers = []
for item in faqs:
    variations = [item["question"]] + item.get("examples", [])
    for phrase in variations:
        training_texts.append(phrase)
        training_answers.append(item["answer"])

@st.cache_resource
def build_search_model(texts):
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

vectorizer, faq_matrix = build_search_model(training_texts)

def get_answer(message):
    cleaned = message.strip()
    if not cleaned:
        return "Please type a question so I can help.", 0.0

    query_vector = vectorizer.transform([cleaned])
    scores = cosine_similarity(query_vector, faq_matrix).flatten()
    best_index = int(scores.argmax())
    best_score = float(scores[best_index])

    # This threshold prevents the bot from confidently answering unrelated questions.
    if best_score < 0.16:
        return (
            "I'm not sure about that yet. Please contact your college help desk or "
            "student affairs office. You can also try asking about admissions, fees, "
            "examinations, attendance, library services, scholarships, or technical support.",
            best_score,
        )
    return training_answers[best_index], best_score

st.title("🎓 CampusConnect")
st.subheader("AI Chatbot for Student Support Services")
st.write(
    "Ask common questions about college services. This demo uses TF-IDF text "
    "matching to find the closest FAQ; it does not connect to official college systems."
)

with st.sidebar:
    st.header("About this project")
    st.write("**Technology:** Python, Streamlit, scikit-learn")
    st.write("**NLP method:** TF-IDF + cosine similarity")
    st.write("**Data:** Sample college FAQs")
    st.info("Replace the sample answers with verified information from your institution before real use.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm CampusConnect. What would you like to know about student services?",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("For example: How can I check my exam timetable?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer, score = get_answer(prompt)
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("How did the chatbot match this question?"):
            st.write(f"Similarity score: {score:.3f}")
            st.caption("A higher score means the wording is more similar to a stored FAQ. It is not a probability of correctness.")
    st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.caption("Academic prototype • Verify official policies with your college administration.")
