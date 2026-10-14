import json, random
from pathlib import Path
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CampusConnect",page_icon="🎓",layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.5rem;max-width:1100px}
.hero{padding:1.5rem;border-radius:20px;background:linear-gradient(120deg,#17365d,#287783);color:white;margin-bottom:1rem}
.hero h1{color:white;margin:0}.hero p{color:#e6f4f5;margin:.4rem 0 0}
div.stButton>button{border-radius:12px;min-height:2.6rem;white-space:normal}
</style>""",unsafe_allow_html=True)

@st.cache_data
def load_data():
    return json.loads(Path(__file__).with_name("faqs.json").read_text(encoding="utf-8"))
faqs=load_data()
phrases=[]; owners=[]
for i,item in enumerate(faqs):
    for phrase in [item["question"]]+item["examples"]:
        phrases.append(phrase); owners.append(i)
@st.cache_resource
def build_model(phrases):
    v=TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1,2),sublinear_tf=True)
    return v,v.fit_transform(phrases)
vectorizer,matrix=build_model(phrases)

def answer_question(q):
    scores=cosine_similarity(vectorizer.transform([q]),matrix).flatten()
    pos=int(scores.argmax()); score=float(scores[pos]); item=faqs[owners[pos]]
    if score<0.12:
        return ("I couldn't confidently match that question to a student-support topic, and I don't want to guess. Try a topic from the sidebar or contact your college's official help desk.",["How can I check my exam results?","How can I pay my college fees?","I can't log in to the student portal."],"Need a clearer question")
    return random.choice(item["answers"]),item["followups"],item["category"]

if "messages" not in st.session_state:
    st.session_state.messages=[{"role":"assistant","content":"Hi! 👋 I'm **CampusConnect**, your student-support guide. Ask me a question or choose a topic. I'll suggest useful next steps.","suggestions":["Where do I find my exam timetable?","I forgot my student portal password.","How do I apply for a scholarship?"],"category":"Welcome"}]
if "pending" not in st.session_state: st.session_state.pending=None

with st.sidebar:
    st.markdown("## 🎓 CampusConnect")
    st.caption("Your college-service guide")
    st.divider()
    category=st.selectbox("Explore a topic",[x["category"] for x in faqs])
    st.caption("Tap a question to ask it:")
    for n,item in enumerate([x for x in faqs if x["category"]==category]):
        if st.button(item["question"],key=f"topic_{category}_{n}",use_container_width=True):
            st.session_state.pending=item["question"]; st.rerun()
    st.divider()
    if st.button("🧹 Start a new chat",use_container_width=True):
        st.session_state.messages=[{"role":"assistant","content":"Fresh start! What student-service question can I help with?","suggestions":["Admissions","Fees","Examinations"],"category":"Welcome"}]
        st.session_state.pending=None; st.rerun()
    st.caption("Demo only. Never enter passwords, OTPs, or private student data.")

st.markdown("""<div class="hero"><h1>🎓 CampusConnect</h1><p>Student support, made simpler. Ask naturally, explore topics, and get practical next steps.</p></div>""",unsafe_allow_html=True)
a,b,c=st.columns(3)
a.info("💬 **Ask naturally**\\n\\nUse your own words")
b.info("🧭 **Explore topics**\\n\\nFees, exams, library and more")
c.info("🔒 **Privacy first**\\n\\nNo student login required")

for mi,msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"],avatar="🎓" if msg["role"]=="assistant" else "🙂"):
        st.markdown(msg["content"])
        if msg.get("category") and msg["category"]!="Welcome": st.caption("Topic: "+msg["category"])
        if msg["role"]=="assistant" and msg.get("suggestions"):
            st.markdown("**You could also ask:**")
            cols=st.columns(min(3,len(msg["suggestions"])))
            for j,s in enumerate(msg["suggestions"]):
                with cols[j%len(cols)]:
                    if st.button(s,key=f"sugg_{mi}_{j}",use_container_width=True):
                        st.session_state.pending=s; st.rerun()

typed=st.chat_input("Ask about fees, exams, attendance, scholarships...")
question=st.session_state.pending or typed
if question:
    st.session_state.pending=None
    st.session_state.messages.append({"role":"user","content":question})
    reply,suggestions,category=answer_question(question)
    st.session_state.messages.append({"role":"assistant","content":reply,"suggestions":suggestions,"category":category})
    st.rerun()

with st.expander("🧪 Try a quick demo"):
    cols=st.columns(3)
    for i,q in enumerate(["My exam result isn't showing","I forgot my password","How do I get placement help?"]):
        with cols[i]:
            if st.button(q,key=f"demo_{i}",use_container_width=True):
                st.session_state.pending=q; st.rerun()
st.divider()
st.caption("Academic prototype • Answers are general guidance. Confirm fees, deadlines, eligibility, and policies with your college.")
