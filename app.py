"""
JIET Universe RAG Chatbot — Streamlit App
==========================================
Stack:
  • LLM       : Google Gemini 1.5 Flash (via google-generativeai)
  • Embeddings: sentence-transformers (all-MiniLM-L6-v2)  [free, local]
  • Vector DB : FAISS (in-memory)
  • Framework : Streamlit
  • RAG        : LangChain (document splitting + retrieval chain)

Install dependencies:
    pip install streamlit google-generativeai langchain langchain-google-genai
                langchain-community faiss-cpu sentence-transformers pypdf

Run:
    streamlit run jiet_chatbot_app.py
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import time
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JIET Universe Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  [data-testid="stAppViewContainer"] {
      background: linear-gradient(135deg, #f0f6ff 0%, #ffffff 100%);
  }
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #00529C 0%, #003a70 100%);
  }
  [data-testid="stSidebar"] * { color: #ffffff !important; }
  [data-testid="stSidebar"] .stTextInput input {
      background: rgba(255,255,255,0.15);
      border: 1px solid rgba(255,255,255,0.4);
      color: #fff !important;
  }
  [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.25); }

  /* ── Header ── */
  .jiet-header {
      background: linear-gradient(90deg, #00529C, #0076cc);
      color: white;
      padding: 18px 28px;
      border-radius: 12px;
      margin-bottom: 20px;
      display: flex;
      align-items: center;
      gap: 16px;
      box-shadow: 0 4px 18px rgba(0,82,156,0.25);
  }
  .jiet-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
  .jiet-header p  { margin: 0; font-size: 0.9rem; opacity: 0.85; }

  /* ── Chat bubbles ── */
  .chat-user {
      background: linear-gradient(135deg, #00529C, #0076cc);
      color: white;
      padding: 12px 16px;
      border-radius: 18px 18px 4px 18px;
      margin: 8px 0 8px 40px;
      font-size: 0.95rem;
      box-shadow: 0 2px 8px rgba(0,82,156,0.2);
  }
  .chat-bot {
      background: white;
      color: #1a1a2e;
      padding: 14px 18px;
      border-radius: 18px 18px 18px 4px;
      margin: 8px 40px 8px 0;
      font-size: 0.95rem;
      border-left: 4px solid #FF6600;
      box-shadow: 0 2px 10px rgba(0,0,0,0.07);
  }
  .chat-label {
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      margin-bottom: 4px;
      opacity: 0.65;
  }

  /* ── Source cards ── */
  .source-card {
      background: #f0f6ff;
      border: 1px solid #cce0ff;
      border-radius: 8px;
      padding: 8px 12px;
      margin: 4px 0;
      font-size: 0.80rem;
      color: #00529C;
  }

  /* ── Stats bar ── */
  .stat-box {
      background: white;
      border-radius: 10px;
      padding: 12px 18px;
      text-align: center;
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
      border-top: 3px solid #00529C;
  }
  .stat-box h3 { margin: 0; font-size: 1.4rem; color: #00529C; }
  .stat-box p  { margin: 0; font-size: 0.75rem; color: #666; }

  /* ── Input ── */
  [data-testid="stChatInput"] textarea {
      border-radius: 24px !important;
      border: 2px solid #00529C !important;
  }

  /* ── Spinner ── */
  .typing-dot {
      display: inline-block;
      width: 8px; height: 8px;
      border-radius: 50%;
      background: #00529C;
      margin: 0 2px;
      animation: bounce 1.2s infinite;
  }
  .typing-dot:nth-child(2){ animation-delay:.2s; }
  .typing-dot:nth-child(3){ animation-delay:.4s; }
  @keyframes bounce {
    0%,80%,100%{ transform:translateY(0); }
    40%         { transform:translateY(-8px); }
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  KNOWLEDGE BASE  (JIET RAG Documents)
# ─────────────────────────────────────────────────────────────────────────────
JIET_KNOWLEDGE_BASE = """
# JIET Universe — Complete Institutional Knowledge Base

## 1. INSTITUTIONAL OVERVIEW
- Full Name: Jodhpur Institute of Engineering and Technology (JIET Universe)
- Established: 2003
- Location: NH-62, Pali Road, Mogra, Jodhpur – 342802, Rajasthan, India
- Type: Private Multi-Disciplinary Institution
- Phone (Admissions): +91-9799999186, +91-9773378530
- Student Cell: 9799999189
- Email: info@jietjodhpur.ac.in | admission@jietjodhpur.ac.in
- Website: www.jietjodhpur.ac.in
- WhatsApp: +91-9799999186
- Accreditation: NAAC Accredited; NBA Accredited Programs; AICTE Approved
- Alumni Network: 12,000+ worldwide
- Admission 2026-2027 is currently open.

## 2. VISION, MISSION & CORE VALUES
Vision: To become a globally recognised institution in technical and professional education and provide career and research-oriented, value-based education to serve society.

Mission:
- Develop a holistic educational approach that blends fundamentals with hands-on experience.
- Build a diverse academic environment fostering problem-solving, team spirit, and leadership.
- Promote exchange of ideas, innovation, research, and entrepreneurial skills for global challenges.
- Inculcate ethical values and a sense of responsibility towards society.

Quality Policy: To provide quality education through faculty development, updation of facilities, and continuous improvement, meeting apex bodies norms.

Core Values: Commitment, Respect, Excellence, Accountability, Diversity.

## 3. ENGINEERING PROGRAMS (B.Tech)
Duration: 4 Years | Eligibility: 10+2 with min 50% in Physics, Chemistry, Mathematics.
- B.Tech Computer Science & Engineering (CSE)
- B.Tech Artificial Intelligence & Machine Learning (AI-ML)
- B.Tech Data Science
- B.Tech Cyber Security
- B.Tech Electronics & Communication Engineering (ECE)
- B.Tech Electrical Engineering (EE)
- B.Tech Mechanical Engineering (ME)
- B.Tech Civil Engineering (CE)

M.Tech Programs: Power System, Data Science, Thermal Engineering, Digital Communication.
Eligibility for M.Tech: BE/B.Tech in relevant branch with valid GATE score.

Ph.D. Programs: Ph.D. in CSE, EE, ME, ECE.

## 4. OTHER PROGRAMS
Under-Graduate:
- BCA (Bachelor of Computer Applications) – 3 Years
- BBA (Bachelor of Business Administration) – 3 Years
- B.Des (Bachelor of Design) – 4 Years
- B.Sc. Nursing – 4 Years (INC Approved)
- Bachelor of Hotel Management (BHM) – 4 Years
- MBBS – 5.5 Years (NMC Approved via JIET Hospital)

Post-Graduate:
- MCA (Master of Computer Applications) – 2 Years
- MBA (Master of Business Administration) – 2 Years

Diploma:
- Diploma in Pharmacy (D.Pharma) – 2 Years (PCI Approved)

## 5. FEES STRUCTURE
Fees vary by program. Visit www.jietjodhpur.ac.in/fees-structure for the latest fee details.
Approximate fee range for B.Tech: ₹60,000 – ₹1,20,000 per year (may vary).
Scholarships available up to 100% tuition fee waiver.

## 6. SCHOLARSHIPS
- Institutional Scholarships: Merit-based; up to 100% tuition fee waiver.
- Government Scholarships: SC/ST/OBC/EWS and minority welfare schemes.
- AICTE Scholarships: Pragati (girl students) and Saksham (differently-abled).
- Corporate Scholarships: Industry-sponsored awards for deserving students.
Details: www.jietjodhpur.ac.in/scholarships

## 7. TRAINING & PLACEMENT
- Highest Package: ₹33 LPA
- Average Package: ₹7 LPA
- On-campus Placements: 1,100+ (engineering)
- Companies Visited: 300+
- Alumni: 12,000+ worldwide

Top Recruiters: Airbus, Tekion, Publicis Sapient, Infosys, Optum, BYJU's, Synopsys,
Celebal Technologies, UltraTech, Tata Motors, Josh Technology, Polaris, JTEKT,
AU Small Finance Bank, NCR Corporation, HM Health Solutions, PolicyBazaar, iCubesWire.

Top Training Partners: Celebal Technologies, IIT Jodhpur, ISRO, DRDO, CSIR-CEERI Pilani,
CADC, All India Radio, NBC.

The Training & Placement Cell provides round-the-clock support, organises soft skills,
aptitude, and technical training, facilitates internships, industry visits, and live projects.

## 8. INDUSTRY CONNECT & MoUs
Partners: Infosys, Cisco, Celebal Technologies, BOSCH, SAS, Open Innovation Lab,
IDEA Forage Innovation, Gehlot Machinery, BAFNA Electric, AIESEC-Jodhpur.
JIET has signed an MoU with AIESEC-Jodhpur for global student opportunities.
Details: www.jietjodhpur.ac.in/mou

## 9. ACCREDITATIONS & AFFILIATIONS
- NAAC Accredited
- NBA Accredited (selected engineering programs)
- AICTE Approved (Engineering, Management, Pharmacy, Hotel Management)
- NMC Approved (MBBS via JIET Hospital)
- INC Approved (B.Sc. Nursing)
- PCI Approved (D.Pharma)
- Affiliated to RTU (Rajasthan Technical University)
- NIRF Participant Institution

## 10. CAMPUS & FACILITIES
- Location: NH-62, Pali Road, Mogra, Jodhpur (on national highway)
- Library: Digital and physical library with e-resources and journals
- AICTE-IDEA Lab (Lab ID: 2024000120): Innovation and maker space
- Advanced discipline-specific laboratories
- Sports grounds, gymnasium, indoor sports
- NCC Unit on campus
- JIET Hospital & Cancer Research Centre (attached multispecialty hospital)

## 11. STUDENT LIFE
Clubs & Activities:
- NCC (National Cadet Corps)
- Student Council
- Design Club — Meraki Annual Design Fest
- Technical Clubs (Robotics, Coding, IoT)
- Cultural & Literary Clubs
- SPIC MACAY Chapter
- Entrepreneurship & Innovation Cell
- Sports Teams

Learning Support:
- Choice Based Credit System (CBCS)
- MOOC Courses (NPTEL, Coursera) with academic credit
- Self-learning portal
- Feedback & Grievance Redressal System
- JIET Handbook for students
- Kaizen Annual Magazine

## 12. INNOVATION & RESEARCH
- AICTE-IDEA Lab for prototyping and design thinking
- Startup Programs with structured entrepreneurship curriculum
- Multiple student-founded startups incubated at JIET
- Research published in national and international journals
- IFSTEM 2025: National Conference on Innovation and Sustainability
- Collaborations with IIT Jodhpur for joint research
- International exchange: Italian architecture students, AIESEC global internships
- Students completed Green Leaders sustainability project in Indonesia

## 13. NOTABLE ACHIEVEMENTS
- Gautam Sariyala (JIET alumnus): AIR-22, UPSC IES 2024
- Kuru Bhandari (JIET alumnus): Admitted to National University of Singapore for Master's
- JIET students excelled in NPTEL November 2024 results
- MBA Batch 2023-25: Outstanding campus placements
- ECE students: Internship at CSIR-CEERI, Pilani
- Students at IGDC (India Game Developer Conference)
- Italian architecture students' international exchange at JIET campus

## 14. ADMISSION PROCESS 2026-2027
How to Apply: Visit https://admissions.jietjodhpur.ac.in
- B.Tech: 10+2 with 50% in PCM; JEE Main / REAP score accepted
- M.Tech: BE/B.Tech + valid GATE score
- MBA: Any bachelor's degree; CAT/MAT/CMAT score preferred
- B.Sc. Nursing: 10+2 with PCB (min 45%); INC norms
- MBBS: 10+2 PCB + NEET qualification mandatory
- BCA/MCA/BBA/BHM/B.Des: 10+2 from any recognised board

Contact Admissions:
- Phone: +91-9799999186 | +91-9773378530
- Email: admission@jietjodhpur.ac.in
- WhatsApp: +91-9799999186

## 15. DEPARTMENTS
1. Department of Computer Science & Engineering — B.Tech CSE, AI-ML, Data Science, Cyber Security
2. Department of Electronics & Communication Engineering — B.Tech ECE, M.Tech Digital Communication
3. Department of Electrical Engineering — B.Tech EE, M.Tech Power Systems
4. Department of Mechanical Engineering — B.Tech ME, M.Tech Thermal Engineering
5. Department of Civil Engineering — B.Tech Civil
6. School of Design — B.Des (Product, Fashion, Interior Design)
7. Department of Management Studies — MBA, BBA
8. Department of Computer Applications — BCA, MCA
9. School of Hotel Management — BHM
10. School of Nursing — B.Sc. Nursing
11. Department of Pharmacy — D.Pharma
12. JIET Hospital & Cancer Research Centre — MBBS, multispecialty healthcare

## 16. GOVERNANCE
- Governing Board: Senior leadership comprising academics, industry leaders, and promotors
- Academic Mentors: World-class mentors guiding curriculum strategy
- IQAC: Internal Quality Assurance Cell for continuous quality improvement
- Student Council: Elected body representing students
- Mandatory Disclosure: Published per AICTE guidelines
- NIRF data submitted annually
"""

# ─────────────────────────────────────────────────────────────────────────────
#  RAG SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are JIET Assistant, the official AI-powered chatbot for JIET Universe 
(Jodhpur Institute of Engineering and Technology), located at NH-62, Pali Road, 
Mogra, Jodhpur – 342802, Rajasthan, India.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROLE & PERSONALITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- You are a friendly, knowledgeable, and professional campus counsellor.
- You speak in a warm, encouraging, and student-friendly tone.
- You greet users respectfully (use "Namaste" for first message).
- You are patient, helpful, and never dismissive.
- You can communicate in both English and Hindi if the user prefers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIMARY RESPONSIBILITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Answer questions about:
1. Academic Programs     — B.Tech, M.Tech, Ph.D., BCA, MCA, MBA, BBA, 
                           B.Des, B.Sc Nursing, Hotel Management, D.Pharma, MBBS
2. Admissions            — eligibility, process, important dates, entrance exams
3. Fees & Scholarships   — fee structure, scholarship types, application process
4. Placements            — packages, recruiters, training cell activities
5. Campus & Facilities   — labs, library, IDEA Lab, hostel, sports, hospital
6. Student Life          — clubs, NCC, events, Meraki fest, SPIC MACAY
7. Industry Connect      — MoUs, internships, industry partners
8. Research & Innovation — IDEA Lab, startups, publications, conferences
9. Contact & Location    — address, phone numbers, email, website

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES — FOLLOW ALWAYS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. CONTEXT FIRST: Answer ONLY using the retrieved context provided below.
2. NO HALLUCINATION: Never invent fees, dates, faculty names, or statistics 
   not present in the context.
3. OUT OF SCOPE: If the answer is not in the context, respond exactly with:
   "I don't have that specific information right now. For accurate details, 
    please contact JIET directly:
    📞 +91-9799999186
    📧 admission@jietjodhpur.ac.in
    🌐 www.jietjodhpur.ac.in"
4. OFF-TOPIC: If the question is completely unrelated to JIET (e.g., general 
   coding help, news, jokes), politely say:
   "I'm here specifically to help with JIET-related queries. For other topics, 
    I'd recommend using a general search engine. Is there anything about JIET 
    I can help you with?"
5. ADMISSION LINK: Always mention https://admissions.jietjodhpur.ac.in 
   when answering any admission-related question.
6. NEVER make promises about results, selections, or guaranteed placements.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Use bullet points (•) or numbered lists for multiple items.
- Use bold (**text**) for important terms, package figures, and deadlines.
- Keep responses concise — 3 to 8 lines for simple queries.
- For complex topics (e.g., listing all programs), use structured sections.
- Always end with a helpful follow-up offer:
  Example: "Would you like more details about fees or the admission process?"
- Use relevant emojis sparingly to keep tone friendly:
  🎓 for programs, 💼 for placements, 📞 for contact, 🏛️ for campus, 
  💰 for fees/scholarships, 🔬 for research.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE INTERACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User: "What B.Tech courses are available?"
You: "🎓 JIET offers the following B.Tech programs (4-year full-time):
  • Computer Science & Engineering (CSE)
  • Artificial Intelligence & Machine Learning (AI-ML)
  • Data Science
  • Cyber Security
  • Electronics & Communication Engineering (ECE)
  • Electrical Engineering (EE)
  • Mechanical Engineering (ME)
  • Civil Engineering (CE)
  
  Eligibility: 10+2 with min. 50% in Physics, Chemistry & Mathematics.
  Ready to apply? Visit 👉 https://admissions.jietjodhpur.ac.in
  Would you like details on any specific branch?"

User: "What is the highest placement package?"
You: "💼 JIET has an excellent placement record!
  • **Highest Package:** ₹33 LPA
  • **Average Package:** ₹7 LPA
  • **1,100+** on-campus placements in engineering alone
  • **300+** companies visit the campus
  Top recruiters include Airbus, Infosys, Publicis Sapient, Celebal Technologies, 
  Tata Motors, Optum, and many more.
  Would you like to know about our Training & Placement Cell activities?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT (from RAG retrieval — use this to answer):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION HISTORY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chat_history}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STUDENT QUESTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{question}

JIET Assistant Answer:"""

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: Build vector store (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Setting up JIET Knowledge Base...")
def build_vector_store():
    """Split knowledge base into chunks and embed into FAISS."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n- ", "\n", " "],
    )
    chunks = splitter.split_text(JIET_KNOWLEDGE_BASE)
    docs   = [Document(page_content=c) for c in chunks]

    embeddings   = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


@st.cache_resource(show_spinner="🤖 Loading Gemini LLM...")
def build_llm(api_key: str):
    """Initialise Gemini 1.5 Flash LLM via LangChain."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3,
        max_output_tokens=1024,
        convert_system_message_to_human=True,
    )


def build_chain(api_key: str):
    """Build the ConversationalRetrievalChain."""
    vector_store = build_vector_store()
    llm          = build_llm(api_key)
    retriever    = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=SYSTEM_PROMPT,
    )
    memory = ConversationBufferWindowMemory(
        k=6,
        memory_key="chat_history",
        return_messages=False,
        output_key="answer",
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False,
    )
    return chain


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "chain"         not in st.session_state: st.session_state.chain         = None
if "total_queries" not in st.session_state: st.session_state.total_queries = 0
if "api_key_set"   not in st.session_state: st.session_state.api_key_set   = False


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 JIET Universe")
    st.markdown("**AI Chatbot — Powered by Gemini**")
    st.markdown("---")

    # API Key Hardcoded
    st.markdown("### 🔑 System Connection")
    api_key = "AIzaSyBgwcrf6JtKrsXvuQlsazIiWHQOSD9pFKY"

    if not st.session_state.api_key_set or st.session_state.chain is None:
        with st.spinner("Initialising advanced RAG system..."):
            try:
                st.session_state.chain       = build_chain(api_key)
                st.session_state.api_key_set = True
                st.success("✅ Advanced Chatbot ready!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.markdown("---")
    st.markdown("### 💬 Quick Questions")
    quick_qs = [
        "What B.Tech courses are offered?",
        "What is the highest placement package?",
        "How do I apply for admission?",
        "What scholarships are available?",
        "Tell me about the JIET campus",
        "What are M.Tech programs?",
        "Who are the top recruiters?",
        "What is the fee structure?",
    ]
    for q in quick_qs:
        if st.button(q, use_container_width=True):
            st.session_state["quick_input"] = q

    st.markdown("---")
    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        st.metric("Est. 2003", "20+ yrs")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.total_queries = 0
        st.session_state.chain         = None
        st.session_state.api_key_set   = False
        st.rerun()

    st.markdown("---")
    st.markdown("""
**📞 Contact JIET**
- 📱 +91-9799999186
- 📧 info@jietjodhpur.ac.in
- 🌐 [jietjodhpur.ac.in](https://www.jietjodhpur.ac.in)
""")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PAGE — HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="jiet-header">
  <div style="font-size:2.5rem;">🎓</div>
  <div>
    <h1>JIET Universe Assistant</h1>
    <p>Ask me anything about admissions, programs, placements, scholarships, campus life & more!</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Stats row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-box"><h3>₹33 LPA</h3><p>Highest Package</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-box"><h3>12,000+</h3><p>Alumni Worldwide</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-box"><h3>300+</h3><p>Recruiters</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-box"><h3>100%</h3><p>Max Scholarship</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CHAT DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="chat-bot">
      <div class="chat-label">🤖 JIET ASSISTANT</div>
      Namaste! 🙏 I'm your JIET Universe AI Assistant, powered by Google Gemini.<br><br>
      I can help you with:<br>
      &bull; <b>Courses & Programs</b> — B.Tech, M.Tech, MBA, BCA, Nursing & more<br>
      &bull; <b>Admissions 2026-27</b> — eligibility, process, important dates<br>
      &bull; <b>Placements</b> — packages, recruiters, training cell<br>
      &bull; <b>Scholarships</b> — up to 100% fee waiver options<br>
      &bull; <b>Campus Life</b> — facilities, clubs, hostel, IDEA Lab<br><br>
      What would you like to know? 😊
    </div>
    """, unsafe_allow_html=True)

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-user">
              <div class="chat-label">👤 YOU</div>
              {msg["content"]}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-bot">
              <div class="chat-label">🤖 JIET ASSISTANT</div>
              {msg["content"]}
            </div>""", unsafe_allow_html=True)
            # Show sources if available
            if msg.get("sources"):
                with st.expander("📚 Source Chunks Used", expanded=False):
                    for i, src in enumerate(msg["sources"][:3], 1):
                        st.markdown(
                            f'<div class="source-card">📄 <b>Chunk {i}:</b> '
                            f'{src[:200]}...</div>',
                            unsafe_allow_html=True
                        )

# ─────────────────────────────────────────────────────────────────────────────
#  CHAT INPUT
# ─────────────────────────────────────────────────────────────────────────────
# Handle quick questions from sidebar
quick_val = st.session_state.pop("quick_input", None)

user_input = st.chat_input(
    "Ask about JIET programs, admissions, placements...",
    key="chat_input"
) or quick_val

if user_input:
    if st.session_state.chain is None:
        st.warning("⚠️ Chatbot is still initialising. Please wait a moment.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate response
        with st.spinner(""):
            st.markdown("""
            <div class="chat-bot">
              <div class="chat-label">🤖 JIET ASSISTANT</div>
              <span class="typing-dot"></span>
              <span class="typing-dot"></span>
              <span class="typing-dot"></span>
            </div>""", unsafe_allow_html=True)
            try:
                result  = st.session_state.chain({"question": user_input})
                answer  = result.get("answer", "I couldn't generate a response.")
                sources = [
                    doc.page_content
                    for doc in result.get("source_documents", [])
                ]
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": sources,
                })
                st.session_state.total_queries += 1
            except Exception as e:
                error_msg = (
                    f"⚠️ Error generating response: {str(e)}. "
                    "Please check your API key or try again."
                )
                st.session_state.messages.append({
                    "role": "assistant", "content": error_msg, "sources": []
                })
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.8rem; padding:8px;'>
  🎓 JIET Universe AI Chatbot &nbsp;|&nbsp;
  Powered by <b>Google Gemini 1.5 Flash</b> + <b>RAG (FAISS + LangChain)</b> &nbsp;|&nbsp;
  <a href='https://www.jietjodhpur.ac.in' target='_blank'>www.jietjodhpur.ac.in</a>
</div>
""", unsafe_allow_html=True)
