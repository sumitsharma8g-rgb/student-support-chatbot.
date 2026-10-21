# Project 1: AI Chatbot for Student Support Services

An end-to-end Python implementation of an **AI Chatbot for Student Support Services** designed for university helpdesks. The system combines Natural Language Processing (TF-IDF vectorization, n-gram matching, and intent classification), regex-based Named Entity Extraction, multi-turn dialogue state tracking, and a relational **SQLite** database backend.

## Project Structure

```
Project_1_AI_Student_Support_Chatbot/
├── student_support_chatbot.py   # Main chatbot application (NLP + SQLite + CLI + Web UI + Demo)
├── intents_dataset.json         # Structured NLP training dataset (13 intents, patterns & responses)
├── test_chatbot.py              # Automated unit test suite (unittest)
├── requirements.txt             # Python dependencies (runs on Python 3.8+ standard library)
└── README.md                    # Project documentation and usage guide
```

## Key Features

1. **Hybrid NLP Intent Classification (13 Intents)**:
   - Supports `greeting`, `admissions`, `fees_and_scholarships`, `course_registration`, `exams_and_grading`, `library_services`, `hostel_and_facilities`, `it_helpdesk`, `career_and_placements`, `counseling_and_wellness`, `create_ticket`, `check_ticket`, and `goodbye`.
   - Automatically uses `scikit-learn` (`TfidfVectorizer` + `LogisticRegression` + Cosine Similarity) when installed, or falls back seamlessly to its built-in pure-Python unigram/bigram TF-IDF Cosine Similarity engine.
2. **Named Entity & Slot Extraction**:
   - Automatically detects **Student IDs** (`STU2026001`–`STU2026003`), **Course Codes** (`CS101`, `CS204`, `CS305`, `MA102`, `EC202`, `MB301`), **Academic Programs** (`B.Tech`, `MBA`, `MCA`, `B.Sc`, `BBA`), **Support Ticket IDs** (`TKT-1001`+), and email addresses.
3. **SQLite Relational Database (`sqlite3`)**:
   - Stores and queries live tables for `courses`, `program_fees`, `students`, `support_tickets`, and `chat_logs`.
4. **Multi-Turn Slot Filling & Ticket Routing**:
   - Collects issue descriptions across turns and automatically assigns new support tickets (`TKT-1004`, etc.) to the relevant department (Admissions Cell, Finance & Scholarship Desk, Controller of Examinations, Campus IT Support Team, etc.).

## How to Run

### 1. Run Automated Benchmark Evaluation & Demo Walkthrough
```bash
python3 student_support_chatbot.py --demo
```

### 2. Run Interactive Terminal (CLI) Chatbot
```bash
python3 student_support_chatbot.py --cli
```

### 3. Launch Built-In Web Browser UI
```bash
python3 student_support_chatbot.py --web --port 8080
```
Then open `http://localhost:8080` in your browser.

### 4. Run Unit Tests
```bash
python3 test_chatbot.py
```
