# CampusConnect — AI Chatbot for Student Support Services

A beginner-friendly college project built with Python, Streamlit, and scikit-learn.

## Features
- Chat-style user interface
- FAQ support for admissions, fees, examinations, results, attendance, library, scholarships, certificates, portal login, class timetable, placements, and student support
- NLP-style text matching using TF-IDF and cosine similarity
- Fallback response for questions that do not match the stored FAQs
- Displays a similarity score for demonstration and viva purposes
- No API key or paid service required

## Requirements
- Python 3.9 or newer
- Internet connection only to install the packages (the app itself runs locally)

## Run the project
1. Extract the ZIP folder.
2. Open a terminal inside the project folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Customize the FAQs
Edit `faqs.json`. Each FAQ contains:
- `category`: topic label
- `question`: representative question
- `examples`: alternate phrasings
- `answer`: response shown to the user

Keep answers accurate and replace sample guidance with verified college policies before any real deployment.

## How it works
1. The app loads stored FAQ questions and example phrasings.
2. TF-IDF converts the text into numerical vectors.
3. Cosine similarity compares the user's question with the stored phrases.
4. The closest matching answer is returned if its score passes a small threshold.
5. If no match is strong enough, the app recommends contacting official college support.

## Limitations
- It is a retrieval-based FAQ chatbot, not a generative AI assistant.
- The similarity score is not a probability or guarantee of correctness.
- It does not connect to student databases, payment systems, or live university portals.
- It may misunderstand spelling errors, mixed-language questions, or questions outside its FAQ dataset.
- Do not enter passwords, OTPs, or sensitive personal data.

## Suggested viva demonstration
Try these questions:
- How can I apply for admission?
- I forgot my student portal password.
- When are my exams?
- Where can I get scholarship information?
- Tell me a joke about space (should trigger the fallback response).

## Project structure
- `app.py` — Streamlit application and matching logic
- `faqs.json` — editable FAQ dataset
- `requirements.txt` — Python dependencies
- `PROJECT_REPORT.docx` — project report
