# CampusConnect Interactive
An improved student-support chatbot with topic navigation, clickable questions, follow-up prompts, varied answers, chat history, and fallback handling.

## Run
pip install -r requirements.txt
streamlit run app.py

## Deploy
Keep app.py, faqs.json, and requirements.txt in the root of your GitHub repository. In Streamlit Community Cloud choose branch `main` and main file `app.py`.

## Limitation
This is a retrieval-based FAQ chatbot, not a generative AI model. It has no live college data. Confirm sample guidance before real use.
