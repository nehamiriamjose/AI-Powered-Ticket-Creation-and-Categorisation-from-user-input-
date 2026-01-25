# AI-Powered-Ticket-Creation-and-Categorisation-from-user-input-
To automate the initial phase of ticket management by developing an AI system that can understand and process natural language requests from users. The system will automatically generate structured service tickets—including a title, description, category, and initial priority—from unstructured user input provided via chat, email, or a web form. 

Tech Stack
Backend
Python
FastAPI
SQLite
Scikit-lear
spaCy

Frontend
Streamlit
HTML/CSS

ML & NLP
TF-IDF Vectorization
Logistic Regression
spaCy NER
Hybrid ML + Rule-based logic

Setup Instructions
Create Virtual Environment
python -m venv venv
venv\Scripts\activate      # Windows
OR
source venv/bin/activate   # Linux/Mac


How to Run the Backend
uvicorn backend.api:app --reload

How to Run the Frontend (Streamlit)
Open a new terminal and run:streamlit run streamlit_app.py

Application Flow
User logs into Streamlit UI
User submits a ticket description
Frontend calls FastAPI backend
Backend processes input using ML & NLP
Ticket is stored in SQLite database
UI displays ticket details, priority, and SLA
Support/Admin update ticket status

Future Enhancements
Email alerts
Advanced role-based access
Cloud deployment
Analytics and reporting module

link:https://github.com/nehamiriamjose/AI-Powered-Ticket-Creation-and-Categorisation-from-user-input-/tree/main
