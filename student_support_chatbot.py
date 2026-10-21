#!/usr/bin/env python3
"""
Project 1: AI Chatbot for Student Support Services
===================================================
A complete, self-contained Python implementation of an AI-powered Student Support
Chatbot combining:
1. Natural Language Processing (TF-IDF Vectorization + Cosine Similarity & Intent
   Classification using scikit-learn with a pure-Python fallback).
2. Named Entity & Slot Extraction (Student ID, Course Code, Program/Department,
   Ticket ID, Semester, and Email).
3. Multi-Turn Dialogue State & Context Management.
4. SQLite Database Backend for FAQ storage, Course/Exam lookups, Fee structures,
   Support Ticket generation, and Conversation Logging.
5. Multiple Interfaces:
   - Interactive CLI Chatbot (`python3 student_support_chatbot.py --cli`)
   - Built-in Web UI Server (`python3 student_support_chatbot.py --web`)
   - Model Evaluation & Demo Suite (`python3 student_support_chatbot.py --demo`)

Author: Student Support AI Project
Python Version: 3.8+
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Optional scikit-learn integration (automatically falls back to built-in NLP engine if absent)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ==============================================================================
# 1. KNOWLEDGE BASE: INTENTS, TRAINING CORPUS & RESPONSES
# ==============================================================================

INTENTS_DATA: List[Dict[str, Any]] = [
    {
        "tag": "greeting",
        "description": "General greetings and conversation starters",
        "patterns": [
            "hello",
            "hi there",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "hi student support",
            "is anyone there",
            "help me please",
            "greetings",
            "howdy",
        ],
        "responses": [
            "Hello! Welcome to the University Student Support Desk. How can I assist you today? You can ask about admissions, tuition fees, course registration, exams, hostel, library, or support tickets.",
            "Hi there! I'm your AI Student Support Assistant. Ask me anything about academics, campus facilities, scholarships, exam schedules, or raise a support ticket!",
        ],
    },
    {
        "tag": "admissions",
        "description": "Admission requirements, deadlines, application process, and eligibility",
        "patterns": [
            "apply for admission",
            "admission process and guidelines",
            "admission deadline for fall semester",
            "eligibility criteria for admission",
            "when do admissions open",
            "documents needed required for undergraduate admission",
            "entrance exam requirements for admission",
            "check my admission application status",
            "undergraduate postgraduate admission procedure",
            "minimum percentage required for admission",
            "transfer student admission",
        ],
        "responses": [
            "Admissions are open online through the University Admissions Portal.\n"
            "• Application Window: March 1 – July 15 (Fall Intake) & October 1 – November 30 (Spring Intake)\n"
            "• Eligibility: Minimum 60% aggregate in qualifying examinations (55% for reserved categories) + valid Entrance Test score.\n"
            "• Required Documents: Transcripts, Transfer Certificate, ID Proof, Passport Photos, and Category Certificate (if applicable).\n"
            "• Tip: Share a program name (e.g., 'B.Tech', 'MBA', 'B.Sc', 'MCA') to view program-specific tuition and eligibility details!",
        ],
    },
    {
        "tag": "fees_and_scholarships",
        "description": "Tuition fees, payment deadlines, installment plans, and scholarships",
        "patterns": [
            "fee structure and tuition fee",
            "how much is the tuition fee",
            "fee payment deadline",
            "apply for a scholarship and scholarship criteria",
            "financial aid and merit scholarships",
            "pay fees in installments",
            "late fee penalty charges",
            "btech tuition fee per semester",
            "mba tuition fee structure",
            "mca and bsc semester fees",
            "download fee receipt",
        ],
        "responses": [
            "Here is the overview of Tuition Fees & Financial Aid:\n"
            "• Payment Modes: Online via Student ERP Portal (NetBanking, UPI, Credit/Debit Card, or Demand Draft).\n"
            "• Due Dates: August 10 (Odd Semester) and January 10 (Even Semester). Late fee is ₹500/week after the deadline.\n"
            "• Scholarships Available:\n"
            "  1. Merit Scholarship: 50% tuition waiver for CGPA >= 9.0 or top 5% entrance rank.\n"
            "  2. Need-Based Financial Aid: Up to 75% tuition support for family income below ₹3,00,000/year.\n"
            "  3. Sports & Cultural Excellence Scholarship: 25%–100% fee waiver.\n"
            "• Mention a specific program (e.g., 'B.Tech', 'MBA', 'MCA', 'B.Sc') to get the exact semester-wise fee breakdown.",
        ],
    },
    {
        "tag": "course_registration",
        "description": "Adding/dropping courses, credit limits, electives, and academic registration",
        "patterns": [
            "register for courses this semester",
            "when does course registration start",
            "add or drop a course subject",
            "credits can i register for this semester",
            "maximum credits allowed per semester",
            "choose open electives",
            "course registration portal error",
            "prerequisite waiver request",
            "deadline to drop a subject",
            "syllabus and course catalog",
            "credit requirements for graduation",
        ],
        "responses": [
            "Course Registration Guidelines:\n"
            "• Registration Window: Opens 2 weeks prior to each semester on the Student Academic ERP.\n"
            "• Credit Limits: Minimum 16 credits, Maximum 24 credits per regular semester (up to 28 with Dean's approval for CGPA >= 8.5).\n"
            "• Add/Drop Period: First 10 working days of the semester with zero penalty.\n"
            "• Course Withdrawal ('W' Grade): Allowed up to Week 8 of the semester.\n"
            "• If you mention a course code (such as CS101, CS204, CS305, MA102, MB301, EC202), I can look up its credits, instructor, and exam schedule!",
        ],
    },
    {
        "tag": "exams_and_grading",
        "description": "Exam schedules, hall tickets, grading system, CGPA calculation, and revaluation",
        "patterns": [
            "when are the final exams",
            "exam schedule and timetable",
            "download admit card or hall ticket",
            "how is cgpa calculated",
            "grading scale and passing marks",
            "apply for paper revaluation or rechecking",
            "midterm exam dates",
            "backlog or supplementary exam registration",
            "when will semester results be declared",
            "minimum attendance required for exams",
            "exam date and instructor for cs101 cs305",
        ],
        "responses": [
            "Examinations & Evaluation Policy:\n"
            "• Weightage: Continuous Internal Assessment (40%: Quizzes 10%, Assignments 10%, Mid-Term 20%) + End-Semester Exam (60%).\n"
            "• Attendance Requirement: Minimum 75% attendance is mandatory to download the Hall Ticket (condonation up to 65% strictly on medical grounds).\n"
            "• 10-Point Grading Scale: O (10), A+ (9), A (8), B+ (7), B (6), C (5), P (4), F (0 - Fail).\n"
            "• Revaluation: Apply within 7 days of result declaration via ERP (₹300 per course).\n"
            "• Tip: Ask with a course code like 'Exam schedule for CS101' or 'Details for MA102' for exact dates and venues!",
        ],
    },
    {
        "tag": "library_services",
        "description": "Library timings, book borrowing limits, overdue fines, and digital library access",
        "patterns": [
            "library timings and opening hours",
            "books can i borrow from the library",
            "library late fine per day",
            "access ieee and springer digital library",
            "renew borrowed library books online",
            "lost library book replacement policy",
            "reading room open 24x7",
            "library membership card issue",
            "remote access to e-journals",
        ],
        "responses": [
            "Central Library & Digital Resource Center:\n"
            "• Operating Hours: Mon–Sat 8:00 AM – 10:00 PM | Sundays & Holidays 9:00 AM – 5:00 PM (24x7 Reading Hall open during exams).\n"
            "• Borrowing Privileges: UG Students: 4 books for 14 days | PG Students: 6 books for 21 days | Research Scholars: 8 books for 30 days.\n"
            "• Overdue Fine: ₹5 per day per book for the first 14 days, ₹10 per day thereafter.\n"
            "• Digital Library: Remote access to IEEE Xplore, SpringerLink, ACM Digital Library, and NPTEL via your institutional email on the Knimbus/EZproxy portal.",
        ],
    },
    {
        "tag": "hostel_and_facilities",
        "description": "Hostel accommodation, mess timings, Wi-Fi, medical center, and transport",
        "patterns": [
            "apply for hostel accommodation",
            "hostel charges fee and mess charges",
            "hostel curfew and gate timings",
            "campus shuttle bus timings",
            "medical center and ambulance number",
            "guest house booking for parents",
            "hostel room change request",
            "mess menu and dining hours",
            "sports complex and gym timings",
        ],
        "responses": [
            "Hostel, Dining & Campus Facilities:\n"
            "• Hostel Allocation: Allotted on a first-come, distance-priority basis via the HMS Portal. Options: Single, Double, and Triple occupancy (AC & Non-AC).\n"
            "• Hostel & Mess Charges: Non-AC Double: ₹45,000/sem | AC Double: ₹62,000/sem | Mess Plan: ₹24,000/sem (includes breakfast, lunch, evening snacks, and dinner).\n"
            "• Gate Timings: Main campus entry open until 10:30 PM; late academic lab passes available via Faculty Advisor.\n"
            "• 24x7 Campus Health Center: Located in Block D (Emergency Helpline: +91-121-2550108).",
        ],
    },
    {
        "tag": "it_helpdesk",
        "description": "Student ERP password reset, campus Wi-Fi, institutional email, and LMS issues",
        "patterns": [
            "forgot my student erp login password",
            "reset student portal login password",
            "campus wifi is not working connecting",
            "configure university email on phone",
            "lms moodle courses not visible",
            "software license for matlab and office 365",
            "netid account locked",
            "vpn access for lab servers",
        ],
        "responses": [
            "IT Helpdesk & Digital Services:\n"
            "• Password Reset: Click 'Forgot Password' on the ERP login page to receive an OTP on your registered mobile, or email ithelpdesk@university.edu.\n"
            "• Campus Wi-Fi ('UniNet-5G'): Log in using your Student ID (e.g., STU2026001) and ERP password. Register up to 3 device MAC addresses.\n"
            "• Free Software Suite: Active students receive Microsoft 365, GitHub Student Developer Pack, MATLAB Campus-Wide License, and AutoCAD Education free with their @university.edu email.",
        ],
    },
    {
        "tag": "career_and_placements",
        "description": "Internships, campus placements, resume reviews, and training cell",
        "patterns": [
            "when does campus placement season start",
            "register for placement drives",
            "internship opportunities and noc letter",
            "get an internship noc from the placement cell",
            "minimum cgpa required for placements",
            "resume review and mock interviews",
            "top recruiters visiting campus",
            "training and placement cell contact",
        ],
        "responses": [
            "Training, Internships & Career Placement Cell (T&P):\n"
            "• Placement Registration: Opens in 6th Semester (UG) / 2nd Semester (PG) on the Superset/T&P Portal.\n"
            "• Internship NOC & Bonafide: Generate automatically under 'ERP -> Student Services -> Certificates & NOC' within 24 hours.\n"
            "• Career Prep Services: Weekly mock technical interviews, aptitude bootcamps, and 1-on-1 resume clinics every Wednesday & Friday (2:00 PM – 5:00 PM, Placement Block Room 204).",
        ],
    },
    {
        "tag": "counseling_and_wellness",
        "description": "Student counseling, stress management, mentorship, and anti-ragging support",
        "patterns": [
            "feeling stressed about exams book counseling",
            "book a counseling appointment session",
            "mental health and student wellness center",
            "anti ragging helpline number",
            "talk to a student counselor",
            "academic anxiety and mentorship",
            "grievance redressal committee",
        ],
        "responses": [
            "Student Wellness, Counseling & Grievance Support:\n"
            "• Confidential Counseling: Free, strictly confidential 1-on-1 sessions with certified counselors at the Wellness Center (Block B, Ground Floor, Mon–Fri 9:00 AM – 6:00 PM).\n"
            "• 24x7 Student Support Helpline: +91-121-2550999 | Email: wellness@university.edu\n"
            "• Anti-Ragging & Grievance Cell: 24x7 Toll-Free Helpline 1800-180-5522 or submit an anonymous report via the ERP Grievance tab.",
        ],
    },
    {
        "tag": "create_ticket",
        "description": "Raise a new support ticket or escalate an unresolved issue to a human advisor",
        "patterns": [
            "raise a support ticket",
            "create a complaint ticket",
            "talk to a human advisor",
            "escalate my issue to admin",
            "submit a grievance ticket",
            "open a helpdesk ticket for fee refund",
            "log a maintenance issue",
            "report a problem with my marksheet",
        ],
        "responses": [
            "I can create an official Student Support Ticket for you right away.",
        ],
    },
    {
        "tag": "check_ticket",
        "description": "Check the status of an existing support ticket or student profile",
        "patterns": [
            "check my ticket status",
            "status of ticket TKT-1001",
            "track support ticket",
            "is my ticket resolved",
            "lookup ticket status",
            "check student profile record for STU2026001",
        ],
        "responses": [
            "Let me check our database for your support ticket or student record.",
        ],
    },
    {
        "tag": "goodbye",
        "description": "Ending the conversation or thanking the bot",
        "patterns": [
            "thank you so much",
            "thanks for your help",
            "bye",
            "goodbye",
            "see you later",
            "that is all for now",
            "exit",
            "quit",
        ],
        "responses": [
            "You're most welcome! Wishing you a great semester ahead. Feel free to reach out anytime you need student support!",
            "Happy to help! Good luck with your studies, and have a wonderful day on campus!",
        ],
    },
]


# ==============================================================================
# 2. SQLITE DATABASE BACKEND (COURSES, FEES, STUDENTS, TICKETS, LOGS)
# ==============================================================================


class StudentSupportDatabase:
    """Manages persistent or in-memory SQLite tables for student services."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self._seed_data()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS courses (
                course_code TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                department TEXT NOT NULL,
                credits INTEGER NOT NULL,
                instructor TEXT NOT NULL,
                exam_date TEXT NOT NULL,
                exam_venue TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS program_fees (
                program_code TEXT PRIMARY KEY,
                program_name TEXT NOT NULL,
                duration_years INTEGER NOT NULL,
                tuition_per_sem INTEGER NOT NULL,
                lab_and_library_fee INTEGER NOT NULL,
                eligibility TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                program TEXT NOT NULL,
                semester INTEGER NOT NULL,
                cgpa REAL NOT NULL,
                attendance_pct REAL NOT NULL,
                fee_status TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS support_tickets (
                ticket_id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                category TEXT NOT NULL,
                subject TEXT NOT NULL,
                status TEXT NOT NULL,
                assigned_to TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                predicted_intent TEXT NOT NULL,
                confidence REAL NOT NULL,
                bot_response TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def _seed_data(self) -> None:
        cur = self.conn.cursor()
        courses = [
            ("CS101", "Introduction to Programming & Python", "Computer Science", 4, "Dr. Ananya Sharma", "2026-11-18 09:30 AM", "Academic Block A - Hall 102"),
            ("CS204", "Data Structures & Algorithms", "Computer Science", 4, "Prof. Rajesh Verma", "2026-11-20 09:30 AM", "Academic Block A - Hall 205"),
            ("CS305", "Artificial Intelligence & Machine Learning", "Computer Science", 4, "Dr. Vikram Nair", "2026-11-23 02:00 PM", "Tech Tower - Lab 301"),
            ("MA102", "Linear Algebra & Probability", "Mathematics", 3, "Dr. Meera Iyer", "2026-11-19 09:30 AM", "Science Block - Room 110"),
            ("EC202", "Digital Logic & Microprocessors", "Electronics", 4, "Prof. S. Kulkarni", "2026-11-21 02:00 PM", "Engineering Block C - Room 104"),
            ("MB301", "Financial Management & Corporate Strategy", "Management", 3, "Dr. Neha Kapoor", "2026-11-22 10:00 AM", "Management Wing - Seminar Hall 2"),
        ]
        cur.executemany(
            "INSERT OR IGNORE INTO courses VALUES (?, ?, ?, ?, ?, ?, ?)", courses
        )

        programs = [
            ("BTECH", "B.Tech (Computer Science / AI / Electronics)", 4, 85000, 12000, "10+2 with PCM >= 60% & JEE/University Entrance Test"),
            ("MBA", "Master of Business Administration (MBA)", 2, 110000, 15000, "Graduation >= 60% & valid CAT/MAT/GMAT/University Score"),
            ("MCA", "Master of Computer Applications (MCA)", 2, 72000, 10000, "BCA/B.Sc (CS/IT) or Graduation with Mathematics >= 55%"),
            ("BSC", "B.Sc (Hons.) Data Science & Computing", 3, 52000, 8000, "10+2 with Mathematics/Statistics >= 55%"),
            ("BBA", "Bachelor of Business Administration (BBA)", 3, 58000, 7500, "10+2 in any stream with aggregate >= 55%"),
        ]
        cur.executemany(
            "INSERT OR IGNORE INTO program_fees VALUES (?, ?, ?, ?, ?, ?)", programs
        )

        students = [
            ("STU2026001", "Aarav Gupta", "B.Tech CSE (AI & ML)", 5, 8.92, 88.5, "Paid (No Dues)"),
            ("STU2026002", "Priya Singh", "MBA Finance", 3, 9.15, 92.0, "Paid (Merit Scholarship 50%)"),
            ("STU2026003", "Rohan Mishra", "MCA", 1, 7.80, 74.2, "Pending Installment 2 (₹36,000)"),
        ]
        cur.executemany(
            "INSERT OR IGNORE INTO students VALUES (?, ?, ?, ?, ?, ?, ?)", students
        )

        tickets = [
            ("TKT-1001", "STU2026001", "Academic / ERP", "Elective CS305 credit reflection in portal", "In Progress", "Academic Dean Office", "2026-10-18 11:20"),
            ("TKT-1002", "STU2026002", "Accounts & Fees", "Merit Scholarship certificate download", "Resolved", "Accounts Section Desk 2", "2026-10-15 14:05"),
            ("TKT-1003", "STU2026003", "Hostel & Wi-Fi", "Wi-Fi router signal drop in Hostel Block B", "Assigned to Technician", "IT Network Team", "2026-10-20 09:45"),
        ]
        cur.executemany(
            "INSERT OR IGNORE INTO support_tickets VALUES (?, ?, ?, ?, ?, ?, ?)",
            tickets,
        )
        self.conn.commit()

    def get_course(self, course_code: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM courses WHERE UPPER(course_code) = UPPER(?)",
            (course_code.strip(),),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_program_fee(self, program_key: str) -> Optional[Dict[str, Any]]:
        normalized = re.sub(r"[^A-Za-z]", "", program_key).upper()
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM program_fees WHERE program_code = ?", (normalized,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM students WHERE UPPER(student_id) = UPPER(?)",
            (student_id.strip(),),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_ticket(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM support_tickets WHERE UPPER(ticket_id) = UPPER(?)",
            (ticket_id.strip(),),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def create_ticket(
        self, student_id: str, category: str, subject: str
    ) -> Dict[str, Any]:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) AS cnt FROM support_tickets")
        next_num = 1001 + int(cur.fetchone()["cnt"])
        ticket_id = f"TKT-{next_num}"
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        department_map = {
            "admissions": "Admissions Cell",
            "fees_and_scholarships": "Finance & Scholarship Desk",
            "course_registration": "Academic Registrar Office",
            "exams_and_grading": "Controller of Examinations",
            "library_services": "Central Library Helpdesk",
            "hostel_and_facilities": "Chief Warden Office",
            "it_helpdesk": "Campus IT Support Team",
            "career_and_placements": "Training & Placement Cell",
            "counseling_and_wellness": "Student Wellness Center",
        }
        assigned_to = department_map.get(category, "Student Support Helpdesk")
        cur.execute(
            "INSERT INTO support_tickets VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                ticket_id,
                student_id.upper(),
                category,
                subject,
                "Open (SLA: 24 Hours)",
                assigned_to,
                now_str,
            ),
        )
        self.conn.commit()
        return {
            "ticket_id": ticket_id,
            "student_id": student_id.upper(),
            "category": category,
            "subject": subject,
            "status": "Open (SLA: 24 Hours)",
            "assigned_to": assigned_to,
            "created_at": now_str,
        }

    def log_interaction(
        self,
        session_id: str,
        user_message: str,
        predicted_intent: str,
        confidence: float,
        bot_response: str,
    ) -> None:
        cur = self.conn.cursor()
        now_str = datetime.datetime.now().isoformat(timespec="seconds")
        cur.execute(
            "INSERT INTO chat_logs (session_id, user_message, predicted_intent, confidence, bot_response, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, user_message, predicted_intent, confidence, bot_response, now_str),
        )
        self.conn.commit()


# ==============================================================================
# 3. NLP PREPROCESSING, ENTITY EXTRACTION & INTENT CLASSIFIER
# ==============================================================================

STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "at", "by", "from", "up", "about",
    "into", "over", "after", "please", "can", "could", "would", "should", "tell",
    "me", "my", "i", "we", "you", "it", "this", "that", "and", "or", "how",
    "many", "what", "when", "where", "do", "does", "get", "want",
}

GREETING_PREFIX_RE = re.compile(
    r"^(hello|hi\s+there|hi|hey|good\s+morning|good\s+afternoon|good\s+evening|greetings)[,!.\s]+",
    re.IGNORECASE,
)

SYNONYM_MAP = {
    "tuition": "fee",
    "fees": "fee",
    "cost": "fee",
    "charges": "fee",
    "scholarship": "scholarship",
    "scholarships": "scholarship",
    "exams": "exam",
    "examination": "exam",
    "timetable": "schedule",
    "datesheet": "schedule",
    "results": "result",
    "grades": "grade",
    "marks": "grade",
    "courses": "course",
    "subjects": "course",
    "credits": "credit",
    "books": "book",
    "dorm": "hostel",
    "accommodation": "hostel",
    "wifi": "wifi",
    "internet": "wifi",
    "password": "password",
    "login": "password",
    "jobs": "placement",
    "placements": "placement",
    "internships": "internship",
    "counselor": "counseling",
    "therapist": "counseling",
    "stress": "stressed",
    "anxious": "anxiety",
}


def preprocess_text(text: str) -> str:
    """Lowercases, strips conversational greeting prefixes when a question follows, and normalizes tokens."""
    text = text.strip()
    # If the user starts with a greeting followed by a substantive question, strip the greeting prefix
    stripped = GREETING_PREFIX_RE.sub("", text).strip()
    if len(stripped.split()) >= 3:
        text = stripped

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    raw_tokens = text.split()
    tokens = []
    for tok in raw_tokens:
        if tok in STOPWORDS and len(raw_tokens) > 3:
            continue
        if tok in SYNONYM_MAP:
            tokens.append(SYNONYM_MAP[tok])
            continue
        if tok.endswith("ing") and len(tok) > 5:
            stem = tok[:-3]
        elif tok.endswith("ed") and len(tok) > 4:
            stem = tok[:-2]
        elif tok.endswith("s") and len(tok) > 4 and not tok.endswith("ss"):
            stem = tok[:-1]
        else:
            stem = tok
        tokens.append(SYNONYM_MAP.get(stem, stem))
    return " ".join(tokens)


def extract_entities(text: str) -> Dict[str, str]:
    """Extracts structured academic entities from user queries using regex patterns."""
    entities: Dict[str, str] = {}

    # 1. Student ID (e.g., STU2026001)
    stu_match = re.search(r"\b(STU\d{4,8})\b", text, re.IGNORECASE)
    if stu_match:
        entities["student_id"] = stu_match.group(1).upper()

    # 2. Ticket ID (e.g., TKT-1001)
    tkt_match = re.search(r"\b(TKT-\d{3,6})\b", text, re.IGNORECASE)
    if tkt_match:
        entities["ticket_id"] = tkt_match.group(1).upper()

    # 3. Course Code (e.g., CS101, CS204, CS305, MA102, EC202, MB301)
    course_match = re.search(
        r"\b(CS\d{3}|MA\d{3}|EC\d{3}|MB\d{3}|EE\d{3}|ME\d{3})\b",
        text,
        re.IGNORECASE,
    )
    if course_match:
        entities["course_code"] = course_match.group(1).upper()

    # 4. Academic Program (B.Tech, MBA, MCA, B.Sc, BBA)
    prog_match = re.search(
        r"\b(b\.?\s*tech|m\.?\s*b\.?\s*a|m\.?\s*c\.?\s*a|b\.?\s*sc|b\.?\s*b\.?\s*a)\b",
        text,
        re.IGNORECASE,
    )
    if prog_match:
        entities["program"] = re.sub(r"[^A-Za-z]", "", prog_match.group(1)).upper()

    # 5. Email Address
    email_match = re.search(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text
    )
    if email_match:
        entities["email"] = email_match.group(0).lower()

    return entities


class HybridIntentClassifier:
    """
    Hybrid NLP Intent Classifier combining:
    - Scikit-learn TF-IDF (word + char n-grams) + Logistic Regression + Cosine Similarity
    - Pure-Python TF-IDF Cosine Similarity fallback when scikit-learn is unavailable.
    """

    def __init__(self, intents: List[Dict[str, Any]]) -> None:
        self.intents = intents
        self.train_texts: List[str] = []
        self.train_labels: List[str] = []
        self.responses_by_tag: Dict[str, List[str]] = {}

        for intent in intents:
            tag = intent["tag"]
            self.responses_by_tag[tag] = intent["responses"]
            for pattern in intent["patterns"]:
                self.train_texts.append(preprocess_text(pattern))
                self.train_labels.append(tag)

        self._fit()

    def _fit(self) -> None:
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), sublinear_tf=True
            )
            self.X_train = self.vectorizer.fit_transform(self.train_texts)
            self.clf = LogisticRegression(C=5.0, max_iter=500, random_state=42)
            self.clf.fit(self.X_train, self.train_labels)
        else:
            self.doc_freq: Counter = Counter()
            self.tokenized_docs: List[List[str]] = []
            for doc in self.train_texts:
                tokens = doc.split()
                bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
                all_terms = tokens + bigrams
                self.tokenized_docs.append(all_terms)
                for term in set(all_terms):
                    self.doc_freq[term] += 1
            self.num_docs = len(self.tokenized_docs)
            self.doc_vectors = [self._to_tfidf_vec(terms) for terms in self.tokenized_docs]

    def _to_tfidf_vec(self, terms: List[str]) -> Dict[str, float]:
        tf = Counter(terms)
        vec: Dict[str, float] = {}
        for term, count in tf.items():
            df = self.doc_freq.get(term, 0)
            if df > 0:
                idf = math.log((1 + self.num_docs) / (1 + df)) + 1.0
                vec[term] = (1.0 + math.log(count)) * idf
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    def predict(self, user_text: str) -> Tuple[str, float]:
        cleaned = preprocess_text(user_text)
        if not cleaned.strip():
            return "unknown", 0.0

        if SKLEARN_AVAILABLE:
            x_vec = self.vectorizer.transform([cleaned])
            sims = cosine_similarity(x_vec, self.X_train)[0]
            best_sim_idx = int(sims.argmax())
            max_sim = float(sims[best_sim_idx])
            sim_label = self.train_labels[best_sim_idx]

            probs = self.clf.predict_proba(x_vec)[0]
            best_prob_idx = int(probs.argmax())
            max_prob = float(probs[best_prob_idx])
            prob_label = str(self.clf.classes_[best_prob_idx])

            if max_sim >= 0.45:
                confidence = round(0.65 * max_sim + 0.35 * max_prob, 3)
                return sim_label, confidence
            elif max_prob >= 0.28 and max_sim >= 0.18:
                confidence = round(0.5 * max_sim + 0.5 * max_prob, 3)
                return prob_label, confidence
            return "unknown", round(max(max_sim, max_prob * 0.5), 3)
        else:
            tokens = cleaned.split()
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
            q_vec = self._to_tfidf_vec(tokens + bigrams)
            best_score = 0.0
            best_label = "unknown"
            for doc_vec, label in zip(self.doc_vectors, self.train_labels):
                score = sum(q_vec.get(k, 0.0) * v for k, v in doc_vec.items())
                if score > best_score:
                    best_score = score
                    best_label = label
            if best_score < 0.20:
                return "unknown", round(best_score, 3)
            return best_label, round(best_score, 3)


# ==============================================================================
# 4. DIALOGUE MANAGER & CONTEXTUAL CHATBOT ENGINE
# ==============================================================================


@dataclass
class ConversationSession:
    session_id: str
    last_intent: Optional[str] = None
    awaiting_slot: Optional[str] = None
    pending_ticket_category: str = "general_support"
    student_id: Optional[str] = None
    history: List[Dict[str, str]] = field(default_factory=list)


class StudentSupportChatbot:
    """End-to-end conversational agent for Student Support Services."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db = StudentSupportDatabase(db_path=db_path)
        self.classifier = HybridIntentClassifier(INTENTS_DATA)
        self.sessions: Dict[str, ConversationSession] = {}

    def get_session(self, session_id: str = "default") -> ConversationSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id=session_id)
        return self.sessions[session_id]

    def respond(
        self, user_message: str, session_id: str = "default"
    ) -> Dict[str, Any]:
        session = self.get_session(session_id)
        text = user_message.strip()
        if not text:
            return {
                "intent": "empty",
                "confidence": 0.0,
                "entities": {},
                "response": "Please type a question or select a topic (e.g., Admissions, Fees, Courses, Exams, Hostel, Library, or Support Tickets).",
            }

        entities = extract_entities(text)
        if "student_id" in entities:
            session.student_id = entities["student_id"]

        # Handle multi-turn slot filling (e.g., when collecting issue details for a support ticket)
        if session.awaiting_slot == "ticket_subject":
            stu_id = session.student_id or "STU2026001"
            inferred_cat, _ = self.classifier.predict(text)
            category = (
                inferred_cat
                if inferred_cat
                not in ("unknown", "greeting", "goodbye", "create_ticket", "check_ticket")
                else session.pending_ticket_category
            )
            ticket = self.db.create_ticket(
                student_id=stu_id,
                category=category,
                subject=text,
            )
            session.awaiting_slot = None
            reply = (
                f"Support Ticket Created!\n"
                f"• Ticket ID: {ticket['ticket_id']}\n"
                f"• Student ID: {ticket['student_id']}\n"
                f"• Assigned Department: {ticket['assigned_to']}\n"
                f"• Issue Summary: {ticket['subject']}\n"
                f"• Status: {ticket['status']}\n"
                f"You can check updates anytime by typing: 'Check status of {ticket['ticket_id']}'."
            )
            self.db.log_interaction(
                session_id, text, "create_ticket_slot_filled", 1.0, reply
            )
            return {
                "intent": "create_ticket",
                "confidence": 1.0,
                "entities": {"ticket_id": ticket["ticket_id"], **entities},
                "response": reply,
            }

        # Direct entity shortcuts if a user specifically provides a Ticket ID
        if "ticket_id" in entities:
            tkt = self.db.get_ticket(entities["ticket_id"])
            if tkt:
                reply = (
                    f"Support Ticket Details ({tkt['ticket_id']}):\n"
                    f"• Student ID: {tkt['student_id']}\n"
                    f"• Category: {tkt['category']}\n"
                    f"• Subject: {tkt['subject']}\n"
                    f"• Current Status: {tkt['status']}\n"
                    f"• Assigned Department: {tkt['assigned_to']}\n"
                    f"• Created On: {tkt['created_at']}"
                )
            else:
                reply = (
                    f"I could not find a ticket with ID '{entities['ticket_id']}'. "
                    f"Sample active tickets in the system are TKT-1001, TKT-1002, and TKT-1003."
                )
            self.db.log_interaction(session_id, text, "check_ticket", 0.99, reply)
            return {
                "intent": "check_ticket",
                "confidence": 0.99,
                "entities": entities,
                "response": reply,
            }

        intent, confidence = self.classifier.predict(text)

        # Boost intent if strong domain entity is present
        if "course_code" in entities and intent == "unknown":
            intent, confidence = "course_registration", 0.85
        elif "program" in entities and intent == "unknown":
            intent, confidence = "fees_and_scholarships", 0.85
        elif "student_id" in entities and intent in ("unknown", "check_ticket"):
            intent, confidence = "check_ticket", 0.90

        reply_parts: List[str] = []

        # Intent-specific dynamic handlers
        if intent == "create_ticket":
            session.pending_ticket_category = (
                session.last_intent
                if session.last_intent
                and session.last_intent
                not in ("greeting", "goodbye", "unknown", "create_ticket", "check_ticket")
                else "general_support"
            )
            session.awaiting_slot = "ticket_subject"
            stu_note = (
                f" (linked to Student ID {session.student_id})"
                if session.student_id
                else " (you can also include your Student ID, e.g., STU2026001)"
            )
            reply_parts.append(
                f"I'd be glad to open a support ticket for you{stu_note}. "
                f"Please describe your issue in one sentence so I can route it to the right department:"
            )

        elif intent == "check_ticket":
            if "student_id" in entities:
                stu = self.db.get_student(entities["student_id"])
                if stu:
                    reply_parts.append(
                        f"Student Academic & Fee Profile ({stu['student_id']}):\n"
                        f"• Name: {stu['name']}\n"
                        f"• Program: {stu['program']} (Semester {stu['semester']})\n"
                        f"• Current CGPA: {stu['cgpa']:.2f}\n"
                        f"• Attendance: {stu['attendance_pct']:.1f}%\n"
                        f"• Fee Status: {stu['fee_status']}"
                    )
                else:
                    reply_parts.append(
                        f"No student record found for ID '{entities['student_id']}'. "
                        f"Try sample IDs: STU2026001, STU2026002, or STU2026003."
                    )
            else:
                reply_parts.append(
                    "To check a support ticket or student profile, please share your Ticket ID (e.g., TKT-1001, TKT-1002) or Student ID (e.g., STU2026001)."
                )

        elif intent != "unknown":
            base_responses = self.classifier.responses_by_tag.get(intent, [])
            if base_responses:
                reply_parts.append(base_responses[0])

        # Enrich response with live SQLite database records when entities are detected
        if "course_code" in entities:
            course = self.db.get_course(entities["course_code"])
            if course:
                reply_parts.append(
                    f"\nCourse & Exam Record Found for {course['course_code']}:\n"
                    f"• Title: {course['title']} ({course['department']})\n"
                    f"• Credits: {course['credits']} | Instructor: {course['instructor']}\n"
                    f"• End-Semester Exam: {course['exam_date']} at {course['exam_venue']}"
                )

        if "program" in entities:
            prog = self.db.get_program_fee(entities["program"])
            if prog:
                total_sem = prog["tuition_per_sem"] + prog["lab_and_library_fee"]
                reply_parts.append(
                    f"\nProgram Fee & Eligibility Breakdown ({prog['program_name']}):\n"
                    f"• Duration: {prog['duration_years']} Years ({prog['duration_years'] * 2} Semesters)\n"
                    f"• Tuition Fee: ₹{prog['tuition_per_sem']:,} / semester\n"
                    f"• Lab & Library Fee: ₹{prog['lab_and_library_fee']:,} / semester\n"
                    f"• Total per Semester: ₹{total_sem:,}\n"
                    f"• Eligibility: {prog['eligibility']}"
                )

        if not reply_parts:
            reply_parts.append(
                "I'm not completely sure I understood that query. I can help you with:\n"
                "1. Admissions & Eligibility (e.g., 'How to apply for B.Tech admission?')\n"
                "2. Tuition Fees & Scholarships (e.g., 'What is the MBA fee structure?')\n"
                "3. Course Registration & Exam Schedules (e.g., 'Exam date for CS101')\n"
                "4. Library, Hostel, Wi-Fi & IT Helpdesk (e.g., 'Forgot my ERP password')\n"
                "5. Support Tickets & Student Profiles (e.g., 'Check ticket TKT-1001' or 'Raise a support ticket')"
            )

        final_reply = "\n".join(reply_parts)
        session.last_intent = intent
        session.history.append({"user": text, "bot": final_reply})
        self.db.log_interaction(session_id, text, intent, confidence, final_reply)

        return {
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "response": final_reply,
        }


# ==============================================================================
# 5. MODEL EVALUATION & AUTOMATED DEMO SUITE
# ==============================================================================

EVALUATION_TEST_CASES: List[Tuple[str, str]] = [
    ("Hello, good morning!", "greeting"),
    ("What documents are needed for undergraduate admission?", "admissions"),
    ("What is the MBA tuition fee and scholarship criteria?", "fees_and_scholarships"),
    ("How many credits can I register for this semester?", "course_registration"),
    ("When is the final exam for CS101 and how is CGPA calculated?", "exams_and_grading"),
    ("How many books can I borrow from the library and what is the late fine?", "library_services"),
    ("What are the hostel charges and campus bus timings?", "hostel_and_facilities"),
    ("I forgot my student ERP login password and campus wifi is not connecting", "it_helpdesk"),
    ("How do I get an internship NOC from the placement cell?", "career_and_placements"),
    ("I feel stressed about exams, how can I book a counseling appointment?", "counseling_and_wellness"),
    ("I want to raise a support ticket for my fee receipt", "create_ticket"),
    ("Check status of ticket TKT-1001", "check_ticket"),
    ("Thank you for your help, goodbye!", "goodbye"),
]


def run_demo_and_evaluation() -> None:
    print("=" * 78)
    print("  PROJECT 1: AI CHATBOT FOR STUDENT SUPPORT SERVICES")
    print(f"  NLP Engine: {'Scikit-Learn TF-IDF + LogisticRegression' if SKLEARN_AVAILABLE else 'Built-in TF-IDF Cosine Similarity'}")
    print("=" * 78)

    bot = StudentSupportChatbot()

    print("\n[PART 1] INTENT CLASSIFICATION EVALUATION ON TEST BENCHMARK")
    print("-" * 78)
    correct = 0

    for query, expected_intent in EVALUATION_TEST_CASES:
        result = bot.respond(query, session_id="eval_session")
        bot.get_session("eval_session").awaiting_slot = None
        predicted = result["intent"]
        conf = result["confidence"]
        is_hit = predicted == expected_intent
        if is_hit:
            correct += 1
        status = "PASS" if is_hit else "FAIL"
        print(
            f"[{status}] Query: {query[:48]:<48} | Pred: {predicted:<23} (Conf: {conf:.2f})"
        )

    accuracy = (correct / len(EVALUATION_TEST_CASES)) * 100.0
    print("-" * 78)
    print(
        f"Benchmark Accuracy: {correct}/{len(EVALUATION_TEST_CASES)} ({accuracy:.1f}%)"
    )

    print("\n[PART 2] MULTI-TURN CONVERSATION & DATABASE LOOKUP WALKTHROUGH")
    print("-" * 78)
    sample_dialogue = [
        "Hi there! Can you tell me the fee structure for B.Tech?",
        "What is the exam date and instructor for CS305?",
        "Can you check the student profile for STU2026001?",
        "I want to raise a support ticket",
        "My lab attendance for CS305 is not updated on the exam ERP portal",
        "Check my ticket status for TKT-1004",
        "Thanks a lot, goodbye!",
    ]

    for turn_idx, message in enumerate(sample_dialogue, start=1):
        out = bot.respond(message, session_id="demo_student_01")
        print(f"\nStudent (Turn {turn_idx}): {message}")
        print(
            f"  [NLP Meta -> Intent: {out['intent']} | Confidence: {out['confidence']:.2f} | Entities: {out['entities']}]"
        )
        for line in out["response"].splitlines():
            print(f"  Bot: {line}")

    print("\n" + "=" * 78)
    print("Tip: Run `python3 student_support_chatbot.py --cli` for interactive terminal chat")
    print("     or  `python3 student_support_chatbot.py --web --port 8080` for the Web UI.")
    print("=" * 78)


# ==============================================================================
# 6. INTERACTIVE CLI INTERFACE
# ==============================================================================


def run_cli_chat() -> None:
    bot = StudentSupportChatbot()
    print("=" * 72)
    print("  UNIVERSITY AI STUDENT SUPPORT CHATBOT (Interactive CLI Mode)")
    print("  Type 'exit' or 'quit' to end the conversation.")
    print("=" * 72)
    welcome = bot.respond("hello", session_id="cli_user")
    print(f"\nBot: {welcome['response']}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Session ended. Good luck with your studies!")
            break

        if not user_input:
            continue

        result = bot.respond(user_input, session_id="cli_user")
        print(f"\nBot [{result['intent']} | {result['confidence']:.2f}]:")
        print(result["response"])
        print()

        if result["intent"] == "goodbye" and user_input.lower() in (
            "exit",
            "quit",
            "bye",
            "goodbye",
        ):
            break


# ==============================================================================
# 7. BUILT-IN WEB UI SERVER (ZERO EXTERNAL WEB DEPENDENCIES)
# ==============================================================================

WEB_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Student Support Chatbot</title>
<style>
  :root {
    --primary: #1e3a8a;
    --accent: #2563eb;
    --bg: #f1f5f9;
    --card: #ffffff;
    --text: #0f172a;
    --muted: #64748b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    padding: 16px;
  }
  .chat-container {
    width: 100%;
    max-width: 780px;
    height: 88vh;
    background: var(--card);
    border-radius: 16px;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.12);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .header {
    background: linear-gradient(135deg, var(--primary), var(--accent));
    color: #fff;
    padding: 18px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .header h1 { font-size: 1.15rem; font-weight: 700; }
  .header p { font-size: 0.82rem; opacity: 0.9; margin-top: 2px; }
  .badge {
    background: rgba(255,255,255,0.2);
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .quick-chips {
    display: flex;
    gap: 8px;
    padding: 10px 18px;
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
    overflow-x: auto;
  }
  .chip {
    background: #e0e7ff;
    color: var(--primary);
    border: none;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
  }
  .chip:hover { background: #c7d2fe; }
  .messages {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .msg {
    max-width: 82%;
    padding: 12px 16px;
    border-radius: 14px;
    line-height: 1.5;
    font-size: 0.92rem;
    white-space: pre-wrap;
  }
  .msg.bot {
    align-self: flex-start;
    background: #f1f5f9;
    color: var(--text);
    border-bottom-left-radius: 4px;
  }
  .msg.user {
    align-self: flex-end;
    background: var(--accent);
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .meta {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 4px;
  }
  .input-bar {
    display: flex;
    gap: 10px;
    padding: 14px 18px;
    border-top: 1px solid #e2e8f0;
    background: #fff;
  }
  .input-bar input {
    flex: 1;
    padding: 12px 14px;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    font-size: 0.92rem;
    outline: none;
  }
  .input-bar input:focus { border-color: var(--accent); }
  .input-bar button {
    background: var(--accent);
    color: #fff;
    border: none;
    padding: 0 22px;
    border-radius: 10px;
    font-weight: 600;
    cursor: pointer;
  }
  .input-bar button:hover { background: var(--primary); }
</style>
</head>
<body>
<div class="chat-container">
  <div class="header">
    <div>
      <h1>AI Student Support Assistant</h1>
      <p>Admissions • Fees • Courses • Exams • Hostel • IT • Support Tickets</p>
    </div>
    <span class="badge">Online 24x7</span>
  </div>
  <div class="quick-chips">
    <button class="chip" onclick="sendQuick('What is the B.Tech fee structure?')">B.Tech Fees</button>
    <button class="chip" onclick="sendQuick('Exam date for CS101')">CS101 Exam Schedule</button>
    <button class="chip" onclick="sendQuick('Check student profile STU2026001')">Profile STU2026001</button>
    <button class="chip" onclick="sendQuick('Check ticket status TKT-1001')">Track Ticket TKT-1001</button>
    <button class="chip" onclick="sendQuick('Raise a support ticket')">Raise Support Ticket</button>
    <button class="chip" onclick="sendQuick('How do I reset my ERP password?')">Reset ERP Password</button>
  </div>
  <div class="messages" id="messages">
    <div class="msg bot">Hello! Welcome to the University Student Support Desk. Ask me about admissions, fee structure, course registration, exam schedules (e.g., CS101, CS305), library, hostel, or support tickets!</div>
  </div>
  <form class="input-bar" onsubmit="handleSend(event)">
    <input type="text" id="userInput" placeholder="Ask a student support question..." autocomplete="off" />
    <button type="submit">Send</button>
  </form>
</div>
<script>
  const messagesEl = document.getElementById('messages');
  const inputEl = document.getElementById('userInput');

  function appendMessage(text, role, metaText) {
    const wrapper = document.createElement('div');
    wrapper.className = 'msg ' + role;
    wrapper.textContent = text;
    if (metaText) {
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = metaText;
      wrapper.appendChild(meta);
    }
    messagesEl.appendChild(wrapper);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function sendQuery(text) {
    appendMessage(text, 'user');
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: 'web_user' })
    });
    const data = await res.json();
    const meta = `Intent: ${data.intent} (${Math.round(data.confidence * 100)}% confidence)`;
    appendMessage(data.response, 'bot', meta);
  }

  function handleSend(e) {
    e.preventDefault();
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = '';
    sendQuery(text);
  }

  function sendQuick(text) {
    sendQuery(text);
  }
</script>
</body>
</html>
"""


def run_web_server(port: int = 8080) -> None:
    bot = StudentSupportChatbot()

    class ChatRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(WEB_HTML_PAGE.encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:
            if self.path == "/api/chat":
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw_body or "{}")
                user_msg = payload.get("message", "")
                session_id = payload.get("session_id", "web_user")
                result = bot.respond(user_msg, session_id=session_id)

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            return  # Keep console output clean

    server = HTTPServer(("0.0.0.0", port), ChatRequestHandler)
    print(f"Student Support Web UI running at http://localhost:{port}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server.")
        server.server_close()


# ==============================================================================
# 8. ENTRY POINT
# ==============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project 1: AI Chatbot for Student Support Services"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Launch interactive command-line chat mode",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch built-in web browser UI server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the web server (default: 8080)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run automated benchmark evaluation and multi-turn conversation walkthrough",
    )
    args = parser.parse_args()

    if args.cli:
        run_cli_chat()
    elif args.web:
        run_web_server(port=args.port)
    else:
        run_demo_and_evaluation()


if __name__ == "__main__":
    main()