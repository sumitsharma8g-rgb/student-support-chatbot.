#!/usr/bin/env python3
"""Unit tests for Project 1: AI Chatbot for Student Support Services."""

import unittest
from student_support_chatbot import (
    EVALUATION_TEST_CASES,
    StudentSupportChatbot,
    extract_entities,
    preprocess_text,
)


class TestStudentSupportChatbot(unittest.TestCase):
    def setUp(self) -> None:
        self.bot = StudentSupportChatbot(db_path=":memory:")

    def test_benchmark_intent_accuracy(self) -> None:
        """Verify all 13 benchmark test cases classify to the expected intent."""
        for query, expected_intent in EVALUATION_TEST_CASES:
            res = self.bot.respond(query, session_id="unit_test")
            self.bot.get_session("unit_test").awaiting_slot = None
            self.assertEqual(
                res["intent"],
                expected_intent,
                f"Failed on query: {query}",
            )

    def test_entity_extraction(self) -> None:
        """Verify extraction of Student ID, Course Code, Program, Ticket ID, and Email."""
        sample = "Hi, I am STU2026001 in B.Tech checking CS305 and ticket TKT-1001 (user@university.edu)"
        entities = extract_entities(sample)
        self.assertEqual(entities.get("student_id"), "STU2026001")
        self.assertEqual(entities.get("program"), "BTECH")
        self.assertEqual(entities.get("course_code"), "CS305")
        self.assertEqual(entities.get("ticket_id"), "TKT-1001")
        self.assertEqual(entities.get("email"), "user@university.edu")

    def test_multi_turn_ticket_creation(self) -> None:
        """Verify multi-turn support ticket creation and status lookup."""
        r1 = self.bot.respond("Raise a support ticket for STU2026002", session_id="tkt_session")
        self.assertEqual(r1["intent"], "create_ticket")
        self.assertEqual(self.bot.get_session("tkt_session").awaiting_slot, "ticket_subject")

        r2 = self.bot.respond("My campus wifi is not working in Hostel Block B", session_id="tkt_session")
        self.assertIn("TKT-1004", r2["response"])
        self.assertIn("Campus IT Support Team", r2["response"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
