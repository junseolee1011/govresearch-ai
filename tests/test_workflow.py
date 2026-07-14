"""Workflow integration tests."""

from __future__ import annotations

import unittest

from app.graph import build_research_graph


class ResearchWorkflowTest(unittest.TestCase):
    """Test the complete Sprint 1 workflow."""

    def test_graph_generates_report(self) -> None:
        """The graph should execute all agents and return a Markdown report."""
        result = build_research_graph().invoke(
            {"topic": "AI services used by public institutions"}
        )

        self.assertEqual(len(result["plan"]), 4)
        self.assertEqual(len(result["sources"]), 3)
        self.assertIn("# GovResearch-AI Research Report", result["report"])
        self.assertIn("AI Service Inventory and Classification", result["report"])


if __name__ == "__main__":
    unittest.main()
