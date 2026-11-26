"""Offline unit tests for the GovResearch-AI pipeline.

These tests need neither Ollama nor a Tavily API key. They cover the
deterministic parts of the workflow: chunking, tool selection, query
generation, MCP-style tool dispatch in the researcher, and configuration.

Run from the project root:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import ast
import collections
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.planner import extract_search_queries_from_plan
from agents.researcher import (
    chunk_text,
    conduct_research,
    should_use_filesystem,
    should_use_tavily,
)
from config.service_constants import SERVICE_SEARCH_QUERIES
from config.settings import get_settings
from models.schemas import ResearchPlan, SubTask


def _make_plan() -> ResearchPlan:
    """Build a minimal valid research plan."""
    tasks = [
        SubTask(description="Search government case studies", priority="high", estimated_complexity="simple"),
        SubTask(description="Classify services by agency", priority="medium", estimated_complexity="moderate"),
        SubTask(description="Write the final report", priority="low", estimated_complexity="simple"),
    ]
    return ResearchPlan(main_goal="Map AI services in public agencies", sub_tasks=tasks, total_estimated_steps=3)


class FakeMCPClient:
    """Minimal stand-in for MCPClient that returns canned tool results."""

    def call_tool(self, server_name: str, tool_name: str, **kwargs):
        if server_name == "filesystem" and tool_name == "list_files":
            return {"success": True, "files": [{"name": "case.md"}]}
        if server_name == "filesystem" and tool_name == "read_markdown":
            return {"success": True, "content": "가나다 " * 600, "path": "data/case.md", "type": "markdown"}
        if server_name == "tavily" and tool_name == "search":
            return {"success": True, "results": [{"url": "https://a.kr"}, {"url": "https://a.kr"}, {"url": "https://b.kr"}]}
        raise ValueError(f"unexpected call: {server_name}.{tool_name}")


class ChunkTextTest(unittest.TestCase):
    """chunk_text splits text into bounded, overlapping chunks."""

    def test_empty_text_returns_no_chunks(self) -> None:
        self.assertEqual(chunk_text(""), [])

    def test_short_text_is_one_chunk(self) -> None:
        self.assertEqual(chunk_text("short text"), ["short text"])

    def test_long_text_is_split_within_size_limit(self) -> None:
        chunks = chunk_text("가나다 " * 1000, chunk_size=1500, overlap=200)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 1500 for chunk in chunks))


class ToolSelectionTest(unittest.TestCase):
    """Rule-based tool selection in the researcher."""

    def test_filesystem_used_for_case_style_topics(self) -> None:
        self.assertTrue(should_use_filesystem("공공기관 AI 서비스 현황"))
        self.assertTrue(should_use_filesystem("public sector AI case study"))

    def test_filesystem_skipped_for_unrelated_topics(self) -> None:
        self.assertFalse(should_use_filesystem("quantum computing hardware"))

    def test_web_search_always_enabled(self) -> None:
        self.assertTrue(should_use_tavily("anything at all"))


class ResearcherTest(unittest.TestCase):
    """conduct_research with a fake MCP client."""

    def test_collects_chunks_and_deduplicates_urls(self) -> None:
        state = {"topic": "공공기관 AI 사례 현황", "search_queries": ["query"]}
        result = conduct_research(state, None, mcp_client=FakeMCPClient())
        self.assertEqual([r["url"] for r in result["search_results"]], ["https://a.kr", "https://b.kr"])
        self.assertEqual(len(result["local_documents"]), 1)
        self.assertGreater(len(result["document_chunks"]), 1)
        self.assertEqual(result["document_chunks"][0]["chunk_id"], 1)


class QueryGenerationTest(unittest.TestCase):
    """extract_search_queries_from_plan for both topic branches."""

    def test_korean_public_topic_starts_with_main_goal(self) -> None:
        plan = _make_plan()
        queries = extract_search_queries_from_plan(plan, "한국 공공기관 AI 서비스")
        self.assertEqual(queries[0], plan.main_goal)
        self.assertEqual(queries[1:], SERVICE_SEARCH_QUERIES)

    def test_query_cap_is_respected(self) -> None:
        queries = extract_search_queries_from_plan(_make_plan(), "한국 공공기관 AI 서비스", max_queries=5)
        self.assertEqual(len(queries), 5)

    def test_default_cap_covers_all_predefined_queries(self) -> None:
        queries = extract_search_queries_from_plan(_make_plan(), "한국 공공기관 AI 서비스")
        self.assertEqual(len(queries), 1 + len(SERVICE_SEARCH_QUERIES))

    def test_other_topics_use_plan_tasks(self) -> None:
        queries = extract_search_queries_from_plan(_make_plan(), "climate adaptation")
        self.assertIn("Map AI services in public agencies", queries)
        self.assertIn("Search government case studies", queries)
        self.assertNotIn(SERVICE_SEARCH_QUERIES[0], queries)


class ConfigurationTest(unittest.TestCase):
    """Settings defaults and constants hygiene."""

    def test_default_query_cap(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAX_SEARCH_QUERIES", None)
            self.assertEqual(get_settings().max_search_queries, 70)

    def test_target_cases_has_no_duplicate_keys(self) -> None:
        """A duplicate key in the dict literal silently overrides the earlier entry."""
        source = (Path(__file__).resolve().parent.parent / "config" / "service_constants.py").read_text(encoding="utf-8")
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "TARGET_CASES":
                keys = [key.value for key in node.value.keys]
                duplicates = [k for k, n in collections.Counter(keys).items() if n > 1]
                self.assertEqual(duplicates, [])


if __name__ == "__main__":
    unittest.main()
