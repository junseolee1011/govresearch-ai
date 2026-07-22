"""Summarizer agent for research findings using Ollama."""

from __future__ import annotations

import logging

from app.state import ResearchState
from config.settings import Settings
from tools.logger import log_node_execution
from tools.ollama_client import get_llm
from tools.prompt_loader import render_prompt


LOGGER = logging.getLogger(__name__)


def make_summarize_findings(settings: Settings):
    """Factory function to create a summarizer node with settings injected.

    Args:
        settings: Application settings with Ollama configuration.

    Returns:
        A node function that accepts ResearchState.
    """

    def summarize_findings(state: ResearchState) -> dict[str, str]:
        """Generate a concise research summary using extracted facts.

        Args:
            state: Workflow state with extracted facts and retrieved documents.

        Returns:
            State update containing the generated summary.
        """
        with log_node_execution("summarizer") as metrics:
            try:
                llm = get_llm(settings)

                # Extract retrieved documents
                retrieved_docs = state.get("retrieved_documents", [])
                
                # Build retrieved context
                retrieved_context = ""
                if retrieved_docs:
                    context_snippets = []
                    for doc in retrieved_docs[:5]:  # Limit to top 5 retrieved docs
                        # Handle both Document objects and dictionaries
                        if hasattr(doc, 'title'):
                            title = doc.title
                            content = doc.content
                        else:
                            title = doc.get('title', '')
                            content = doc.get('content', '')
                        context_snippets.append(f"- {title}: {content[:300]}")
                    retrieved_context = "\n".join(context_snippets)
                    LOGGER.info(f"Using {len(retrieved_docs)} retrieved documents in summary")
                else:
                    retrieved_context = "검색된 지식이 없습니다."
                    LOGGER.info("No retrieved documents available")

                # Extract structured facts
                facts = state.get("extracted_facts", [])
                if not facts:
                    summary = "추출된 팩트가 없습니다."
                    LOGGER.warning("No facts to summarize")
                    return {"search_summary": summary}

                # Build facts summary
                facts_text = ""
                for fact in facts[:10]:  # Limit to top 10 facts
                    org = fact.get("organization", "알 수 없음")
                    service = fact.get("service_name", "알 수 없음")
                    country = fact.get("country", "알 수 없음")
                    service_type = fact.get("service_type", "알 수 없음")
                    purpose = fact.get("purpose", "")
                    facts_text += f"- {org} ({country}): {service} ({service_type}) - {purpose[:100]}\n"

                topic = state.get("topic", "연구 주제")

                # Generate summary from facts
                prompt = f"""다음 추출된 구조화된 팩트를 바탕으로 연구 주제에 대한 통합 요약을 작성하세요.

연구 주제: {topic}

검색된 지식 (Retrieved Context):
{retrieved_context}

추출된 팩트 (Extracted Facts):
{facts_text}

요약은 다음 내용을 포함해야 합니다:
- 주요 조직 및 서비스 현황
- 서비스 유형별 분포
- 국내외 사례 비교
- 공통 구현 패턴
- 기술 트렌드
- 성공 요인

요약은 한국어로 작성하고 3-5문단으로 구성하세요."""

                LOGGER.debug("Summarizer prompt length: %d characters", len(prompt))

                summary = llm.invoke(prompt)
                metrics["response_length"] = len(summary)
                metrics["fact_count"] = len(facts)

                LOGGER.info(
                    "Summarizer generated summary (%d characters) from %d facts and %d retrieved documents",
                    len(summary),
                    len(facts),
                    len(retrieved_docs),
                )
                return {"search_summary": summary}

            except Exception as e:
                LOGGER.error("Summarizer failed: %s", str(e), exc_info=True)
                raise

    return summarize_findings
