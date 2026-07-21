"""Report writer agent using Ollama to generate Markdown reports."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.state import ResearchState
from config.settings import Settings
from tools.logger import log_node_execution
from tools.ollama_client import get_llm
from tools.prompt_loader import render_prompt


LOGGER = logging.getLogger(__name__)


def make_write_report(settings: Settings):
    """Factory function to create a report writer node with settings injected.

    Args:
        settings: Application settings with Ollama configuration.

    Returns:
        A node function that accepts ResearchState.
    """

    def write_report(state: ResearchState) -> dict[str, str]:
        """Generate a structured Markdown report using programmatic structure.

        Args:
            state: Workflow state with research plan, sources, and findings.

        Returns:
            State update containing the generated Markdown report.
        """
        with log_node_execution("report_writer") as metrics:
            try:
                llm = get_llm(settings)
                
                # Build report structure programmatically
                report = build_report_structure(state, llm)
                
                metrics["response_length"] = len(report)
                
                # Add metadata footer
                generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
                report += f"\n\n---\n\n생성일: {generated_at}"
                
                LOGGER.info(
                    "Report writer generated report (%d characters).", len(report)
                )
                return {"report": report}
            except Exception as e:
                LOGGER.error("Report writer failed: %s", str(e), exc_info=True)
                raise

    return write_report


def build_report_structure(state: ResearchState, llm) -> str:
    """Build report structure programmatically with LLM content generation.
    
    Args:
        state: Workflow state with research data.
        llm: Ollama LLM instance for content generation.
        
    Returns:
        Complete Markdown report.
    """
    topic = state["topic"]
    plan = state["plan"]
    sources = state.get("sources", [])
    findings = state.get("findings", [])
    search_queries = state.get("search_queries", [])
    search_summary = state.get("search_summary", "")
    search_results = state.get("search_results", [])
    retrieved_docs = state.get("retrieved_documents", [])
    cache_hit = state.get("cache_hit", False)
    
    # Build report sections
    report = "# GovResearch-AI 연구 보고서\n\n"
    
    # Research Objective
    report += "## 연구 목표\n"
    report += f"{topic}\n\n"
    
    # Search Queries
    report += "## 사용된 검색 쿼리\n"
    for i, query in enumerate(search_queries, 1):
        report += f"{i}. {query}\n"
    report += "\n"
    
    # Key Findings
    report += "## 주요 발견 사항\n"
    for finding in findings:
        report += f"- {finding}\n"
    report += "\n"
    
    # Retrieved Knowledge (RAG)
    if retrieved_docs:
        report += "## 검색된 지식 (Retrieved Knowledge)\n"
        for doc in retrieved_docs[:5]:
            report += f"- **{doc['title']}** (유사도: {doc.get('score', 0):.2f})\n"
            report += f"  {doc['content'][:200]}...\n"
        report += "\n"
    
    # New Findings
    if cache_hit:
        report += "## 새로운 발견 (New Findings)\n"
        report += f"{search_summary}\n\n"
    
    # Comparison
    if retrieved_docs and cache_hit:
        report += "## 비교 분석 (Comparison)\n"
        report += generate_comparison(retrieved_docs, search_summary, llm)
        report += "\n"
    
    # Service Inventory
    report += "## 서비스 인벤토리\n"
    report += "| 기관명 | 서비스명 | 서비스 유형 |\n"
    report += "| --- | --- | --- |\n"
    for source in sources:
        report += f"| {source['institution']} | {source['title']} | {source['service_type']} |\n"
    report += "\n"
    
    # Service Classification
    report += "## 서비스 분류\n"
    report += "### 서비스 유형별 분류\n"
    service_types = {}
    for source in sources:
        service_type = source["service_type"]
        service_types[service_type] = service_types.get(service_type, 0) + 1
    for service_type, count in service_types.items():
        report += f"- {service_type}: {count}\n"
    report += "\n"
    
    # References
    report += "## 참고 문헌\n"
    for source in sources:
        report += f"- [{source['title']}]({source['url']})\n"
    report += "\n"
    
    # Recommendations (LLM generated)
    report += "## 권장 사항 (Recommendations)\n"
    recommendations = generate_recommendations(topic, findings, retrieved_docs, llm)
    report += recommendations
    report += "\n"
    
    # Next Steps (LLM generated)
    report += "## 다음 단계\n"
    next_steps = generate_next_steps(topic, findings, llm)
    report += next_steps
    
    return report


def generate_next_steps(topic: str, findings: list[str], llm) -> str:
    """Generate next steps using LLM.
    
    Args:
        topic: Research topic.
        findings: List of findings.
        llm: Ollama LLM instance.
        
    Returns:
        Generated next steps text.
    """
    findings_text = "\n".join(f"- {finding}" for finding in findings)
    
    prompt = f"""다음 연구 주제와 발견 사항을 바탕으로 향후 조사 방향을 3-5개 제안하세요.

연구 주제: {topic}

발견 사항:
{findings_text}

다음 단계는 한국어로 작성하고 각 단계는 번호로 구분하세요."""
    
    try:
        return llm.invoke(prompt)
    except Exception as e:
        LOGGER.error("Failed to generate next steps: %s", str(e))
        return "향후 조사 방향을 생성하는데 실패했습니다.\n- 추가적인 웹 검색 수행\n- 전문가 인터뷰\n- 사례 심층 분석"


def generate_comparison(retrieved_docs: list[dict], search_summary: str, llm) -> str:
    """Generate comparison between retrieved knowledge and new findings.
    
    Args:
        retrieved_docs: List of retrieved documents.
        search_summary: Summary of new search results.
        llm: Ollama LLM instance.
        
    Returns:
        Generated comparison text.
    """
    retrieved_text = "\n".join(f"- {doc['title']}: {doc['content'][:150]}" for doc in retrieved_docs[:3])
    
    prompt = f"""다음 검색된 지식과 새로운 발견을 비교 분석하세요.

검색된 지식:
{retrieved_text}

새로운 발견:
{search_summary}

비교 분석은 한국어로 작성하고 다음 내용을 포함하세요:
- 일치하는 내용
- 새로운 정보
- 시간적 변화나 트렌드"""
    
    try:
        return llm.invoke(prompt)
    except Exception as e:
        LOGGER.error("Failed to generate comparison: %s", str(e))
        return "비교 분석을 생성하는데 실패했습니다."


def generate_recommendations(topic: str, findings: list[str], retrieved_docs: list[dict], llm) -> str:
    """Generate recommendations based on research findings.
    
    Args:
        topic: Research topic.
        findings: List of findings.
        retrieved_docs: List of retrieved documents.
        llm: Ollama LLM instance.
        
    Returns:
        Generated recommendations text.
    """
    findings_text = "\n".join(f"- {finding}" for finding in findings)
    
    prompt = f"""다음 연구 결과를 바탕으로 실질적인 권장 사항을 3-5개 제안하세요.

연구 주제: {topic}

발견 사항:
{findings_text}

권장 사항은 한국어로 작성하고 각 권장 사항은 번호로 구분하세요."""
    
    try:
        return llm.invoke(prompt)
    except Exception as e:
        LOGGER.error("Failed to generate recommendations: %s", str(e))
        return "권장 사항을 생성하는데 실패했습니다.\n- 추가 데이터 수집\n- 정책 검토\n- 이해관계자 협의"
