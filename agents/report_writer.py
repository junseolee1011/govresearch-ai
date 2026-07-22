"""Report writer agent using Ollama to generate Markdown reports."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.state import ResearchState
from config.settings import Settings
from tools.logger import log_node_execution
from tools.ollama_client import get_llm
from tools.prompt_loader import render_prompt
from tools.source_ranker import calculate_average_quality, categorize_sources


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
    facts = state.get("extracted_facts", [])
    search_queries = state.get("search_queries", [])
    search_summary = state.get("search_summary", "")
    reflection_metrics = state.get("reflection_metrics", {})
    
    # Build report sections
    report = "# GovResearch-AI 연구 보고서\n\n"
    
    # Executive Summary
    report += "## Executive Summary\n"
    report += generate_executive_summary(topic, search_summary, facts, llm)
    report += "\n"
    
    # Research Objective
    report += "## 연구 목표\n"
    report += f"{topic}\n\n"
    
    # Methodology
    report += "## Methodology\n"
    report += generate_methodology(search_queries, reflection_metrics, llm)
    report += "\n"
    
    # Search Strategy
    report += "## Search Strategy\n"
    for i, query in enumerate(search_queries, 1):
        report += f"{i}. {query}\n"
    report += "\n"
    
    # Organization Inventory
    report += "## Organization Inventory\n"
    report += generate_organization_inventory(facts)
    report += "\n"
    
    # Domestic Cases
    report += "## Domestic Cases\n"
    report += generate_domestic_cases(facts)
    report += "\n"
    
    # International Cases
    report += "## International Cases\n"
    report += generate_international_cases(facts)
    report += "\n"
    
    # Technology Stack
    report += "## Technology Stack\n"
    report += generate_technology_stack(facts)
    report += "\n"
    
    # Source Statistics
    report += "## Source Statistics\n"
    report += generate_source_statistics(facts)
    report += "\n"
    
    # Implementation Patterns
    report += "## Implementation Patterns\n"
    report += generate_implementation_patterns(facts, llm)
    report += "\n"
    
    # Key Findings
    report += "## Key Findings\n"
    report += search_summary
    report += "\n\n"
    
    # Recommendations
    report += "## Recommendations\n"
    report += generate_recommendations_sprint5(topic, facts, search_summary, llm)
    report += "\n"
    
    # Limitations
    report += "## Limitations\n"
    report += generate_limitations(facts, reflection_metrics, llm)
    report += "\n"
    
    # References with citations
    report += "## References\n"
    report += generate_references(facts)
    
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


def generate_executive_summary(topic: str, summary: str, facts: list[dict], llm) -> str:
    """Generate executive summary.
    
    Args:
        topic: Research topic.
        summary: Search summary.
        facts: Extracted facts.
        llm: Ollama LLM instance.
        
    Returns:
        Generated executive summary.
    """
    prompt = f"""다음 연구 결과를 바탕으로 실행 요약(Executive Summary)을 작성하세요.

연구 주제: {topic}

연구 요약:
{summary}

실행 요약은 한국어로 작성하고 2-3문단으로 구성하세요."""
    
    try:
        return llm.invoke(prompt)
    except Exception as e:
        LOGGER.error("Failed to generate executive summary: %s", str(e))
        return summary


def generate_methodology(queries: list[str], metrics: dict, llm) -> str:
    """Generate methodology section.
    
    Args:
        queries: Search queries used.
        metrics: Reflection metrics.
        llm: Ollama LLM instance.
        
    Returns:
        Generated methodology text.
    """
    metrics_text = "\n".join(f"- {k}: {v}" for k, v in metrics.items())
    
    prompt = f"""다음 정보를 바탕으로 연구 방법론(Methodology)을 작성하세요.

검색 쿼리 수: {len(queries)}
품질 지표:
{metrics_text}

방법론은 한국어로 작성하고 검색 전략, 품질 관리, 데이터 추출 방법을 설명하세요."""
    
    try:
        return llm.invoke(prompt)
    except Exception as e:
        LOGGER.error("Failed to generate methodology: %s", str(e))
        return f"검색 쿼리 {len(queries)}개를 사용하여 웹 검색을 수행했습니다."


def generate_organization_inventory(facts: list[dict]) -> str:
    """Generate organization inventory table.
    
    Args:
        facts: Extracted facts.
        
    Returns:
        Markdown table of organizations.
    """
    if not facts:
        return "조직 정보를 찾을 수 없습니다.\n"
    
    report = "| 기관명 | 국가 | 서비스명 | 서비스 유형 |\n"
    report += "| --- | --- | --- | --- |\n"
    
    for fact in facts:
        org = fact.get("organization", "알 수 없음")
        country = fact.get("country", "알 수 없음")
        service = fact.get("service_name", "알 수 없음")
        service_type = fact.get("service_type", "알 수 없음")
        report += f"| {org} | {country} | {service} | {service_type} |\n"
    
    return report


def generate_domestic_cases(facts: list[dict]) -> str:
    """Generate domestic cases section.
    
    Args:
        facts: Extracted facts.
        
    Returns:
        Domestic cases text.
    """
    domestic = [f for f in facts if f.get("country", "").lower() in ["korea", "south korea", "한국", "kr"]]
    
    if not domestic:
        return "국내 사례를 찾을 수 없습니다.\n"
    
    report = ""
    for fact in domestic[:10]:
        org = fact.get("organization", "알 수 없음")
        service = fact.get("service_name", "알 수 없음")
        purpose = fact.get("purpose", "")
        report += f"- **{org}**: {service}\n"
        if purpose:
            report += f"  - 목적: {purpose[:100]}...\n"
    
    return report


def generate_international_cases(facts: list[dict]) -> str:
    """Generate international cases section.
    
    Args:
        facts: Extracted facts.
        
    Returns:
        International cases text.
    """
    international = [f for f in facts if f.get("country", "").lower() not in ["korea", "south korea", "한국", "kr", "알 수 없음"]]
    
    if not international:
        return "국제 사례를 찾을 수 없습니다.\n"
    
    report = ""
    for fact in international[:10]:
        org = fact.get("organization", "알 수 없음")
        country = fact.get("country", "알 수 없음")
        service = fact.get("service_name", "알 수 없음")
        report += f"- **{org}** ({country}): {service}\n"
    
    return report


def generate_technology_stack(facts: list[dict]) -> str:
    """Generate technology stack section.
    
    Args:
        facts: Extracted facts.
        
    Returns:
        Technology stack text.
    """
    llms = set()
    rag_count = 0
    
    for fact in facts:
        llm = fact.get("llm")
        if llm:
            llms.add(llm)
        if fact.get("rag"):
            rag_count += 1
    
    report = ""
    if llms:
        report += "### LLM Models\n"
        for llm in llms:
            report += f"- {llm}\n"
        report += "\n"
    
    report += f"### RAG Usage\n"
    report += f"- RAG 사용 사례: {rag_count}개\n"
    
    return report


def generate_source_statistics(facts: list[dict]) -> str:
    """Generate source statistics section.
    
    Args:
        facts: Extracted facts.
        
    Returns:
        Source statistics text.
    """
    if not facts:
        return "출처 통계를 생성할 데이터가 없습니다.\n"
    
    # Build source list from facts
    sources = []
    for fact in facts:
        url = fact.get("source_url", "")
        if url:
            sources.append({"url": url, "source_quality": fact.get("source_quality", 3)})
    
    # Categorize sources
    categories = categorize_sources(sources)
    
    # Calculate average quality
    avg_quality = calculate_average_quality(sources)
    
    report = "### Source Type Distribution\n"
    for category, count in categories.items():
        report += f"- {category}: {count}\n"
    report += "\n"
    
    report += f"### Average Source Quality\n"
    report += f"- Overall quality score: {avg_quality:.2f}/5\n"
    
    return report


def generate_implementation_patterns(facts: list[dict], llm) -> str:
    """Generate implementation patterns section.
    
    Args:
        facts: Extracted facts.
        llm: Ollama LLM instance.
        
    Returns:
        Implementation patterns text.
    """
    service_types = {}
    for fact in facts:
        st = fact.get("service_type")
        if st:
            service_types[st] = service_types.get(st, 0) + 1
    
    types_text = "\n".join(f"- {t}: {c}개" for t, c in service_types.items())
    
    prompt = f"""다음 서비스 유형 분포를 바탕으로 구현 패턴을 분석하세요.

서비스 유형 분포:
{types_text}

구현 패턴은 한국어로 작성하고 공통 패턴, 기술 트렌드, 성공 요인을 설명하세요."""
    
    try:
        return llm.invoke(prompt)
    except Exception as e:
        LOGGER.error("Failed to generate implementation patterns: %s", str(e))
        return types_text


def generate_recommendations_sprint5(topic: str, facts: list[dict], summary: str, llm) -> str:
    """Generate recommendations based on Sprint 5 requirements.
    
    Args:
        topic: Research topic.
        facts: Extracted facts.
        summary: Search summary.
        llm: Ollama LLM instance.
        
    Returns:
        Generated recommendations.
    """
    prompt = f"""다음 연구 결과를 바탕으로 전문적인 권장 사항을 작성하세요.

연구 주제: {topic}

연구 요약:
{summary}

권장 사항은 실질적이고 실행 가능해야 하며, 연구 결과에서 도출된 내용이어야 합니다.
일반적인 조언이 아닌 구체적인 권장 사항을 제시하세요.

한국어로 작성하고 번호로 구분하세요."""
    
    try:
        return llm.invoke(prompt)
    except Exception as e:
        LOGGER.error("Failed to generate recommendations: %s", str(e))
        return "권장 사항을 생성하는데 실패했습니다."


def generate_limitations(facts: list[dict], metrics: dict, llm) -> str:
    """Generate limitations section.
    
    Args:
        facts: Extracted facts.
        metrics: Reflection metrics.
        llm: Ollama LLM instance.
        
    Returns:
        Limitations text.
    """
    fact_count = len(facts)
    org_count = len(set(f.get("organization") for f in facts if f.get("organization")))
    
    prompt = f"""다음 정보를 바탕으로 연구의 한계점을 작성하세요.

팩트 수: {fact_count}
조직 수: {org_count}
품질 지표: {metrics}

한계점은 한국어로 작성하고 데이터 수집 범위, 품질 제약, 기타 한계를 설명하세요."""
    
    try:
        return llm.invoke(prompt)
    except Exception as e:
        LOGGER.error("Failed to generate limitations: %s", str(e))
        return f"연구는 {fact_count}개의 팩트와 {org_count}개의 조직을 기반으로 수행되었습니다."


def generate_references(facts: list[dict]) -> str:
    """Generate numbered references.
    
    Args:
        facts: Extracted facts.
        
    Returns:
        Numbered references list.
    """
    if not facts:
        return "참고 문헌이 없습니다.\n"
    
    report = ""
    for i, fact in enumerate(facts, 1):
        url = fact.get("source_url", "")
        service = fact.get("service_name", "알 수 없음")
        org = fact.get("organization", "알 수 없음")
        
        if url:
            report += f"[{i}] {org} - {service}. {url}\n"
        else:
            report += f"[{i}] {org} - {service}\n"
    
    return report
