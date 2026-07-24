"""Report writer agent using Ollama to generate Markdown reports from validated facts."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.state import ResearchState
from config.settings import Settings
from tools.logger import log_node_execution
from tools.ollama_client import get_llm


LOGGER = logging.getLogger(__name__)


def make_write_report(settings: Settings):
    """Factory function to create a report writer node with settings injected.

    Args:
        settings: Application settings with Ollama configuration.

    Returns:
        A node function that accepts ResearchState.
    """

    def write_report(state: ResearchState) -> dict[str, str]:
        """Generate a structured Markdown report from validated facts only.

        Args:
            state: Workflow state with validated facts.

        Returns:
            State update containing the generated Markdown report.
        """
        with log_node_execution("report_writer") as metrics:
            try:
                facts = state.get("validated_facts", [])
                
                # Quality gate: check if we have enough data
                if len(facts) < 5:
                    report = generate_insufficient_data_report(facts)
                    metrics["response_length"] = len(report)
                    LOGGER.warning(f"Insufficient data for report: {len(facts)} facts")
                    return {"report": report}
                
                # Build report from validated facts only
                report = build_report_from_facts(state, facts)
                
                metrics["response_length"] = len(report)
                metrics["fact_count"] = len(facts)
                
                # Add metadata footer
                generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
                report += f"\n\n---\n\n생성일: {generated_at}"
                
                LOGGER.info(
                    "Report writer generated report (%d characters) from %d validated facts.", len(report), len(facts)
                )
                return {"report": report}
            except Exception as e:
                LOGGER.error("Report writer failed: %s", str(e), exc_info=True)
                raise

    return write_report


def generate_insufficient_data_report(facts: list[dict]) -> str:
    """Generate report when insufficient data is available.
    
    Args:
        facts: Validated facts (may be empty or insufficient).
        
    Returns:
        Insufficient data report.
    """
    report = "# GovResearch-AI 연구 보고서\n\n"
    report += "## 데이터 부족\n\n"
    report += f"연구 데이터가 부족하여 보고서를 생성할 수 없습니다.\n\n"
    report += f"검증된 팩트 수: {len(facts)}\n"
    report += "최소 5개 이상의 검증된 팩트가 필요합니다.\n\n"
    
    if facts:
        report += "## 검증된 팩트\n\n"
        for i, fact in enumerate(facts, 1):
            org = fact.get("organization", "알 수 없음")
            service = fact.get("service_name", "알 수 없음")
            report += f"{i}. {org} - {service}\n"
    
    return report


def build_report_from_facts(state: ResearchState, facts: list[dict]) -> str:
    """Build report structure from validated facts only.
    
    Args:
        state: Workflow state with research data.
        facts: Validated facts.
        
    Returns:
        Complete Markdown report.
    """
    topic = state["topic"]
    search_queries = state.get("search_queries", [])
    search_summary = state.get("search_summary", "")
    reflection_metrics = state.get("reflection_metrics", {})
    
    # Build report sections
    report = "# GovResearch-AI 연구 보고서\n\n"
    
    # Executive Summary
    report += "## Executive Summary\n"
    report += generate_executive_summary_from_facts(topic, facts, search_summary)
    report += "\n"
    
    # Research Objective
    report += "## 연구 목표\n"
    report += f"{topic}\n\n"
    
    # Methodology
    report += "## Methodology\n"
    report += generate_real_methodology(search_queries, reflection_metrics)
    report += "\n"
    
    # Search Strategy
    report += "## Search Strategy\n"
    for i, query in enumerate(search_queries, 1):
        report += f"{i}. {query}\n"
    report += "\n"
    
    # Organization Inventory
    report += "## Organization Inventory\n"
    report += generate_organization_inventory_from_facts(facts)
    report += "\n"
    
    # Domestic Cases
    report += "## Domestic Cases\n"
    report += generate_domestic_cases_from_facts(facts)
    report += "\n"
    
    # International Cases
    report += "## International Cases\n"
    report += generate_international_cases_from_facts(facts)
    report += "\n"
    
    # Technology Stack
    report += "## Technology Stack\n"
    report += generate_technology_stack_from_facts(facts)
    report += "\n"
    
    # Implementation Patterns
    report += "## Implementation Patterns\n"
    report += generate_implementation_patterns_from_facts(facts)
    report += "\n"
    
    # Key Findings
    report += "## Key Findings\n"
    report += search_summary
    report += "\n\n"
    
    # Recommendations (only if we have enough facts)
    if len(facts) >= 5:
        report += "## Recommendations\n"
        report += generate_recommendations_from_facts(facts)
        report += "\n"
    
    # Limitations
    report += "## Limitations\n"
    report += generate_limitations_from_facts(facts, reflection_metrics)
    report += "\n"
    
    # References with citations
    report += "## References\n"
    report += generate_references_from_facts(facts)
    
    return report


def generate_executive_summary_from_facts(topic: str, facts: list[dict], summary: str) -> str:
    """Generate executive summary from facts.
    
    Args:
        topic: Research topic.
        facts: Validated facts.
        summary: Search summary.
        
    Returns:
        Executive summary.
    """
    org_count = len(set(f.get("organization") for f in facts if f.get("organization")))
    service_count = len(set(f.get("service_name") for f in facts if f.get("service_name")))
    
    report = f"본 연구는 '{topic}'에 대한 조사를 수행했습니다.\n\n"
    report += f"총 {org_count}개 기관의 {service_count}개 AI 서비스를 식별했습니다.\n\n"
    report += summary
    
    return report


def generate_real_methodology(queries: list[str], metrics: dict) -> str:
    """Generate real methodology description.
    
    Args:
        queries: Search queries used.
        metrics: Reflection metrics.
        
    Returns:
        Real methodology description.
    """
    report = "### 검색 엔진\n"
    report += "- Tavily (웹 검색)\n\n"
    
    report += "### LLM\n"
    report += "- Qwen3 via Ollama (로컬 추론)\n\n"
    
    report += "### 리트리버\n"
    report += "- ChromaDB (벡터 데이터베이스)\n\n"
    
    report += "### 리플렉션\n"
    report += "- 활성화됨 (품질 평가)\n\n"
    
    report += "### 팩트 검증\n"
    report += "- 활성화됨 (데이터 검증)\n\n"
    
    report += "### 검색 쿼리\n"
    for i, query in enumerate(queries, 1):
        report += f"{i}. {query}\n"
    
    return report


def generate_organization_inventory_from_facts(facts: list[dict]) -> str:
    """Generate organization inventory from facts.
    
    Args:
        facts: Validated facts.
        
    Returns:
        Organization inventory table.
    """
    if not facts:
        return "조직 정보를 찾을 수 없습니다.\n"
    
    report = "| 기관명 | 서비스명 | 목적 |\n"
    report += "| --- | --- | --- |\n"
    
    for fact in facts:
        org = fact.get("organization", "알 수 없음")
        service = fact.get("service_name", "알 수 없음")
        purpose = fact.get("purpose", "알 수 없음")
        report += f"| {org} | {service} | {purpose} |\n"
    
    return report


def generate_domestic_cases_from_facts(facts: list[dict]) -> str:
    """Generate domestic cases from facts.
    
    Args:
        facts: Validated facts.
        
    Returns:
        Domestic cases text.
    """
    # Since we removed country field, all facts are treated as domestic
    if not facts:
        return "국내 사례를 찾을 수 없습니다.\n"
    
    report = ""
    for fact in facts:
        org = fact.get("organization", "알 수 없음")
        service = fact.get("service_name", "알 수 없음")
        purpose = fact.get("purpose", "")
        report += f"- **{org}**: {service} ({purpose})\n"
        if purpose:
            report += f"  - 목적: {purpose[:100]}...\n"
    
    return report


def generate_international_cases_from_facts(facts: list[dict]) -> str:
    """Generate international cases from facts.
    
    Args:
        facts: Validated facts.
        
    Returns:
        International cases text.
    """
    # Since we removed country field, we cannot distinguish international cases
    return "국제 사례 분석은 현재 스키마에서 지원되지 않습니다.\n"


def generate_technology_stack_from_facts(facts: list[dict]) -> str:
    """Generate technology stack from facts.
    
    Args:
        facts: Validated facts.
        
    Returns:
        Technology stack text.
    """
    # Since we removed technology, llm_model, and rag fields, this section is simplified
    return "기술 스택 분석은 현재 스키마에서 지원되지 않습니다.\n"


def generate_implementation_patterns_from_facts(facts: list[dict]) -> str:
    """Generate implementation patterns from facts.
    
    Args:
        facts: Validated facts.
        
    Returns:
        Implementation patterns text.
    """
    # Since we removed service_type field, this section is simplified
    if not facts:
        return "구현 패턴 분석을 위한 데이터가 부족합니다.\n"
    
    report = f"총 {len(facts)}개의 AI 서비스 구현 사례가 분석되었습니다.\n"
    
    # Analyze purposes instead of service types
    purposes = {}
    for fact in facts:
        purpose = fact.get("purpose", "알 수 없음")
        purposes[purpose] = purposes.get(purpose, 0) + 1
    
    if purposes:
        report += "\n### 서비스 목적 분포\n"
        for purpose, count in sorted(purposes.items(), key=lambda x: x[1], reverse=True):
            report += f"- {purpose}: {count}개\n"
    
    return report


def generate_recommendations_from_facts(facts: list[dict]) -> str:
    """Generate recommendations from facts only.
    
    Args:
        facts: Validated facts.
        
    Returns:
        Recommendations based on facts.
    """
    # Simplified recommendations based on available fields
    if not facts:
        return "추천사항을 생성하기에 충분한 데이터가 없습니다.\n"
    
    total_count = len(facts)
    report = f"분석된 {total_count}개의 AI 서비스 구현 사례를 기반으로 다음과 같은 추천사항을 제안합니다.\n\n"
    
    # Analyze purposes
    purposes = {}
    for fact in facts:
        purpose = fact.get("purpose", "알 수 없음")
        purposes[purpose] = purposes.get(purpose, 0) + 1
    
    if purposes:
        dominant_purpose = max(purposes.items(), key=lambda x: x[1])
        report += f"가장 일반적인 서비스 목적은 '{dominant_purpose[0]}'으로 {dominant_purpose[1]}개 사례에서 확인되었습니다. "
        report += "이를 참고하여 유사한 목적의 서비스 구현을 고려할 수 있습니다.\n\n"
    
    report += f"분석된 {total_count}개 기관의 구현 사례를 벤치마킹하여 서비스 기획에 활용하십시오.\n"
    
    return report


def generate_limitations_from_facts(facts: list[dict], metrics: dict) -> str:
    """Generate limitations from facts.
    
    Args:
        facts: Validated facts.
        metrics: Reflection metrics.
        
    Returns:
        Limitations text.
    """
    fact_count = len(facts)
    org_count = len(set(f.get("organization") for f in facts if f.get("organization")))
    
    report = f"본 연구는 {fact_count}개의 검증된 팩트와 {org_count}개의 조직을 기반으로 수행되었습니다.\n\n"
    
    if fact_count < 10:
        report += "데이터 수집 범위가 제한적입니다.\n\n"
    
    report += "품질 지표:\n"
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            report += f"- {k}: {v}\n"
    
    return report


def generate_references_from_facts(facts: list[dict]) -> str:
    """Generate numbered references from validated facts.
    
    Args:
        facts: Validated facts.
        
    Returns:
        Numbered references list.
    """
    if not facts:
        return "참고 문헌이 없습니다.\n"
    
    report = ""
    for i, fact in enumerate(facts, 1):
        url = fact.get("source_url", "")
        title = fact.get("source_title", "알 수 없음")
        org = fact.get("organization", "알 수 없음")
        
        if url:
            report += f"[{i}] {org} - {title}. {url}\n"
        else:
            report += f"[{i}] {org} - {title}\n"
    
    return report
