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
    """Build simplified report structure from validated facts only.
    
    Args:
        state: Workflow state with research data.
        facts: Validated facts.
        
    Returns:
        Complete Markdown report.
    """
    topic = state["topic"]
    
    # Build report sections
    report = "# GovResearch-AI 연구 보고서\n\n"
    
    # Executive Summary
    report += "## Executive Summary\n"
    report += generate_executive_summary_from_facts(topic, facts)
    report += "\n"
    
    # AI 서비스 목록
    report += "## AI 서비스 목록\n"
    report += generate_service_list_from_facts(facts)
    report += "\n"
    
    # 사례 요약
    report += "## 사례 요약\n"
    report += generate_case_summary_from_facts(facts)
    report += "\n"
    
    # References
    report += "## References\n"
    report += generate_references_from_facts(facts)
    
    return report


def generate_executive_summary_from_facts(topic: str, facts: list[dict]) -> str:
    """Generate executive summary from validated facts only.
    
    Args:
        topic: Research topic.
        facts: Validated facts.
        
    Returns:
        Executive summary.
    """
    if not facts:
        return f"본 연구는 '{topic}'에 대한 조사를 수행했습니다.\n\n검증된 AI 서비스 사례를 찾을 수 없습니다.\n"
    
    # Count unique organizations and services
    orgs = set()
    services = set()
    for f in facts:
        org = f.get("organization")
        if org:
            orgs.add(str(org))
        
        service = f.get("service_name")
        if service:
            services.add(str(service))
    
    org_count = len(orgs)
    service_count = len(services)
    
    report = f"본 연구는 '{topic}'에 대한 조사를 수행했습니다.\n\n"
    report += f"총 {org_count}개 기관의 {service_count}개 AI 서비스를 식별했습니다.\n\n"
    report += "이 보고서는 검증된 팩트만을 기반으로 작성되었습니다.\n"
    
    return report


def generate_service_list_from_facts(facts: list[dict]) -> str:
    """Generate service list table from validated facts.
    
    Args:
        facts: Validated facts.
        
    Returns:
        Service list table.
    """
    if not facts:
        return "AI 서비스 목록을 찾을 수 없습니다.\n"
    
    report = "| 기관 | 서비스 | 목적 |\n"
    report += "| --- | --- | --- |\n"
    
    for fact in facts:
        org = fact.get("organization", "알 수 없음")
        service = fact.get("service_name", "알 수 없음")
        purpose = fact.get("purpose")
        if not purpose:
            purpose = "미확인"
        report += f"| {org} | {service} | {purpose} |\n"
    
    return report


def generate_case_summary_from_facts(facts: list[dict]) -> str:
    """Generate case summary from validated facts.
    
    Args:
        facts: Validated facts.
        
    Returns:
        Case summary text.
    """
    if not facts:
        return "사례 요약을 찾을 수 없습니다.\n"
    
    report = ""
    for fact in facts:
        org = fact.get("organization", "알 수 없음")
        service = fact.get("service_name", "알 수 없음")
        purpose = fact.get("purpose")
        if purpose:
            report += f"- **{org}**: {service} - {purpose}\n"
        else:
            report += f"- **{org}**: {service}\n"
    
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
