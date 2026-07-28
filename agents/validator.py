"""Data validator for extracted facts."""

import logging
from typing import TYPE_CHECKING

from app.state import ResearchState
from config.settings import Settings
from tools.logger import log_node_execution

if TYPE_CHECKING:
    pass


LOGGER = logging.getLogger(__name__)


def validate_fact(fact: dict) -> tuple[bool, str]:
    """Validate a single fact.
    
    Args:
        fact: Fact dictionary to validate.
        
    Returns:
        Tuple of (is_valid, rejection_reason).
    """
    # Required: both organization AND service_name must be present
    org = fact.get("organization")
    service = fact.get("service_name")
    
    # Handle list types - check if any element is truthy
    def has_value(field):
        if field is None:
            return False
        if isinstance(field, list):
            return any(item for item in field if item)
        return bool(field)
    
    # Reject if organization is null
    if not has_value(org):
        return False, "organization is null"
    
    # Reject if service_name is null
    if not has_value(service):
        return False, "service_name is null"
    
    # Reject generic organization names
    generic_org_terms = ["공공기관", "정부", "기관", "문서 제목", "보도자료 제목"]
    if isinstance(org, str):
        for term in generic_org_terms:
            if term in org:
                return False, f"organization contains generic term: {term}"
    
    # Reject abstract service_name concepts
    invalid_service_terms = ["AI 전환", "AI 활용", "AX", "AI 정책", "AI 교육", "AI", "Artificial Intelligence", "ANI", "AGI", "LLM", "Generative AI", "AI technology", "AI strategy", "AI utilization"]
    if isinstance(service, str):
        for term in invalid_service_terms:
            if term in service:
                return False, f"service_name contains abstract concept: {term}"
    
    # Reject if service_name equals document title
    source_title = fact.get("source_title", "")
    if isinstance(service, str) and isinstance(source_title, str):
        if service == source_title:
            return False, "service_name equals document title"
    
    # source_url is required
    if not fact.get("source_url"):
        return False, "missing source_url"
    
    # Validate URL format
    try:
        from urllib.parse import urlparse
        parsed = urlparse(fact["source_url"])
        if not parsed.scheme or not parsed.netloc:
            return False, f"invalid URL format: {fact['source_url']}"
    except Exception as e:
        return False, f"invalid URL parsing: {str(e)}"
    
    return True, ""


def validate_facts(facts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Validate all facts and remove duplicates.
    
    Args:
        facts: List of extracted facts.
        
    Returns:
        Tuple of (validated_facts, rejected_facts).
    """
    validated = []
    rejected = []
    
    # Track seen combinations to detect duplicates
    seen = set()
    
    # Stage 6 Debug: Every rejected fact with reason
    print("\n========================")
    print("Stage 6")
    print("========================")
    print("Rejected facts:")
    
    for fact in facts:
        # Validate required fields
        is_valid, reason = validate_fact(fact)
        if not is_valid:
            org = fact.get("organization", "unknown")
            url = fact.get("source_url", "unknown")
            LOGGER.warning(f"Rejected fact from {org} ({url}): {reason}")
            print(f"Rejected: {reason}")
            print(f"  Organization: {org}")
            print(f"  URL: {url}")
            rejected.append(fact)
            continue
        
        # Check for duplicates (organization + source_url)
        key = (fact["organization"], fact["source_url"])
        if key in seen:
            org = fact.get("organization", "unknown")
            url = fact.get("source_url", "unknown")
            LOGGER.warning(f"Rejected duplicate fact: {org} ({url})")
            print(f"Rejected: duplicate")
            print(f"  Organization: {org}")
            print(f"  URL: {url}")
            rejected.append(fact)
            continue
        
        seen.add(key)
        validated.append(fact)
    
    print("========================\n")
    
    LOGGER.info(f"Validated {len(validated)} facts, rejected {len(rejected)} facts")
    LOGGER.info(f"Rejected fact count: {len(rejected)}")
    
    # Stage 7 Debug: Summary counts
    print("\n========================")
    print("Stage 7")
    print("========================")
    print(f"Extracted facts: {len(facts)}")
    print(f"Validated facts: {len(validated)}")
    print(f"Rejected facts: {len(rejected)}")
    print("========================\n")
    
    return validated, rejected


def make_validate_facts(settings: Settings):
    """Factory function to create validator node.
    
    Args:
        settings: Application settings.
        
    Returns:
        Node function for validation.
    """
    
    def validate(state: ResearchState) -> dict[str, object]:
        """Validate extracted facts.
        
        Args:
            state: Workflow state with extracted facts.
            
        Returns:
            State update with validated facts.
        """
        with log_node_execution("validator") as metrics:
            try:
                facts = state.get("extracted_facts", [])
                LOGGER.info(f"Validator receives {len(facts)} facts")
                
                if not facts:
                    LOGGER.warning("No facts to validate")
                    return {"validated_facts": [], "rejected_facts": []}
                
                validated, rejected = validate_facts(facts)
                
                metrics["validated_count"] = len(validated)
                metrics["rejected_count"] = len(rejected)
                
                LOGGER.info(f"Validation complete: {len(validated)} valid, {len(rejected)} rejected")
                
                return {
                    "validated_facts": validated,
                    "rejected_facts": rejected,
                }
                
            except Exception as e:
                LOGGER.error("Validation failed: %s", str(e), exc_info=True)
                raise
    
    return validate
