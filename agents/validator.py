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
    """Validate a single fact with simplified rules.
    
    Args:
        fact: Fact dictionary to validate.
        
    Returns:
        Tuple of (is_valid, rejection_reason).
    """
    # Required: has_service must be true
    has_service = fact.get("has_service", False)
    if not has_service:
        return False, "has_service is false"
    
    # Required: organization must be present and not null
    org = fact.get("organization")
    if org is None or (isinstance(org, str) and not org.strip()):
        return False, "organization is null or empty"
    
    # Required: service_name must be present and not null
    service = fact.get("service_name")
    if service is None or (isinstance(service, str) and not service.strip()):
        return False, "service_name is null or empty"
    
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
    
    print("\n========================")
    print("Validator Results")
    print("========================")
    
    for i, fact in enumerate(facts, 1):
        # Validate required fields
        is_valid, reason = validate_fact(fact)
        
        org = fact.get("organization", "N/A")
        service = fact.get("service_name", "N/A")
        url = fact.get("source_url", "N/A")
        
        print(f"\nFact {i}:")
        print(f"  Organization: {org}")
        print(f"  Service Name: {service}")
        print(f"  URL: {url}")
        
        if not is_valid:
            LOGGER.warning(f"Rejected fact from {org} ({url}): {reason}")
            print(f"  Validator Result: REJECTED")
            print(f"  Reject Reason: {reason}")
            rejected.append(fact)
        else:
            # Check for duplicates (organization + source_url)
            key = (fact["organization"], fact["source_url"])
            if key in seen:
                LOGGER.warning(f"Rejected duplicate fact: {org} ({url})")
                print(f"  Validator Result: REJECTED")
                print(f"  Reject Reason: duplicate")
                rejected.append(fact)
            else:
                seen.add(key)
                validated.append(fact)
                print(f"  Validator Result: ACCEPTED")
    
    print("\n========================")
    print(f"Total facts: {len(facts)}")
    print(f"Validated: {len(validated)}")
    print(f"Rejected: {len(rejected)}")
    print("========================\n")
    
    LOGGER.info(f"Validated {len(validated)} facts, rejected {len(rejected)} facts")
    
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
