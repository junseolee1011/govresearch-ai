"""Data validator for extracted facts."""

import logging
from typing import TYPE_CHECKING

from app.state import ResearchState
from config.settings import Settings
from tools.logger import log_node_execution

if TYPE_CHECKING:
    pass


LOGGER = logging.getLogger(__name__)


def validate_fact(fact: dict) -> bool:
    """Validate a single fact.
    
    Args:
        fact: Fact dictionary to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    # Required fields must not be null
    if not fact.get("organization"):
        LOGGER.debug("Rejected fact: missing organization")
        return False
    
    if not fact.get("source_url"):
        LOGGER.debug("Rejected fact: missing source_url")
        return False
    
    if not fact.get("service_name"):
        LOGGER.debug("Rejected fact: missing service_name")
        return False
    
    # Validate URL format
    try:
        from urllib.parse import urlparse
        parsed = urlparse(fact["source_url"])
        if not parsed.scheme or not parsed.netloc:
            LOGGER.debug(f"Rejected fact: invalid URL {fact['source_url']}")
            return False
    except Exception:
        LOGGER.debug(f"Rejected fact: invalid URL {fact['source_url']}")
        return False
    
    return True


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
    
    for fact in facts:
        # Validate required fields
        if not validate_fact(fact):
            rejected.append(fact)
            continue
        
        # Check for duplicates (organization + service_name)
        key = (fact["organization"], fact["service_name"])
        if key in seen:
            LOGGER.debug(f"Rejected duplicate fact: {fact['organization']} - {fact['service_name']}")
            rejected.append(fact)
            continue
        
        seen.add(key)
        validated.append(fact)
    
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
