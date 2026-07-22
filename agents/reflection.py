"""Reflection agent for research quality evaluation based on validated facts."""

import logging
from typing import TYPE_CHECKING

from app.state import ResearchState
from config.settings import Settings
from tools.logger import log_node_execution

if TYPE_CHECKING:
    pass


LOGGER = logging.getLogger(__name__)


def evaluate_research_quality(state: ResearchState, settings: Settings) -> dict[str, object]:
    """Evaluate research quality based on validated facts.
    
    Args:
        state: Workflow state with validated facts.
        settings: Application settings.
        
    Returns:
        State update with reflection decision and metrics.
    """
    with log_node_execution("reflection") as metrics:
        try:
            facts = state.get("validated_facts", [])
            reflection_count = state.get("reflection_count", 0)
            
            # Calculate metrics from validated facts
            unique_organizations = set()
            government_organizations = 0
            international_cases = 0
            implementation_examples = 0
            
            for fact in facts:
                org = fact.get("organization")
                if org:
                    unique_organizations.add(org)
                
                # Count government organizations
                url = fact.get("source_url", "")
                if ".gov" in url or "gov.kr" in url or "go.kr" in url or "or.kr" in url:
                    government_organizations += 1
                
                # Count international cases
                country = fact.get("country")
                if country and country.lower() not in ["korea", "south korea", "한국", "kr"]:
                    international_cases += 1
                
                # Count implementation examples
                service_type = fact.get("service_type")
                if service_type:
                    implementation_examples += 1
            
            # Calculate duplicate ratio
            extracted_count = len(state.get("extracted_facts", []))
            validated_count = len(facts)
            duplicate_ratio = (extracted_count - validated_count) / extracted_count if extracted_count > 0 else 0
            
            # Reflection criteria
            criteria = {
                "validated_facts": validated_count,
                "unique_organizations": len(unique_organizations),
                "government_organizations": government_organizations,
                "international_cases": international_cases,
                "implementation_examples": implementation_examples,
                "duplicate_ratio": duplicate_ratio,
                "reflection_count": reflection_count,
            }
            
            # Log metrics
            LOGGER.info(f"Research quality metrics: {criteria}")
            metrics.update(criteria)
            
            # Determine if additional research is needed
            needs_research = (
                validated_count < 5
                or len(unique_organizations) < 5
                or government_organizations < 5
                or international_cases == 0
                or implementation_examples < 5
                or duplicate_ratio > 0.3
            )
            
            # Check reflection loop limit
            if reflection_count >= 2:
                LOGGER.warning("Maximum reflection loop reached, proceeding with current results")
                needs_research = False
            
            decision = "research_again" if needs_research else "continue"
            
            LOGGER.info(f"Reflection decision: {decision}")
            
            return {
                "reflection_decision": decision,
                "reflection_metrics": criteria,
                "reflection_count": reflection_count + 1,
            }
            
        except Exception as e:
            LOGGER.error("Reflection failed: %s", str(e), exc_info=True)
            raise


def make_reflect(settings: Settings):
    """Factory function to create reflection node.
    
    Args:
        settings: Application settings.
        
    Returns:
        Node function for reflection.
    """
    
    def reflect(state: ResearchState) -> dict[str, object]:
        """Evaluate research quality."""
        return evaluate_research_quality(state, settings)
    
    return reflect


def should_research_again(state: ResearchState) -> str:
    """Decide whether to perform additional research.
    
    Args:
        state: Workflow state with reflection decision.
        
    Returns:
        "research" or "continue".
    """
    decision = state.get("reflection_decision", "continue")
    return decision
