"""Fact extraction agent for structured data extraction."""

import json
import logging
from typing import TYPE_CHECKING

from app.state import ResearchState
from config.settings import Settings
from tools.logger import log_node_execution
from tools.ollama_client import get_llm
from tools.source_ranker import rank_source

if TYPE_CHECKING:
    pass


LOGGER = logging.getLogger(__name__)


def extract_facts_from_document(doc: dict, llm) -> dict:
    """Extract structured facts from a single document with strict rules.
    
    Args:
        doc: Document dictionary with title, url, content.
        llm: Ollama LLM instance.
        
    Returns:
        Structured fact dictionary with null for unknown values.
    """
    title = doc.get("title", "")
    url = doc.get("url", "")
    content = doc.get("content", "")
    source_quality = rank_source(url)
    
    prompt = f"""Extract structured information from the following document about an AI service used by a public institution.

Document Title: {title}
URL: {url}
Content: {content[:1500]}

Extract the following fields as JSON. Use null if information is not available. Do not invent values. Do not infer missing information.

Required fields:
- organization: Name of the organization/institution (null if not found)
- country: Country where the service is deployed (null if not found)
- service_name: Name of the AI service (null if not found)
- service_type: Type of AI service (null if not found)
- technology: Technology stack used (null if not found)
- llm_model: LLM model used (null if not found)
- rag: Boolean indicating if RAG is used (null if not found)
- launch_year: Year of launch (null if not found)
- purpose: Purpose of the AI service (null if not found)
- benefits: List of benefits (empty array if not found)
- limitations: List of limitations (empty array if not found)
- source_url: URL of the source
- source_title: Title of the source
- source_quality: Quality score (1-5)

Return only valid JSON, nothing else."""

    try:
        response = llm.invoke(prompt)
        
        # Try to parse JSON from response
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            facts = json.loads(json_str)
        else:
            # Fallback: create minimal fact structure with nulls
            facts = {
                "organization": None,
                "country": None,
                "service_name": None,
                "service_type": None,
                "technology": None,
                "llm_model": None,
                "rag": None,
                "launch_year": None,
                "purpose": None,
                "benefits": [],
                "limitations": [],
                "source_url": url,
                "source_title": title,
                "source_quality": source_quality,
            }
        
        # Ensure required fields exist with null defaults
        facts.setdefault("organization", None)
        facts.setdefault("country", None)
        facts.setdefault("service_name", None)
        facts.setdefault("service_type", None)
        facts.setdefault("technology", None)
        facts.setdefault("llm_model", None)
        facts.setdefault("rag", None)
        facts.setdefault("launch_year", None)
        facts.setdefault("purpose", None)
        facts.setdefault("benefits", [])
        facts.setdefault("limitations", [])
        
        # Set source fields
        facts["source_url"] = url
        facts["source_title"] = title
        facts["source_quality"] = source_quality
        
        return facts
        
    except Exception as e:
        LOGGER.error(f"Failed to extract facts from {title}: {e}")
        # Return minimal structure with nulls on error
        return {
            "organization": None,
            "country": None,
            "service_name": None,
            "service_type": None,
            "technology": None,
            "llm_model": None,
            "rag": None,
            "launch_year": None,
            "purpose": None,
            "benefits": [],
            "limitations": [],
            "source_url": url,
            "source_title": title,
            "source_quality": source_quality,
        }


def make_extract_facts(settings: Settings):
    """Factory function to create fact extractor node.
    
    Args:
        settings: Application settings.
        
    Returns:
        Node function for fact extraction.
    """
    
    def extract_facts(state: ResearchState) -> dict[str, object]:
        """Extract structured facts from all documents.
        
        Args:
            state: Workflow state with search results.
            
        Returns:
            State update with extracted facts.
        """
        with log_node_execution("fact_extractor") as metrics:
            try:
                llm = get_llm(settings)
                
                search_results = state.get("search_results", [])
                if not search_results:
                    LOGGER.warning("No search results to extract facts from")
                    return {"extracted_facts": []}
                
                facts = []
                for doc in search_results:
                    fact = extract_facts_from_document(doc, llm)
                    facts.append(fact)
                
                metrics["fact_count"] = len(facts)
                LOGGER.info(f"Extracted {len(facts)} facts from {len(search_results)} documents")
                
                return {"extracted_facts": facts}
                
            except Exception as e:
                LOGGER.error("Fact extraction failed: %s", str(e), exc_info=True)
                raise
    
    return extract_facts
