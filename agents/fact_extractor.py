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
    """Extract structured facts from a single document.
    
    Args:
        doc: Document dictionary with title, url, content.
        llm: Ollama LLM instance.
        
    Returns:
        Structured fact dictionary.
    """
    title = doc.get("title", "")
    url = doc.get("url", "")
    content = doc.get("content", "")
    source_quality = rank_source(url)
    
    prompt = f"""Extract structured information from the following document about an AI service used by a public institution.

Document Title: {title}
URL: {url}
Content: {content[:1000]}

Extract the following fields as JSON. Use null if information is not available. Do not invent values.

Required fields:
- organization: Name of the organization/institution
- service_name: Name of the AI service
- country: Country where the service is deployed
- service_type: Type of AI service (e.g., chatbot, document analysis, prediction, etc.)
- llm: LLM model used (if known)
- rag: Boolean indicating if RAG is used
- launch_year: Year of launch
- purpose: Purpose of the AI service
- benefit: Benefits or outcomes
- source_url: URL of the source
- source_quality: Quality score (1-5)

Return only valid JSON, nothing else."""

    try:
        response = llm.invoke(prompt)
        
        # Try to parse JSON from response
        # Handle cases where LLM adds extra text
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            facts = json.loads(json_str)
        else:
            # Fallback: create minimal fact structure
            facts = {
                "organization": None,
                "service_name": title,
                "country": None,
                "service_type": None,
                "llm": None,
                "rag": None,
                "launch_year": None,
                "purpose": None,
                "benefit": None,
                "source_url": url,
                "source_quality": source_quality,
            }
        
        # Ensure source_url and source_quality are set
        facts["source_url"] = url
        facts["source_quality"] = source_quality
        
        return facts
        
    except Exception as e:
        LOGGER.error(f"Failed to extract facts from {title}: {e}")
        # Return minimal structure on error
        return {
            "organization": None,
            "service_name": title,
            "country": None,
            "service_type": None,
            "llm": None,
            "rag": None,
            "launch_year": None,
            "purpose": None,
            "benefit": None,
            "source_url": url,
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
