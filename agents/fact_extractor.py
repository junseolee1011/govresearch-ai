"""Fact extraction agent for structured data extraction with two-stage pipeline."""

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


def extract_basic_facts(doc: dict, llm) -> dict:
    """Stage 1: Extract basic facts (organization, service_name, source_url).
    
    Args:
        doc: Document dictionary with title, url, content.
        llm: Ollama LLM instance.
        
    Returns:
        Basic fact dictionary.
    """
    title = doc.get("title", "")
    url = doc.get("url", "")
    content = doc.get("content", "")
    source_quality = rank_source(url)
    
    prompt = f"""Extract the following basic information from the document about an AI service used by a public institution.

Document Title: {title}
URL: {url}
Content: {content[:1500]}

Extract these fields as JSON. Use null if not found. Do not invent values.

Required fields:
- organization: Name of the organization/institution
- service_name: Name of the AI service
- source_url: URL of the source (use the provided URL)
- source_title: Title of the source

Return only valid JSON, nothing else."""

    try:
        response = llm.invoke(prompt)
        
        # Stage 3 Debug: Raw LLM response
        print("\n========================")
        print("Stage 3")
        print("========================")
        print(f"Document: {title}")
        print(f"Raw LLM Response:")
        print(response)
        print("========================\n")
        
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            try:
                facts = json.loads(json_str)
            except json.JSONDecodeError as e:
                # Stage 4 Debug: JSON parsing error
                print("\n========================")
                print("Stage 4")
                print("========================")
                print(f"JSON Parsing Error: {e}")
                print(f"JSON String: {json_str}")
                print("========================\n")
                facts = {
                    "organization": None,
                    "service_name": None,
                    "source_url": url,
                    "source_title": title,
                }
        else:
            facts = {
                "organization": None,
                "service_name": None,
                "source_url": url,
                "source_title": title,
            }
        
        facts.setdefault("organization", None)
        facts.setdefault("service_name", None)
        facts["source_url"] = url
        facts["source_title"] = title
        facts["source_quality"] = source_quality
        
        return facts
        
    except Exception as e:
        LOGGER.error(f"Failed to extract basic facts from {title}: {e}")
        return {
            "organization": None,
            "service_name": None,
            "source_url": url,
            "source_title": title,
            "source_quality": source_quality,
        }


def enrich_facts(basic_facts: dict, doc: dict, llm) -> dict:
    """Stage 2: Enrich facts with technology, RAG, launch year, benefits.
    
    Args:
        basic_facts: Basic fact dictionary from stage 1.
        doc: Original document dictionary.
        llm: Ollama LLM instance.
        
    Returns:
        Enriched fact dictionary.
    """
    title = doc.get("title", "")
    content = doc.get("content", "")
    
    prompt = f"""Extract additional technical details from the following document.

Document Title: {title}
Content: {content[:1500]}

Extract these fields as JSON. Use null if not found. Use empty array [] if not found. Do not invent values.

Fields:
- country: Country where the service is deployed
- service_type: Type of AI service
- technology: Technology stack used
- llm_model: LLM model used
- rag: Boolean indicating if RAG is used
- launch_year: Year of launch
- purpose: Purpose of the AI service
- benefits: List of benefits
- limitations: List of limitations

Return only valid JSON, nothing else."""

    try:
        response = llm.invoke(prompt)
        
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            enrichment = json.loads(json_str)
        else:
            enrichment = {
                "country": None,
                "service_type": None,
                "technology": None,
                "llm_model": None,
                "rag": None,
                "launch_year": None,
                "purpose": None,
                "benefits": [],
                "limitations": [],
            }
        
        # Merge with basic facts
        basic_facts.update(enrichment)
        
        return basic_facts
        
    except Exception as e:
        LOGGER.error(f"Failed to enrich facts from {title}: {e}")
        # Return basic facts without enrichment
        return basic_facts


def extract_facts_from_document(doc: dict, llm) -> dict:
    """Extract structured facts from a single document using two-stage pipeline.
    
    Args:
        doc: Document dictionary with title, url, content.
        llm: Ollama LLM instance.
        
    Returns:
        Structured fact dictionary.
    """
    # Stage 1: Extract basic facts
    basic_facts = extract_basic_facts(doc, llm)
    
    # Stage 2: Enrich with technical details
    enriched_facts = enrich_facts(basic_facts, doc, llm)
    
    return enriched_facts


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
                LOGGER.info(f"Fact Extractor receives {len(search_results)} documents")
                
                if not search_results:
                    LOGGER.warning("No search results to extract facts from")
                    return {"extracted_facts": []}
                
                facts = []
                for doc in search_results:
                    fact = extract_facts_from_document(doc, llm)
                    facts.append(fact)
                
                metrics["fact_count"] = len(facts)
                LOGGER.info(f"Extracted {len(facts)} facts from {len(search_results)} documents")
                
                # Stage 5 Debug: Every extracted fact before validation
                print("\n========================")
                print("Stage 5")
                print("========================")
                print(f"Extracted facts ({len(facts)} total):")
                for i, fact in enumerate(facts, 1):
                    print(f"\nFact {i}:")
                    print(f"  Organization: {fact.get('organization', 'N/A')}")
                    print(f"  Service Name: {fact.get('service_name', 'N/A')}")
                    print(f"  Source URL: {fact.get('source_url', 'N/A')}")
                    print(f"  Source Title: {fact.get('source_title', 'N/A')}")
                print("========================\n")
                
                return {"extracted_facts": facts}
                
            except Exception as e:
                LOGGER.error("Fact extraction failed: %s", str(e), exc_info=True)
                raise
    
    return extract_facts
