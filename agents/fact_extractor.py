"""Fact extraction agent with batched LLM processing for performance."""

import json
import logging
import re
import time
from typing import TYPE_CHECKING

from app.state import ResearchState
from config.settings import Settings
from tools.logger import log_node_execution
from tools.ollama_client import get_llm
from tools.source_ranker import rank_source

if TYPE_CHECKING:
    pass


LOGGER = logging.getLogger(__name__)


def clean_json_response(response: str) -> str:
    """Clean LLM response to extract valid JSON.
    
    Removes markdown code blocks, extra explanations, and trailing text.
    
    Args:
        response: Raw LLM response.
        
    Returns:
        Cleaned JSON string.
    """
    # Remove markdown code blocks
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    
    # Find the first [ and last ]
    array_start = response.find('[')
    array_end = response.rfind(']') + 1
    
    if array_start >= 0 and array_end > array_start:
        return response[array_start:array_end]
    
    # Fallback: try to find { and } for single object
    obj_start = response.find('{')
    obj_end = response.rfind('}') + 1
    
    if obj_start >= 0 and obj_end > obj_start:
        return response[obj_start:obj_end]
    
    return response


def parse_json_with_recovery(json_str: str, fallback_data: list[dict]) -> list[dict]:
    """Parse JSON with recovery attempts.
    
    Args:
        json_str: JSON string to parse.
        fallback_data: Fallback data to return if parsing fails.
        
    Returns:
        Parsed list of facts or fallback data.
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        LOGGER.warning(f"JSON parsing failed: {e}, attempting recovery")
        
        # Attempt 1: Remove trailing commas
        cleaned = re.sub(r',\s*([}\]])', r'\1', json_str)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Attempt 2: Remove single quotes (replace with double quotes)
        cleaned = re.sub(r"'([^']*)':", r'"\1":', json_str)
        cleaned = re.sub(r"'([^']*)'", r'"\1"', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Attempt 3: Remove control characters
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        LOGGER.error("All JSON recovery attempts failed, using fallback")
        return fallback_data


def extract_facts_batch(docs: list[dict], llm) -> list[dict]:
    """Extract facts from a batch of documents in a single LLM call.
    
    Args:
        docs: List of document dictionaries.
        llm: Ollama LLM instance.
        
    Returns:
        List of extracted fact dictionaries.
    """
    if not docs:
        return []
    
    # Build batch prompt
    prompt_parts = ["Extract structured information from the following documents about AI services used by public institutions.\n"]
    
    for i, doc in enumerate(docs, 1):
        title = doc.get("title", "")
        url = doc.get("url", "")
        content = doc.get("content", "")[:800]  # Shorter for batching
        
        prompt_parts.append(f"\nDocument {i}:")
        prompt_parts.append(f"Title: {title}")
        prompt_parts.append(f"URL: {url}")
        prompt_parts.append(f"Content: {content}")
    
    prompt_parts.append("\n\nFor each document, extract these fields as a JSON array:")
    prompt_parts.append("- organization: Name of the organization/institution")
    prompt_parts.append("- service_name: Name of the AI service")
    prompt_parts.append("- source_url: URL of the source")
    prompt_parts.append("- source_title: Title of the source")
    prompt_parts.append("- country: Country where the service is deployed")
    prompt_parts.append("- service_type: Type of AI service")
    prompt_parts.append("- technology: Technology stack used")
    prompt_parts.append("- llm_model: LLM model used")
    prompt_parts.append("- rag: Boolean indicating if RAG is used")
    prompt_parts.append("- launch_year: Year of launch")
    prompt_parts.append("- purpose: Purpose of the AI service")
    prompt_parts.append("- benefits: List of benefits")
    prompt_parts.append("- limitations: List of limitations")
    prompt_parts.append("\nUse null if not found. Use empty array [] for lists. Do not invent values.")
    prompt_parts.append("\nReturn only a valid JSON array, nothing else.")
    
    prompt = "\n".join(prompt_parts)
    
    # Prepare fallback data
    fallback_facts = []
    for doc in docs:
        fallback_facts.append({
            "organization": None,
            "service_name": None,
            "source_url": doc.get("url", ""),
            "source_title": doc.get("title", ""),
            "country": None,
            "service_type": None,
            "technology": None,
            "llm_model": None,
            "rag": None,
            "launch_year": None,
            "purpose": None,
            "benefits": [],
            "limitations": [],
            "source_quality": rank_source(doc.get("url", "")),
        })
    
    try:
        # LLM call with timing
        llm_start = time.time()
        response = llm.invoke(prompt)
        llm_duration = time.time() - llm_start
        
        LOGGER.info(f"LLM call completed in {llm_duration:.2f}s for batch of {len(docs)} documents")
        
        # Debug: Raw LLM response
        print("\n========================")
        print("Stage 3")
        print("========================")
        print(f"Batch size: {len(docs)} documents")
        print(f"Raw LLM Response:")
        print(response[:1000] + "..." if len(response) > 1000 else response)
        print("========================\n")
        
        # JSON parsing with timing
        parse_start = time.time()
        cleaned_json = clean_json_response(response)
        facts = parse_json_with_recovery(cleaned_json, fallback_facts)
        parse_duration = time.time() - parse_start
        
        LOGGER.info(f"JSON parsing completed in {parse_duration:.2f}s")
        
        # Ensure facts is a list
        if not isinstance(facts, list):
            LOGGER.warning(f"Expected list but got {type(facts)}, using fallback")
            facts = fallback_facts
        
        # Ensure we have the right number of facts
        if len(facts) != len(docs):
            LOGGER.warning(f"Expected {len(docs)} facts but got {len(facts)}, adjusting")
            # Pad or truncate to match document count
            while len(facts) < len(docs):
                facts.append(fallback_facts[len(facts)])
            facts = facts[:len(docs)]
        
        # Add source quality and ensure all fields exist
        for i, (fact, doc) in enumerate(zip(facts, docs)):
            fact.setdefault("organization", None)
            fact.setdefault("service_name", None)
            fact.setdefault("source_url", doc.get("url", ""))
            fact.setdefault("source_title", doc.get("title", ""))
            fact.setdefault("country", None)
            fact.setdefault("service_type", None)
            fact.setdefault("technology", None)
            fact.setdefault("llm_model", None)
            fact.setdefault("rag", None)
            fact.setdefault("launch_year", None)
            fact.setdefault("purpose", None)
            fact.setdefault("benefits", [])
            fact.setdefault("limitations", [])
            fact["source_quality"] = rank_source(doc.get("url", ""))
        
        LOGGER.info(f"Extracted {len(facts)} facts from batch of {len(docs)} documents")
        return facts
        
    except Exception as e:
        LOGGER.error(f"Batch fact extraction failed: {e}")
        return fallback_facts


def make_extract_facts(settings: Settings):
    """Factory function to create fact extractor node.
    
    Args:
        settings: Application settings.
        
    Returns:
        Node function for fact extraction.
    """
    
    def extract_facts(state: ResearchState) -> dict[str, object]:
        """Extract structured facts from all documents using batching.
        
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
                
                # Process in batches of 5
                batch_size = 5
                all_facts = []
                
                total_start = time.time()
                
                for i in range(0, len(search_results), batch_size):
                    batch = search_results[i:i + batch_size]
                    LOGGER.info(f"Processing batch {i//batch_size + 1}: {len(batch)} documents")
                    
                    batch_facts = extract_facts_batch(batch, llm)
                    all_facts.extend(batch_facts)
                
                total_duration = time.time() - total_start
                LOGGER.info(f"Total fact extraction completed in {total_duration:.2f}s for {len(search_results)} documents")
                
                metrics["fact_count"] = len(all_facts)
                metrics["batch_count"] = (len(search_results) + batch_size - 1) // batch_size
                metrics["total_duration"] = total_duration
                
                # Stage 5 Debug: Every extracted fact before validation
                print("\n========================")
                print("Stage 5")
                print("========================")
                print(f"Extracted facts ({len(all_facts)} total):")
                for i, fact in enumerate(all_facts, 1):
                    print(f"\nFact {i}:")
                    print(f"  Organization: {fact.get('organization', 'N/A')}")
                    print(f"  Service Name: {fact.get('service_name', 'N/A')}")
                    print(f"  Source URL: {fact.get('source_url', 'N/A')}")
                    print(f"  Source Title: {fact.get('source_title', 'N/A')}")
                print("========================\n")
                
                return {"extracted_facts": all_facts}
                
            except Exception as e:
                LOGGER.error("Fact extraction failed: %s", str(e), exc_info=True)
                raise
    
    return extract_facts
