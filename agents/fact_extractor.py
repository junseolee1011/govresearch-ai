"""Fact extraction agent with simplified schema and batched LLM processing."""

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


def parse_json_with_recovery(json_str: str) -> list[dict] | None:
    """Parse JSON with recovery attempts.
    
    Args:
        json_str: JSON string to parse.
        
    Returns:
        Parsed list of facts or None if all recovery attempts fail.
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
        
        LOGGER.error("All JSON recovery attempts failed")
        return None


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
    
    # Build batch prompt with simplified schema
    prompt_parts = ["Extract structured information from the following documents about AI services used by public institutions.\n"]
    
    # Pre-extractor logging
    print("\n========================")
    print("Pre-Extractor Document Details")
    print("========================")
    
    for i, doc in enumerate(docs, 1):
        title = doc.get("title", "")
        url = doc.get("url", "")
        content = doc.get("content", "")[:1000]  # Longer content for better extraction
        
        # Log document details before extraction
        print(f"\nDocument {i}:")
        print(f"  Title: {title}")
        print(f"  URL: {url}")
        print(f"  Content Length: {len(doc.get('content', ''))}")
        print(f"  Content First 500 chars: {doc.get('content', '')[:500]}...")
        
        prompt_parts.append(f"\nDocument {i}:")
        prompt_parts.append(f"Title: {title}")
        prompt_parts.append(f"URL: {url}")
        prompt_parts.append(f"Content: {content}")
    
    print("========================\n")
    
    prompt_parts.append("\n\nFor each document, extract these fields as a JSON array:")
    prompt_parts.append("- organization: Name of the organization/institution")
    prompt_parts.append("- service_name: Name of the AI service")
    prompt_parts.append("- purpose: Purpose of the AI service")
    prompt_parts.append("- source_url: URL of the source")
    prompt_parts.append("- source_title: Title of the source")
    prompt_parts.append("\nIMPORTANT INSTRUCTIONS:")
    prompt_parts.append("- For every document, return exactly one JSON object.")
    prompt_parts.append("- If the document only describes policies, research, reports, or statistics and does NOT describe an actual AI service, return:")
    prompt_parts.append('  {"organization": null, "service_name": null, "purpose": null, "source_url": "...", "source_title": "..."}')
    prompt_parts.append("- Do NOT invent services. Only extract actual AI services mentioned in the document.")
    prompt_parts.append("- Return exactly N JSON objects, where N equals the number of input documents.")
    prompt_parts.append("- Do not omit documents. Every document must have a corresponding JSON object.")
    prompt_parts.append("- Return ONLY valid JSON. No explanations, no markdown code blocks.")
    
    prompt = "\n".join(prompt_parts)
    
    LOGGER.info(f"Prompt length: {len(prompt)} characters")
    
    try:
        # LLM call with timing
        llm_start = time.time()
        response = llm.invoke(prompt)
        llm_duration = time.time() - llm_start
        
        LOGGER.info(f"LLM response length: {len(response)} characters")
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
        facts = parse_json_with_recovery(cleaned_json)
        parse_duration = time.time() - parse_start
        
        LOGGER.info(f"JSON parsing completed in {parse_duration:.2f}s")
        
        # If parsing failed, return empty list (skip all documents in this batch)
        if facts is None:
            LOGGER.error(f"Failed to parse JSON for batch of {len(docs)} documents, skipping batch")
            return []
        
        # Ensure facts is a list
        if not isinstance(facts, list):
            LOGGER.warning(f"Expected list but got {type(facts)}, skipping batch")
            return []
        
        # Ensure we have the right number of facts
        if len(facts) != len(docs):
            LOGGER.warning(f"Expected {len(docs)} facts but got {len(facts)}, adjusting")
            # Truncate if too many, pad with None if too few
            if len(facts) > len(docs):
                facts = facts[:len(docs)]
            else:
                # Add None for missing facts (will be filtered out later)
                while len(facts) < len(docs):
                    facts.append(None)
        
        # Process facts and ensure all fields exist
        valid_facts = []
        for i, (fact, doc) in enumerate(zip(facts, docs)):
            if fact is None:
                LOGGER.warning(f"Fact {i+1} is None, skipping")
                continue
            
            # Ensure required fields exist
            fact.setdefault("organization", None)
            fact.setdefault("service_name", None)
            fact.setdefault("purpose", None)
            fact.setdefault("source_url", doc.get("url", ""))
            fact.setdefault("source_title", doc.get("title", ""))
            
            valid_facts.append(fact)
        
        LOGGER.info(f"Parsed {len(valid_facts)} facts from batch of {len(docs)} documents")
        return valid_facts
        
    except Exception as e:
        LOGGER.error(f"Batch fact extraction failed: {e}")
        return []


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
                
                # Use smaller batches for better quality (3-5 documents per batch)
                batch_size = 3
                all_facts = []
                
                total_start = time.time()
                
                for i in range(0, len(search_results), batch_size):
                    batch = search_results[i:i + batch_size]
                    LOGGER.info(f"Processing batch {i//batch_size + 1}: {len(batch)} documents")
                    
                    batch_facts = extract_facts_batch(batch, llm)
                    all_facts.extend(batch_facts)
                
                total_duration = time.time() - total_start
                LOGGER.info(f"Total fact extraction completed in {total_duration:.2f}s for {len(search_results)} documents")
                LOGGER.info(f"Parsed fact count: {len(all_facts)}")
                
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
                    print(f"  Purpose: {fact.get('purpose', 'N/A')}")
                    print(f"  Source URL: {fact.get('source_url', 'N/A')}")
                    print(f"  Source Title: {fact.get('source_title', 'N/A')}")
                print("========================\n")
                
                return {"extracted_facts": all_facts}
                
            except Exception as e:
                LOGGER.error("Fact extraction failed: %s", str(e), exc_info=True)
                raise
    
    return extract_facts
