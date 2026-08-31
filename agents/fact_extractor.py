"""Fact extraction agent with simplified single-document processing."""

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
    
    # Remove common LLM explanations that might appear before JSON
    before_patterns = [
        r'Here is the extracted information',
        r'Here is the output',
        r'I\'ll extract',
        r'After analyzing',
        r'Output:',
        r'Document:',
    ]
    
    for pattern in before_patterns:
        matches = list(re.finditer(pattern, response, re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            response = response[last_match.end():]
    
    # Remove common LLM explanations that might appear after JSON
    explanation_patterns = [
        r'\n\nNote that',
        r'\n\nHowever',
        r'\n\nThe',
        r'\n\nPlease note',
        r'\n\nIf you',
        r'\n\nSince',
        r'\n\nBut',
        r'\n\nAlso',
        r'\n\nIn addition',
    ]
    
    for pattern in explanation_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            response = response[:match.start()]
    
    # Find the first { and last }
    obj_start = response.find('{')
    obj_end = response.rfind('}') + 1
    
    if obj_start >= 0 and obj_end > obj_start:
        return response[obj_start:obj_end]
    
    return response


def parse_json_with_recovery(json_str: str) -> dict | None:
    """Parse JSON with recovery attempts.
    
    Args:
        json_str: JSON string to parse.
        
    Returns:
        Parsed fact dictionary or None if all recovery attempts fail.
    """
    try:
        result = json.loads(json_str)
        if isinstance(result, dict):
            return result
        return None
    except json.JSONDecodeError as e:
        LOGGER.warning(f"JSON parsing failed: {e}, attempting recovery")
        
        # Attempt 1: Remove trailing commas
        cleaned = re.sub(r',\s*([}\]])', r'\1', json_str)
        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        
        # Attempt 2: Remove single quotes (replace with double quotes)
        cleaned = re.sub(r"'([^']*)':", r'"\1":', json_str)
        cleaned = re.sub(r"'([^']*)'", r'"\1"', cleaned)
        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        
        # Attempt 3: Remove control characters
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        
        # Attempt 4: Remove extra commas before closing brackets
        cleaned = re.sub(r',\s*\}', '}', json_str)
        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        
        LOGGER.error("All JSON recovery attempts failed")
        return None


def extract_facts_single(doc: dict, llm) -> dict | None:
    """Extract facts from a single document.
    
    Args:
        doc: Document dictionary.
        llm: Ollama LLM instance.
        
    Returns:
        Extracted fact dictionary or None if extraction fails.
    """
    title = doc.get("title", "")
    url = doc.get("url", "")
    content = doc.get("content", "")[:3000]
    
    # Per-document logging
    print("\n======== Document ========")
    print(f"Title: {title}")
    print(f"URL: {url}")
    print(f"Content Length: {len(doc.get('content', ''))}")
    print("========================\n")
    
    # Build prompt for single document
    prompt_parts = [
        "Analyze the following document and determine if it contains an actual AI service used by a public institution.\n",
        f"Title: {title}",
        f"URL: {url}",
        f"Content: {content}\n",
        "First, determine if this document describes a SPECIFIC, NAMED AI service or system that is actually in operation.",
        "Answer YES only if the document mentions a concrete AI service like '포트홀 감지 AI', 'AI 민원상담 챗봇', etc.",
        "Answer NO if the document only discusses:",
        "- AI concepts (AI, ANI, AGI, LLM, Generative AI)",
        "- AI policies or strategies (AI 전환, AX, AI 정책)",
        "- Generic terms (챗봇, 업무 자동화, 맞춤형 서비스)",
        "- Reports, guidelines, or case study collections",
        "- General AI utilization without specific service names\n",
        "If YES, extract these fields:",
        "- organization: Name of the organization/institution (STRING)",
        "- service_name: Name of the AI service (STRING)",
        "- purpose: Purpose of the AI service (STRING, or null if not mentioned)",
        "- source_url: URL of the source",
        "- source_title: Title of the source\n",
        "VALID organization names (actual institutions):",
        "- 한국도로공사, 근로복지공단, 인천국제공항공사, 한국조폐공사, 해양환경공단, 한국지능정보사회진흥원(NIA), 행정안전부, 서울특별시\n",
        "INVALID organization names (NEVER use these):",
        "- 공공기관, 정부, 기관, city government\n",
        "VALID service_name (actual AI service/system names):",
        "- 포트홀 감지 AI, 직업추천 AI, AI 민원상담 챗봇, 유해해양생물 조기 경보시스템\n",
        "INVALID service_name (NEVER use these):",
        "- AI, AX, AI 전환, AI 활용, AI 정책, AI 교육, 챗봇, ChatGPT, 업무 자동화, 맞춤형 서비스",
        "- 범정부 인공지능 공통 기반, RAG, 공공 인공지능(AI) 서비스 지원 사업",
        "- 공공부문 AI 도입·활용 가이드라인, 공공 AI 서비스 실증사례집\n",
        "Examples:",
        'Document: "ANI (Artificial Narrow Intelligence) is..." → {"has_service": false}',
        'Document: "AI 전환(AI Transformation, AX)을 추진하고 있다" → {"has_service": false}',
        'Document: "한국도로공사는 포트홀 감지 AI를 운영한다" → {"has_service": true, "organization": "한국도로공사", "service_name": "포트홀 감지 AI", "purpose": "포트홀 자동 탐지", "source_url": "...", "source_title": "..."}',
        'Document: "공공기관 AI 활용 현황을 알리오에 공시한다" → {"has_service": false}',
        'Document: "AI 정부 서비스 사례집을 발간했다" → {"has_service": false}',
        'Document: "챗봇을 도입하여 민원 상담을 자동화한다" → {"has_service": false}',
        'Document: "AI 민원상담 챗봇 서비스를 제공한다" → {"has_service": true, "organization": "...", "service_name": "AI 민원상담 챗봇", "purpose": "민원 상담 자동화", "source_url": "...", "source_title": "..."}',
        "\nCRITICAL OUTPUT REQUIREMENTS:",
        "- Return ONLY valid JSON. NOTHING ELSE.",
        "- Do NOT use markdown code blocks.",
        "- Do NOT explain your answer.",
        "- Do NOT write any text before or after the JSON.",
        "- Output exactly ONE JSON object.",
        "- If has_service is false: {\"has_service\": false}",
        "- If has_service is true: {\"has_service\": true, \"organization\": \"...\", \"service_name\": \"...\", \"purpose\": \"...\", \"source_url\": \"...\", \"source_title\": \"...\"}",
        "- Start with { and end with }.",
        "- IMPORTANT: Output actual Korean characters (한국어), NOT Unicode escapes like \\ud558\\ub155.",
        "- Write Korean text directly: 한국도로공사, 포트홀 감지 AI",
    ]
    
    prompt = "\n".join(prompt_parts)
    
    try:
        # LLM call
        llm_start = time.time()
        response = llm.invoke(prompt)
        llm_duration = time.time() - llm_start
        
        LOGGER.info(f"LLM call completed in {llm_duration:.2f}s for document: {title[:50]}")
        
        # Debug: Raw LLM response
        print(f"Extracted JSON: {response[:500]}..." if len(response) > 500 else f"Extracted JSON: {response}")
        print()
        
        # JSON parsing - json.loads automatically decodes Unicode escapes
        cleaned_json = clean_json_response(response)
        
        try:
            import json
            fact = json.loads(cleaned_json)
            if not isinstance(fact, dict):
                fact = parse_json_with_recovery(cleaned_json)
        except json.JSONDecodeError as e:
            LOGGER.warning(f"JSON decode failed: {e}, trying recovery")
            fact = parse_json_with_recovery(cleaned_json)
        
        if fact is None:
            LOGGER.error(f"Failed to parse JSON for document: {title[:50]}")
            print("Validator Result: REJECTED")
            print("Reject Reason: JSON parsing failed\n")
            return None
        
        # Fix: Try to recover garbled Korean text by re-encoding
        # Some LLMs output incorrect Unicode sequences that decode to garbage
        for field in ["organization", "service_name", "purpose"]:
            value = fact.get(field)
            if value and isinstance(value, str):
                # Check if the text looks like garbled Korean (many non-Korean chars)
                # Try to fix by encoding as latin-1 and decoding as utf-8
                try:
                    fixed = value.encode('latin-1').decode('utf-8')
                    # Only use if it looks more like valid Korean
                    if '\ud558' not in fixed and '\uc548' not in fixed:
                        fact[field] = fixed
                        LOGGER.debug(f"Fixed garbled text for {field}: {value} -> {fixed}")
                except:
                    pass
        
        # Debug: Show the parsed fact values with proper encoding
        print(f"Parsed organization: {fact.get('organization')}")
        print(f"Parsed service_name: {fact.get('service_name')}")
        print()
        
        # Ensure required fields - use original title to avoid LLM corruption
        fact.setdefault("has_service", False)
        fact.setdefault("organization", None)
        fact.setdefault("service_name", None)
        fact.setdefault("purpose", None)
        fact.setdefault("source_url", url)
        fact.setdefault("source_title", title)
        
        # Fix: If LLM returned corrupted Korean text, try to recover from original content
        # This handles cases where LLM returns Unicode escapes or garbled text
        for field in ["organization", "service_name", "purpose"]:
            value = fact.get(field)
            if value and isinstance(value, str):
                # Check if the value contains mostly Korean characters that might be corrupted
                # If it looks like garbage (many non-Korean, non-ASCII chars), keep the LLM output
                # Otherwise, keep as is since json.loads should have decoded Unicode escapes correctly
                pass
        
        # Convert lists to strings if needed
        for field in ["organization", "service_name", "purpose"]:
            value = fact.get(field)
            if isinstance(value, list):
                if len(value) == 1:
                    fact[field] = str(value[0])
                else:
                    fact[field] = ", ".join(str(v) for v in value if v)
        
        return fact
        
    except Exception as e:
        LOGGER.error(f"Single document fact extraction failed: {e}")
        print("Validator Result: REJECTED")
        print(f"Reject Reason: {e}\n")
        return None


def make_extract_facts(settings: Settings):
    """Factory function to create fact extractor node.
    
    Args:
        settings: Application settings.
        
    Returns:
        Node function for fact extraction.
    """
    
    def extract_facts(state: ResearchState) -> dict[str, object]:
        """Extract structured facts from all documents using single-document processing.
        
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
                
                all_facts = []
                total_start = time.time()
                
                # Process one document at a time
                for i, doc in enumerate(search_results, 1):
                    LOGGER.info(f"Processing document {i}/{len(search_results)}")
                    
                    fact = extract_facts_single(doc, llm)
                    if fact:
                        all_facts.append(fact)
                
                total_duration = time.time() - total_start
                LOGGER.info(f"Total fact extraction completed in {total_duration:.2f}s for {len(search_results)} documents")
                LOGGER.info(f"Parsed fact count: {len(all_facts)}")
                
                metrics["fact_count"] = len(all_facts)
                metrics["document_count"] = len(search_results)
                metrics["total_duration"] = total_duration
                
                # Debug: Every extracted fact before validation
                print("\n========================")
                print("Extracted Facts Summary")
                print("========================")
                print(f"Total documents processed: {len(search_results)}")
                print(f"Total facts extracted: {len(all_facts)}")
                for i, fact in enumerate(all_facts, 1):
                    print(f"\nFact {i}:")
                    print(f"  Organization: {fact.get('organization', 'N/A')}")
                    print(f"  Service Name: {fact.get('service_name', 'N/A')}")
                    print(f"  Purpose: {fact.get('purpose', 'N/A')}")
                    print(f"  Source URL: {fact.get('source_url', 'N/A')}")
                print("========================\n")
                
                return {"extracted_facts": all_facts}
                
            except Exception as e:
                LOGGER.error("Fact extraction failed: %s", str(e), exc_info=True)
                raise
    
    return extract_facts
