"""Report writer agent for PoC generating simple Markdown reports with MCP, Ollama, and RAG."""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.state import ResearchState
from config.settings import Settings
from agents.information_extractor import (
    extract_information_from_chunks,
    extract_information_from_web_results,
    merge_local_and_web_services
)
from tools.embeddings import get_reranker_model
from tools.logger import log_node_execution
from tools.ollama_client import get_llm


LOGGER = logging.getLogger(__name__)


def make_write_report(settings: Settings):
    """Factory function to create report writer node with settings injected.

    Args:
        settings: Application settings.

    Returns:
        A node function that accepts ResearchState.
    """

    def write_report(state: ResearchState) -> dict[str, str]:
        """Generate a Markdown report from search results and local documents using Ollama with RAG.

        Args:
            state: Workflow state with search results and local documents.

        Returns:
            State update containing the generated report.
        """
        with log_node_execution("report_writer") as metrics:
            try:
                topic = state["topic"]
                search_results = state.get("search_results", [])
                local_documents = state.get("local_documents", [])
                document_chunks = state.get("document_chunks", [])
                
                # Use Ollama to generate report with RAG
                report = generate_ollama_report(topic, search_results, local_documents, document_chunks, settings)
                
                metrics["web_result_count"] = len(search_results)
                metrics["local_doc_count"] = len(local_documents)
                metrics["chunk_count"] = len(document_chunks)
                LOGGER.info("Report writer generated report from %d web results, %d local documents, %d chunks using Ollama", 
                           len(search_results), len(local_documents), len(document_chunks))
                return {"report": report}

            except Exception as e:
                LOGGER.error("Report writer failed: %s", str(e), exc_info=True)
                raise

    return write_report


def verify_extraction_quality(services: list[dict]) -> dict:
    """Verify extraction quality before report generation.
    
    Args:
        services: List of extracted services.
        
    Returns:
        Verification results with issues found.
    """
    issues = {
        'duplicate_services': [],
        'missing_organizations': [],
        'total_services': len(services),
        'has_issues': False
    }
    
    # Check for duplicate service names (before normalization)
    service_names = {}
    for service in services:
        name = service['service'].lower()
        if name in service_names:
            issues['duplicate_services'].append({
                'name': service['service'],
                'chunks': [service_names[name]['chunk'], service['chunk']]
            })
            issues['has_issues'] = True
        else:
            service_names[name] = service
    
    # Check for missing organizations
    for service in services:
        if not service.get('organization'):
            issues['missing_organizations'].append({
                'service': service['service'],
                'chunk': service['chunk']
            })
            issues['has_issues'] = True
    
    return issues


def generate_ollama_report(topic: str, search_results: list[dict], local_documents: list[dict], document_chunks: list[dict], settings: Settings) -> str:
    """Generate report using Ollama LLM with RAG chunk selection.
    
    Args:
        topic: Research topic.
        search_results: Search results from Tavily MCP.
        local_documents: Local documents from Filesystem MCP (metadata only).
        document_chunks: Chunked content from local documents.
        settings: Application settings with Ollama configuration.
        
    Returns:
        Ollama-generated Markdown report.
    """
    try:
        llm = get_llm(settings)
    except Exception as e:
        LOGGER.warning(f"Failed to initialize Ollama: {e}. Using extraction-based fallback report generation.")
        # Use extraction-based approach even without Ollama
        selected_chunks = select_relevant_chunks(topic, document_chunks, top_k=15)
        services = extract_ai_services_rule_based(selected_chunks)
        merged_services = merge_duplicate_services(services)
        return generate_report_from_extraction(merged_services, topic, search_results)
    
    # Select relevant chunks (process ALL chunks for maximum extraction)
    # Bypass topic filtering to capture all AI services across all domains
    selected_chunks = document_chunks
    
    # NOTE: Use separate extraction pipelines for local and web results
    # Local PDF chunks are processed with strict filtering for quality
    # Web search results are processed separately with web-specific patterns
    # Then both are merged with deduplication
    
    # Log chunk selection
    LOGGER.info(f"[Research] Total chunks: {len(document_chunks)}")
    selected_chunk_ids = [f'#{c["chunk_id"]}' for c in selected_chunks]
    LOGGER.info(f"[Research] Selected chunks: {', '.join(selected_chunk_ids)}")
    
    # Information Extraction: Extract from local chunks
    try:
        LOGGER.info("[Information Extraction] Starting local chunk extraction")
        local_extracted_data = extract_information_from_chunks(selected_chunks, topic, settings)
        
        # Extract from web results separately (re-enabled with improved encoding handling)
        web_extracted_data = []
        if search_results:
            LOGGER.info(f"[Information Extraction] Starting web result extraction from {len(search_results)} results")
            web_extracted_data = extract_information_from_web_results(search_results, topic, settings, local_extracted_data)
        
        # Merge local and web results
        if local_extracted_data and web_extracted_data:
            LOGGER.info(f"[Information Extraction] Merging {len(local_extracted_data)} local and {len(web_extracted_data)} web services")
            extracted_data = merge_local_and_web_services(local_extracted_data, web_extracted_data)
        elif local_extracted_data:
            extracted_data = local_extracted_data
        elif web_extracted_data:
            extracted_data = web_extracted_data
        else:
            extracted_data = []
        
        if not extracted_data:
            LOGGER.warning("[Information Extraction] No data extracted from either source, using fallback")
            # Fallback to rule-based extraction
            rule_based_services = extract_ai_services_rule_based(selected_chunks)
            # Convert to new format
            services = []
            for item in rule_based_services:
                org = item.get('organization', '기관 미확인')
                service = {
                    'service': item['service'],
                    'organizations': [item['organization']] if item['organization'] else ['기관 미확인'],
                    'role': item.get('purpose', 'AI 기반 서비스'),
                    'l1_category': '보류',
                    'l2_category': '보류',
                    'chunks': [f"#{item['chunk']}"]
                }
                services.append(service)
            merged_services, merge_stats = merge_duplicate_services(services)
            return generate_report_from_extraction(merged_services, topic, search_results)
        
        LOGGER.info(f"[Information Extraction] Successfully extracted {len(extracted_data)} services (local + web merged)")
        
        # Convert extracted data to service format for report generation
        services = []
        for item in extracted_data:
            org = item['agency'] if item['agency'] else '기관 미확인'
            service = {
                'service': item['service'],
                'organizations': [org],
                'role': item.get('role', 'AI 기반 서비스'),
                'l1_category': item.get('l1_category', '보류'),
                'l2_category': item.get('l2_category', '보류'),
                'chunks': [item.get('source_chunk', f"#{item['chunk_id']}")],
                'source': item.get('source', 'local'),
                'web_urls': item.get('web_urls', [])
            }
            LOGGER.info(f"[Report] Converting service: {item['service']} with org: {org}, source: {service['source']}, L1: {service['l1_category']}, L2: {service['l2_category']}, chunk: {service['chunks']}")
            services.append(service)
        
        # Merge duplicate services
        merged_services, merge_stats = merge_duplicate_services(services)
        
        # Generate report from extracted data
        return generate_report_from_extraction(merged_services, topic, search_results)
        
    except Exception as e:
        LOGGER.error(f"[Information Extraction] Failed: {e}")
        # Fallback to rule-based extraction
        rule_based_services = extract_ai_services_rule_based(selected_chunks)
        # Convert to new format
        services = []
        for item in rule_based_services:
            org = item.get('organization', '기관 미확인')
            service = {
                'service': item.get('service', ''),
                'organizations': [org],
                'role': item.get('purpose', 'AI 기반 서비스'),
                'l1_category': '보류',
                'l2_category': '보류',
                'chunks': [f"#{item.get('chunk_id', 'unknown')}"]
            }
            LOGGER.info(f"[Report] Fallback converting service: {item.get('service', '')} with org: {org}, chunk: {service['chunks']}")
            services.append(service)
        merged_services, merge_stats = merge_duplicate_services(services)
        return generate_report_from_extraction(merged_services, topic, search_results)


def verify_extraction_quality(services: list[dict]) -> dict:
    """Verify extraction quality by checking for duplicates and missing organizations.
    
    Args:
        services: List of extracted services.
        
    Returns:
        Dictionary with verification results.
    """
    service_names = [s.get('service', '').lower() for s in services]
    name_counts = {}
    
    for name in service_names:
        if name:
            name_counts[name] = name_counts.get(name, 0) + 1
    
    duplicate_services = [
        {'name': name, 'count': count}
        for name, count in name_counts.items()
        if count > 1
    ]
    
    missing_organizations = [
        {'service': s.get('service', ''), 'chunk': s.get('chunk', '')}
        for s in services
        if not s.get('organization') or not s.get('organization').strip()
    ]
    
    return {
        'total_services': len(services),
        'duplicate_services': duplicate_services,
        'missing_organizations': missing_organizations,
        'has_issues': len(duplicate_services) > 0 or len(missing_organizations) > 0
    }


def extract_keywords(topic: str) -> list[str]:
    """Extract keywords from topic for chunk matching.
    
    Args:
        topic: Research topic.
        
    Returns:
        List of keywords.
    """
    # Remove common stopwords (Korean and English)
    stopwords = {
        '한국', '공공', '기관', '서비스', '현황', '사례', '정책', '보고서', '요약', '분석', '연구',
        'korea', 'public', 'service', 'status', 'case', 'policy', 'report', 'summary', 'analysis', 'research',
        '및', '의', '를', '에', '와', '은', '는', '이', '가', '도', '에서', '으로', '로', '만', '까지'
    }
    
    # Extract words (Korean and English)
    words = re.findall(r'[\w가-힣]+', topic)
    
    # Filter stopwords and short words
    keywords = [w for w in words if w not in stopwords and len(w) > 1]
    
    return keywords


def score_chunk_relevance(chunk: str, topic: str, keywords: list[str]) -> int:
    """Score a chunk based on its relevance to the topic.
    
    Args:
        chunk: Chunk content.
        topic: Research topic.
        keywords: Extracted keywords.
        
    Returns:
        Relevance score (higher is more relevant).
    """
    chunk_lower = chunk.lower()
    topic_lower = topic.lower()
    score = 0
    
    # 1. Exact topic match (highest weight)
    if topic_lower in chunk_lower:
        score += 50
    
    # 2. Keyword exact matches (high weight)
    for keyword in keywords:
        keyword_lower = keyword.lower()
        count = chunk_lower.count(keyword_lower)
        score += count * 10
    
    # 3. Partial phrase matches (medium weight)
    # Check for 2-3 word sequences from topic
    topic_words = topic_lower.split()
    if len(topic_words) >= 2:
        for i in range(len(topic_words) - 1):
            phrase = ' '.join(topic_words[i:i+2])
            if phrase in chunk_lower:
                score += 5
    
    if len(topic_words) >= 3:
        for i in range(len(topic_words) - 2):
            phrase = ' '.join(topic_words[i:i+3])
            if phrase in chunk_lower:
                score += 8
    
    # 4. Position bonus (earlier matches get higher score)
    for keyword in keywords:
        keyword_lower = keyword.lower()
        pos = chunk_lower.find(keyword_lower)
        if pos != -1:
            # The earlier the match, the higher the bonus
            # Position 0 gets bonus 5, position 100 gets bonus ~4
            position_bonus = max(0, 5 - (pos / 50))
            score += position_bonus
    
    # 5. Density bonus (more keywords in a short chunk = higher relevance)
    if keywords:
        keyword_count = sum(1 for kw in keywords if kw.lower() in chunk_lower)
        density = keyword_count / len(keywords)
        score += density * 5
    
    return int(score)


def filter_low_quality_chunks(chunks: list[dict]) -> list[dict]:
    """Filter out low-quality chunks (TOC, cover pages, etc.).
    
    Args:
        chunks: List of chunks with content.
        
    Returns:
        Filtered chunks.
    """
    filtered_chunks = []
    
    # Patterns to identify low-quality chunks
    toc_patterns = ['CONTENTS', '목차', '차례', '사례집', '환경일반', '사회복지일반', '공공질서및안전']
    
    for chunk in chunks:
        content = chunk.get('content', '').strip()
        
        # Check if chunk contains TOC patterns
        toc_count = sum(1 for pattern in toc_patterns if pattern in content)
        
        # If chunk has multiple TOC patterns, it's likely a TOC page
        if toc_count >= 2:
            LOGGER.warning(f"[Chunk Filter] Removed chunk #{chunk.get('chunk_id')}: contains TOC patterns")
            continue
        
        # Check if chunk is mostly TOC-like (short, list format)
        lines = content.split('\n')
        if len(lines) > 0:
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            if len(non_empty_lines) < 5 and toc_count >= 1:
                LOGGER.warning(f"[Chunk Filter] Removed chunk #{chunk.get('chunk_id')}: short TOC-like chunk")
                continue
        
        filtered_chunks.append(chunk)
    
    return filtered_chunks


def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Remove duplicate chunks by content.
    
    Args:
        chunks: List of chunks with content.
        
    Returns:
        Deduplicated chunks.
    """
    seen_contents = set()
    unique_chunks = []
    
    for chunk in chunks:
        content = chunk.get('content', '').strip()
        content_hash = hash(content)
        
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            unique_chunks.append(chunk)
        else:
            LOGGER.warning(f"[Chunk Filter] Removed duplicate chunk #{chunk.get('chunk_id')}")
    
    return unique_chunks


def detect_domain(chunk_content: str) -> str:
    """Detect the domain of a chunk based on keywords.
    
    Args:
        chunk_content: Chunk content.
        
    Returns:
        Domain name (welfare, employment, environment, legal, other).
    """
    content_lower = chunk_content.lower()
    
    # Domain keywords
    welfare_keywords = ['복지', '노인', '아동', '장애', '보건', '의료', '사회', '가족', '돌봄', '상담', '위기가구']
    employment_keywords = ['근로', '노동', '취업', '일자리', '산재', '임금', '고용', '노동부', '감독관']
    environment_keywords = ['환경', '기후', '에너지', '대기', '수질', '폐기물', '홍수', '예보', '재난']
    legal_keywords = ['법', '법령', '규제', '심사', '특허', '소송', '판결', '법제', '검색']
    
    # Count keyword matches for each domain
    welfare_score = sum(1 for kw in welfare_keywords if kw in content_lower)
    employment_score = sum(1 for kw in employment_keywords if kw in content_lower)
    environment_score = sum(1 for kw in environment_keywords if kw in content_lower)
    legal_score = sum(1 for kw in legal_keywords if kw in content_lower)
    
    # Return domain with highest score
    scores = {
        'welfare': welfare_score,
        'employment': employment_score,
        'environment': environment_score,
        'legal': legal_score
    }
    
    max_score = max(scores.values())
    if max_score == 0:
        return 'other'
    
    return max(scores, key=scores.get)


def select_relevant_chunks(topic: str, chunks: list[dict], top_k: int = 5, settings: Settings = None) -> list[dict]:
    """Select relevant chunks based on improved keyword-based scoring with reranking and quality filtering.
    
    Args:
        topic: Research topic.
        chunks: List of chunks with metadata.
        top_k: Number of top chunks to select (default 5, max 50).
        settings: Application settings for reranker initialization.
        
    Returns:
        List of selected chunks.
    """
    if not chunks:
        LOGGER.warning("[Research] No chunks available for selection")
        return []
    
    # Process all chunks for maximum extraction from PDF
    top_k = len(chunks)  # Process all chunks
    
    keywords = extract_keywords(topic)
    LOGGER.info(f"[Retriever] Extracted keywords: {keywords}")
    
    if not keywords:
        # If no keywords, return first few chunks
        LOGGER.warning("[Research] No keywords extracted, using first chunks")
        return chunks[:top_k]
    
    # Score all chunks
    scored_chunks = []
    for chunk in chunks:
        content = chunk["content"]
        score = score_chunk_relevance(content, topic, keywords)
        scored_chunks.append((chunk, score))
    
    # Sort by score (descending)
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 20 for reranking (more than final top_k to allow domain diversity filtering)
    retriever_top_k = min(top_k * 2, 20)
    retriever_selected = [chunk for chunk, score in scored_chunks[:retriever_top_k]]
    
    # Log retrieved chunks with scores
    LOGGER.info("[Retriever] Retrieved Chunks:")
    for chunk, score in scored_chunks[:retriever_top_k]:
        chunk_id = chunk["chunk_id"]
        preview = chunk["content"][:50].replace('\n', ' ')
        LOGGER.info(f"  #{chunk_id} score={score:.2f} | Preview: {preview}...")
    
    # Apply quality filtering
    filtered_chunks = filter_low_quality_chunks(retriever_selected)
    LOGGER.info(f"[Retriever] After quality filtering: {len(filtered_chunks)} chunks")
    
    # Deduplicate chunks
    deduplicated_chunks = deduplicate_chunks(filtered_chunks)
    LOGGER.info(f"[Retriever] After deduplication: {len(deduplicated_chunks)} chunks")
    
    # Apply domain diversity selection
    domain_diverse_chunks = select_domain_diverse_chunks(deduplicated_chunks, top_k)
    LOGGER.info(f"[Retriever] After domain diversity selection: {len(domain_diverse_chunks)} chunks")
    
    # Apply reranking if settings provided
    if settings and len(domain_diverse_chunks) > 0:
        try:
            reranker = get_reranker_model(settings)
            retriever_texts = [chunk["content"] for chunk in domain_diverse_chunks]
            reranked_indices = reranker.rerank(topic, retriever_texts, top_k=top_k)
            
            # Reorder chunks based on reranking
            reranked_chunks = [domain_diverse_chunks[i] for i in reranked_indices]
            
            # Log reranked chunks
            LOGGER.info("[Reranker] Final Chunks:")
            for chunk in reranked_chunks:
                LOGGER.info(f"  #{chunk['chunk_id']}")
            
            return reranked_chunks[:top_k]
        except Exception as e:
            LOGGER.warning(f"[Reranker] Reranking failed: {e}, using domain-diverse chunks")
            return domain_diverse_chunks[:top_k]
    
    return domain_diverse_chunks[:top_k]


def select_domain_diverse_chunks(chunks: list[dict], top_k: int) -> list[dict]:
    """Select chunks with domain diversity.
    
    Args:
        chunks: List of chunks with content.
        top_k: Number of chunks to select.
        
    Returns:
        List of selected chunks with domain diversity.
    """
    if len(chunks) <= top_k:
        return chunks
    
    # Detect domain for each chunk
    chunk_domains = []
    for chunk in chunks:
        domain = detect_domain(chunk["content"])
        chunk_domains.append((chunk, domain))
    
    # Group chunks by domain
    domain_groups = {}
    for chunk, domain in chunk_domains:
        if domain not in domain_groups:
            domain_groups[domain] = []
        domain_groups[domain].append(chunk)
    
    LOGGER.info(f"[Domain Diversity] Domain distribution:")
    for domain, domain_chunks in domain_groups.items():
        LOGGER.info(f"  {domain}: {len(domain_chunks)} chunks")
    
    # Select chunks from each domain proportionally
    selected = []
    num_domains = len(domain_groups)
    chunks_per_domain = max(1, top_k // num_domains)
    
    # Prioritize domains with more chunks
    sorted_domains = sorted(domain_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    for domain, domain_chunks in sorted_domains:
        # Take chunks from this domain
        take = min(chunks_per_domain, len(domain_chunks))
        selected.extend(domain_chunks[:take])
        
        if len(selected) >= top_k:
            break
    
    # If we still need more chunks, fill from remaining domains
    if len(selected) < top_k:
        for domain, domain_chunks in sorted_domains:
            if len(selected) >= top_k:
                break
            already_taken = sum(1 for c in selected if c in domain_chunks)
            remaining = len(domain_chunks) - already_taken
            if remaining > 0:
                take = min(remaining, top_k - len(selected))
                for chunk in domain_chunks:
                    if chunk not in selected and take > 0:
                        selected.append(chunk)
                        take -= 1
    
    LOGGER.info(f"[Domain Diversity] Selected {len(selected)} chunks from {num_domains} domains")
    return selected[:top_k]


def build_ollama_prompt(topic: str, search_results: list[dict], local_documents: list[dict], selected_chunks: list[dict]) -> str:
    """Build prompt for Ollama LLM with RAG chunks (PDF prioritized).
    
    Args:
        topic: Research topic.
        search_results: Search results from Tavily MCP.
        local_documents: Local documents metadata.
        selected_chunks: Selected relevant chunks.
        
    Returns:
        Prompt string for Ollama.
    """
    prompt_parts = [
        f"당신은 공공정책 전문 Research Analyst이다.",
        f"다음 자료를 종합하여 '{topic}'에 대한 Markdown 보고서를 작성하라.",
        "",
        "## 중요: 보고서 작성 규칙 (반드시 준수)",
        "",
        "### PDF 우선 원칙",
        "1. **로컬 PDF 내용을 최우선 근거로 사용한다** - 보고서의 핵심 내용은 반드시 PDF 청크에서 가져와야 한다.",
        "2. **웹 검색 결과는 보조 자료로만 사용한다** - PDF에 없는 최신 정보나 보완이 필요할 때만 웹 결과를 참고한다.",
        "3. **동일한 주제에 대해서는 PDF를 우선한다** - PDF와 웹에 같은 내용이 있으면 PDF를 기반으로 작성한다.",
        "4. **사례는 PDF에서 가져온다** - 구체적인 사례, 데이터, 통계는 PDF 내용을 우선으로 사용한다.",
        "",
        "### 보고서 품질 규칙",
        "5. **동일한 내용은 반복하지 않는다** - 중복되는 정보는 한 번만 언급한다.",
        "6. **웹 검색 결과를 그대로 복사하지 않는다** - 웹 결과는 참고만 하고 직접 요약한다.",
        "7. **Markdown 형식을 유지한다** - 제목, 목록 등의 형식을 지킨다.",
        "",
        "## 보고서 형식",
        "",
        "# 제목",
        "",
        "## Executive Summary",
        "3~5줄 요약 (PDF 내용 기반)",
        "",
        "## 핵심 내용",
        "항목별 요약 (PDF 내용 우선)",
        "",
        "## 주요 사례",
        "Bullet points (PDF 사례 우선)",
        "",
        "## 시사점",
        "Bullet points (PDF 기반 분석)",
        "",
        "## 로컬 참고자료",
        "사용한 로컬 문서 목록",
        "",
        "## 웹 참고자료",
        "사용한 URL 목록",
        "",
        "## 자료",
        ""
    ]
    
    # Add selected chunks FIRST (highest priority)
    if selected_chunks:
        prompt_parts.append("### Local PDF Chunks (최우선 참고 - 보고서의 핵심 근거)")
        prompt_parts.append("아래 PDF 내용을 기반으로 보고서를 작성하라:")
        for i, chunk in enumerate(selected_chunks, 1):
            prompt_parts.append(f"\n--- Chunk {chunk['chunk_id']} (from {chunk['source']}) ---")
            prompt_parts.append(chunk["content"])
            prompt_parts.append("")
    
    # Add web search results SECONDARY (supplementary only)
    if search_results:
        prompt_parts.append("### Web Search Results (보완용 - PDF에 없는 정보만 참고)")
        prompt_parts.append("PDF에 없는 최신 정보나 추가 사례가 필요할 때만 아래를 참고하라:")
        for i, result in enumerate(search_results[:3], 1):
            prompt_parts.append(f"\n--- 결과 {i}: {result.get('title', '알 수 없음')} ---")
            prompt_parts.append(result.get("content", "")[:800])  # Limit content length
            prompt_parts.append(f"URL: {result.get('url', '알 수 없음')}")
            prompt_parts.append("")
    
    return "\n".join(prompt_parts)


def add_references_section(report: str, local_documents: list[dict], selected_chunks: list[dict], search_results: list[dict]) -> str:
    """Add references section to the report.
    
    Args:
        report: Generated report.
        local_documents: Local documents metadata.
        selected_chunks: Selected chunks.
        search_results: Web search results.
        
    Returns:
        Report with references section.
    """
    # Find the position before the end of the report
    references = "\n\n## References\n\n"
    
    # Add local PDF references
    if local_documents:
        references += "### Local PDF\n\n"
        for doc in local_documents:
            path = doc.get("path", "알 수 없음")
            references += f"- {path}\n"
            
            # Add used chunks for this document
            doc_chunks = [c for c in selected_chunks if c["source"] == path]
            if doc_chunks:
                chunk_ids = [f'#{c["chunk_id"]}' for c in doc_chunks]
                references += f"  사용 Chunk: {', '.join(chunk_ids)}\n"
        references += "\n"
    
    # Add web references
    if search_results:
        references += "### Web\n\n"
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "알 수 없음")
            url = result.get("url", "알 수 없음")
            references += f"{i}. [{title}]({url})\n"
    
    return report + references


def extract_ai_services_from_chunks(selected_chunks: list[dict], topic: str, settings: Settings) -> list[dict]:
    """Extract AI services using LLM extraction with parallel batch processing for performance.
    
    Args:
        selected_chunks: Selected chunks with content.
        topic: Research topic.
        settings: Application settings with Ollama configuration.
        
    Returns:
        List of extracted services with chunk, organization, service, purpose.
    """
    try:
        llm = get_llm(settings)
    except Exception as e:
        LOGGER.warning(f"Failed to initialize Ollama for extraction: {e}. Using rule-based extraction.")
        return extract_ai_services_rule_based(selected_chunks)
    
    # Batch size for LLM calls
    BATCH_SIZE = 7
    MAX_WORKERS = 3  # Limit concurrent LLM calls
    
    # Split chunks into batches
    chunk_batches = [selected_chunks[i:i + BATCH_SIZE] for i in range(0, len(selected_chunks), BATCH_SIZE)]
    
    # Preprocess all chunks to remove artifacts
    preprocessed_batches = []
    for batch in chunk_batches:
        preprocessed_batch = [preprocess_chunk_content(chunk.copy()) for chunk in batch]
        preprocessed_batches.append(preprocessed_batch)
    
    all_services = []
    batch_results = {}
    
    LOGGER.info(f"[Extraction] Processing {len(selected_chunks)} chunks in {len(chunk_batches)} batches (batch_size={BATCH_SIZE}, max_workers={MAX_WORKERS})")
    
    def process_batch(batch_idx, batch):
        """Process a single batch with LLM."""
        batch_start_time = time.time()
        try:
            # Build extraction prompt for this batch
            extraction_prompt = build_batch_extraction_prompt(batch, topic)
            
            LOGGER.info(f"[Extraction] Processing Batch #{batch_idx} ({len(batch)} chunks)")
            response = llm.invoke(extraction_prompt)
            extraction_text = response.strip()
            
            batch_duration = time.time() - batch_start_time
            LOGGER.info(f"[Extraction] Batch #{batch_idx}: {batch_duration:.1f} sec")
            
            # Parse using robust JSON parser
            parsed_data = robust_json_parse(extraction_text)
            
            # Convert parsed data to service format
            batch_services = []
            for item in parsed_data:
                if isinstance(item, dict):
                    # Find the chunk content for this service
                    chunk_id = str(item.get('chunk_id', ''))
                    chunk_content = ""
                    for c in batch:
                        if str(c['chunk_id']) == chunk_id:
                            chunk_content = c['content']
                            break
                    
                    service = {
                        'chunk': chunk_id,
                        '_chunk_content': chunk_content,  # Store for validation
                        'organization': item.get('organization', '') or '',
                        'service': item.get('service_name', ''),
                        'purpose': item.get('purpose', '') or ''
                    }
                    if service['service']:  # Only add if service name exists
                        batch_services.append(service)
            
            return batch_idx, batch_services, batch_duration, None
        except Exception as e:
            batch_duration = time.time() - batch_start_time
            LOGGER.error(f"[Extraction] Batch #{batch_idx} failed after {batch_duration:.1f} sec: {e}")
            return batch_idx, [], batch_duration, str(e)
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_batch, idx, preprocessed_batches[idx-1]): idx 
            for idx, batch in enumerate(preprocessed_batches, 1)
        }
        
        for future in as_completed(futures):
            batch_idx, batch_services, batch_duration, error = future.result()
            
            if error:
                batch_results[batch_idx] = 0
            else:
                batch_results[batch_idx] = len(batch_services)
                all_services.extend(batch_services)
                
                # Detailed batch logging
                LOGGER.info(f"[Extraction] Batch #{batch_idx}: {len(batch_services)} services extracted")
                for i, service in enumerate(batch_services[:5], 1):  # Log first 5
                    LOGGER.info(f"  - Service {i}: {service['service']} (Chunk: {service['chunk']})")
                if len(batch_services) > 5:
                    LOGGER.info(f"  - ... and {len(batch_services) - 5} more services")
    
    # Log batch coverage summary
    LOGGER.info(f"[Extraction] Batch summary:")
    total_extracted = sum(batch_results.values())
    LOGGER.info(f"  Total batches: {len(chunk_batches)}")
    LOGGER.info(f"  Total services: {total_extracted}")
    for batch_idx, count in batch_results.items():
        LOGGER.info(f"  Batch #{batch_idx}: {count} services")
    
    LOGGER.info(f"Extracted {len(all_services)} AI services from {len(selected_chunks)} chunks")
    
    # Validate all extracted services
    validated_services = validate_extraction_json(all_services)
    
    return validated_services


def extract_purpose_from_chunk(service_name: str, chunk_content: str) -> str:
    """Extract purpose by finding service context in chunk (rule-based fallback).
    
    Args:
        service_name: Name of the service.
        chunk_content: Original chunk content.
        
    Returns:
        Extracted purpose (20-50 chars) or empty string.
    """
    lines = chunk_content.split('\n')
    service_line_idx = -1
    
    # Find the line containing the service name
    for i, line in enumerate(lines):
        if service_name in line:
            service_line_idx = i
            break
    
    if service_line_idx == -1:
        return ""
    
    # Extract ±3 lines around the service
    start_idx = max(0, service_line_idx - 3)
    end_idx = min(len(lines), service_line_idx + 4)
    context_lines = lines[start_idx:end_idx]
    context_text = ' '.join(context_lines)
    
    # Look for purpose indicators in the context
    purpose_keywords = ['제공', '지원', '운영', '관리', '예보', '탐지', '분석', '검색', '상담', '번역', '통역']
    
    for line in context_lines:
        for keyword in purpose_keywords:
            if keyword in line:
                # Extract this line as potential purpose
                purpose_candidate = line.strip()
                # Clean up artifacts
                purpose_candidate = re.sub(r'[▶●○■□▷◇\-\s]+', ' ', purpose_candidate)
                purpose_candidate = purpose_candidate.strip()
                
                # Check length (20-50 chars)
                if 20 <= len(purpose_candidate) <= 50:
                    return purpose_candidate
                elif len(purpose_candidate) > 50:
                    # Truncate to 50 chars
                    return purpose_candidate[:50].strip()
    
    return ""


def validate_service_name(service_name: str) -> bool:
    """Validate service name to reject promotional phrases and section titles.
    
    Args:
        service_name: Service name to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    if not service_name or not service_name.strip():
        LOGGER.warning(f"[Service Name Validation] Rejected: empty name (Reason=TitleOnly)")
        return False
    
    # Check if it ends with valid service suffixes
    valid_suffixes = ['시스템', '서비스', '플랫폼', '모델', '솔루션']
    has_valid_suffix = any(service_name.endswith(suffix) for suffix in valid_suffixes)
    
    if not has_valid_suffix:
        LOGGER.warning(f"[Service Name Validation] Rejected '{service_name}': no valid suffix (Reason=TitleOnly)")
        return False
    
    # Check for promotional phrases
    promotional_patterns = [
        '위기 속 희망을', '외로움을 줄이고', '안심은 더한', '국민 안전을 위한',
        '사각지대 없는', '막막한', '첫 단추를', '현장 중심', '사례 아카이브'
    ]
    for pattern in promotional_patterns:
        if pattern in service_name:
            LOGGER.warning(f"[Service Name Validation] Rejected '{service_name}': contains promotional phrase '{pattern}' (Reason=Caption)")
            return False
    
    # Check for section titles
    section_titles = ['환경일반', '고용노동', '사회복지일반', '공공질서및안전', '국토교통']
    for title in section_titles:
        if service_name == title or service_name.startswith(title):
            LOGGER.warning(f"[Service Name Validation] Rejected '{service_name}': section title (Reason=TOC)")
            return False
    
    return True


def validate_purpose(purpose: str, service_name: str) -> str:
    """Validate and clean purpose field.
    
    Args:
        purpose: Extracted purpose.
        service_name: Service name for reference.
        
    Returns:
        Cleaned purpose or empty string if invalid.
    """
    if not purpose or not purpose.strip():
        return ""
    
    # Check for invalid patterns
    invalid_patterns = ['CONTENTS', 'Page', '페이지', '출처', '<', '>', '▶', '●', '○', '----']
    for pattern in invalid_patterns:
        if pattern in purpose:
            LOGGER.warning(f"[Purpose Validation] Rejected purpose for '{service_name}': contains '{pattern}' (Reason=InvalidPurpose)")
            return ""
    
    # Check length (should be 20-50 characters for summary)
    if len(purpose) < 20:
        LOGGER.warning(f"[Purpose Validation] Rejected purpose for '{service_name}': too short ({len(purpose)} chars) (Reason=NoDescription)")
        return ""
    
    if len(purpose) > 100:
        LOGGER.warning(f"[Purpose Validation] Rejected purpose for '{service_name}': too long ({len(purpose)} chars) (Reason=InvalidPurpose)")
        return ""
    
    # Check if it's just repeating the service name
    if service_name.lower() in purpose.lower() and len(purpose) < 30:
        LOGGER.warning(f"[Purpose Validation] Rejected purpose for '{service_name}': just repeats service name (Reason=TitleOnly)")
        return ""
    
    # Check for excessive line breaks (indicates copied chunk content)
    if purpose.count('\n') >= 2:
        LOGGER.warning(f"[Purpose Validation] Rejected purpose for '{service_name}': too many line breaks (Reason=InvalidPurpose)")
        return ""
    
    return purpose.strip()


def validate_organization(organization: str, service_name: str, chunk_content: str) -> str:
    """Validate organization field to prevent cross-service contamination.
    
    Args:
        organization: Extracted organization.
        service_name: Service name for reference.
        chunk_content: Chunk content for proximity check.
        
    Returns:
        Validated organization or empty string if invalid.
    """
    if not organization or not organization.strip():
        return ""
    
    # Check if organization appears within 5 lines of service name
    lines = chunk_content.split('\n')
    service_line_idx = -1
    
    for i, line in enumerate(lines):
        if service_name in line:
            service_line_idx = i
            break
    
    if service_line_idx == -1:
        # Service name not found in chunk, accept organization as-is
        return organization.strip()
    
    # Check 5 lines before and after
    org_line_idx = -1
    for i, line in enumerate(lines):
        if organization in line:
            org_line_idx = i
            break
    
    if org_line_idx == -1:
        # Organization not found in chunk, reject
        LOGGER.warning(f"[Organization Validation] Rejected org '{organization}' for '{service_name}': not found in chunk")
        return ""
    
    # Check proximity
    if abs(org_line_idx - service_line_idx) > 5:
        LOGGER.warning(f"[Organization Validation] Rejected org '{organization}' for '{service_name}': too far from service (line {org_line_idx} vs {service_line_idx})")
        return ""
    
    return organization.strip()


def validate_extraction_json(services: list[dict]) -> list[dict]:
    """Final validation of all extracted service fields.
    
    Args:
        services: List of extracted services.
        
    Returns:
        List of validated services.
    """
    validated_services = []
    
    for service in services:
        service_name = service.get('service', '').strip()
        organization = service.get('organization', '').strip()
        purpose = service.get('purpose', '').strip()
        chunk_content = service.get('_chunk_content', '')
        
        # Validate service name (required)
        if not validate_service_name(service_name):
            continue
        
        # Validate purpose - if LLM purpose is invalid, try rule-based extraction
        validated_purpose = validate_purpose(purpose, service_name)
        if not validated_purpose and chunk_content:
            # Try rule-based extraction from chunk
            rule_based_purpose = extract_purpose_from_chunk(service_name, chunk_content)
            if rule_based_purpose:
                validated_purpose = rule_based_purpose
                LOGGER.info(f"[Purpose Extraction] Used rule-based extraction for '{service_name}': {validated_purpose}")
        
        # Validate organization
        validated_org = validate_organization(organization, service_name, chunk_content) if chunk_content else organization
        
        # Apply specific labels for missing fields
        if not validated_org:
            validated_org = "기관 미확인"
            LOGGER.warning(f"[Final Validation] Service '{service_name}': no organization (Reason=NoOrganization)")
        
        if not validated_purpose:
            validated_purpose = "목적 정보 없음"
            LOGGER.warning(f"[Final Validation] Service '{service_name}': no valid purpose (Reason=NoDescription)")
        
        # Build validated service
        validated_service = {
            'service': service_name,
            'organization': validated_org,
            'purpose': validated_purpose,
            'chunk': service.get('chunk', '')
        }
        
        validated_services.append(validated_service)
    
    LOGGER.info(f"[JSON Validation] Validated {len(validated_services)}/{len(services)} services")
    return validated_services


def preprocess_chunk_content(chunk: dict) -> dict:
    """Preprocess chunk content to remove artifacts before LLM extraction.
    
    Args:
        chunk: Chunk with content.
        
    Returns:
        Chunk with cleaned content.
    """
    content = chunk["content"]
    
    # Remove CONTENTS sections (entire blocks)
    content = re.sub(r'CONTENTS.*?(?=\n\n[A-Z가-힣]|$)', '', content, flags=re.DOTALL)
    
    # Remove table of contents markers
    content = re.sub(r'목차|차례|INDEX|목\s*차', '', content)
    
    # Remove page numbers (various formats including 001-999)
    content = re.sub(r'Page\s*\d+', '', content)
    content = re.sub(r'페이지\s*\d+', '', content)
    content = re.sub(r'P\.\s*\d+', '', content)
    content = re.sub(r'\n\s*\d{3}\s*\n', '\n', content)  # 3-digit page numbers
    content = re.sub(r'\n\s*\d{2,3}\s*\n', '\n', content)  # 2-3 digit numbers
    content = re.sub(r'\b\d{3}\b', '', content)  # Standalone 3-digit numbers
    
    # Remove source references (various formats)
    content = re.sub(r'<출처\s*:\s*[^>]*>', '', content)
    content = re.sub(r'<출처:.*?>', '', content)
    content = re.sub(r'출처\s*:\s*[^\n<]*', '', content)
    content = re.sub(r'<출처', '', content)
    
    # Remove image captions and diagram references
    content = re.sub(r'<.*?체계도.*?>', '', content)
    content = re.sub(r'<.*?소개자료.*?>', '', content)
    content = re.sub(r'<.*?이미지.*?>', '', content)
    content = re.sub(r'<.*?그림.*?>', '', content)
    content = re.sub(r'\[.*?그림.*?\]', '', content)
    content = re.sub(r'\[.*?이미지.*?\]', '', content)
    content = re.sub(r'\[.*?사진.*?\]', '', content)
    content = re.sub(r'그림\s*\d+.*', '', content)
    content = re.sub(r'이미지\s*\d+.*', '', content)
    
    # Remove section dividers
    content = re.sub(r'-{5,}', '', content)
    content = re.sub(r'={5,}', '', content)
    content = re.sub(r'_{5,}', '', content)
    
    # Remove bullet points and symbols that indicate structure
    content = re.sub(r'^[▶●○■□▷◇]\s*', '', content, flags=re.MULTILINE)
    
    # Remove empty pages (pages with only whitespace or artifacts)
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are only artifacts or empty
        if stripped and not re.match(r'^[\s\-_=<>]*$', stripped):
            cleaned_lines.append(line)
    content = '\n'.join(cleaned_lines)
    
    # Clean up extra whitespace
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    content = content.strip()
    
    chunk["content"] = content
    return chunk


def build_batch_extraction_prompt(chunks: list[dict], topic: str) -> str:
    """Build prompt for AI service extraction with strict rules (batch processing).
    
    Args:
        chunks: List of chunks with content.
        topic: Research topic.
        
    Returns:
        Prompt string for batch extraction.
    """
    prompt_parts = [
        "You are an information extraction system.",
        "",
        "Your primary task is to extract EVERY AI service mentioned in the provided document chunks.",
        "",
        "STRICT RULES:",
        "- Extract ONLY actual AI services operated by public organizations",
        "- DO NOT extract AI model names, APIs, or technical components",
        "- DO NOT extract research projects, support projects, or development projects",
        "- DO NOT extract features, functions, or technical capabilities",
        "- DO NOT extract systems under construction or datasets",
        "- If no AI service exists, return empty array []",
        "",
        "CRITICAL: DO NOT INCLUDE IN JSON:",
        "- AI model names (GPT, Gemma, Llama, qwen, Claude, etc.)",
        "- API names or endpoints",
        "- Contact Center, 관제센터, 주민센터, 공무원",
        "- Research projects (연구사업, 지원사업)",
        "- Technical terms (기술, 플랫폼, 시스템 구축 사업, ISP, 데이터셋)",
        "- Features (연구, 기능, 음성인식, 상담 기능, 검색 기능)",
        "- Generic terms (AI 음성상담, 행복이음, 공무원, 주민센터, 관제센터)",
        "- If it's not a service, DO NOT put it in the JSON array",
        "",
        "WHAT CONSTITUTES AN AI SERVICE:",
        "- Must be a NAMED independent service used by citizens or public officials",
        "- Must have a clear service name (not a model name or technical term)",
        "- Must use AI technology (machine learning, deep learning, NLP, etc.)",
        "- Must be implemented or operated by a public organization",
        "- Must have a clear function or purpose",
        "- Examples: AI 홍수 예보시스템, AI 운전자격확인시스템, AI 다국어 동시통역 민원안내",
        "",
        "EXTRACTION CONSTRAINTS:",
        "- ONLY extract information present in the provided chunks",
        "- DO NOT use any external knowledge or hallucinate information",
        "- Organization and purpose MUST be extracted from the SAME chunk as the service",
        "- DO NOT use organizations from other chunks or services",
        "- DO NOT use table of contents (CONTENTS, 목차, 차례) as service descriptions",
        "- DO NOT use page numbers or OCR noise as service information",
        "- If a chunk has no clear service name, ignore it",
        "- If purpose information is not available, output '목적 정보 없음'",
        "- DO NOT guess or infer missing information",
        "",
        "SERVICE NAME EXTRACTION (STRICT RULES):",
        "- Extract ONLY actual service names that end with: 시스템, 서비스, 플랫폼, 모델, 솔루션",
        "- DO NOT extract promotional phrases or slogans (e.g., '위기 속 희망을 건네는 문자 한 통', '외로움을 줄이고 안심은 더한')",
        "- DO NOT extract section titles or headings (e.g., '환경일반', '고용노동', '사회복지일반')",
        "- DO NOT extract table of contents entries",
        "- DO NOT extract generic descriptions",
        "- Service names must be actual implementation targets with clear technical terms",
        "",
        "ORGANIZATION EXTRACTION (STRICT PROXIMITY RULE):",
        "- Extract ONLY the organization that actually operates or provides the service",
        "- Look for organization name ONLY within 5 lines BEFORE or AFTER the service name",
        "- DO NOT extract organizations mentioned far away from the service",
        "- DO NOT extract participating organizations, cooperation organizations, executing organizations, or supporting organizations",
        "- DO NOT extract document-writing organizations, API-providing organizations, or model-developing organizations",
        "- DO NOT extract 주민센터, 공무원, 상담원, 관제센터 as organizations",
        "- If no operating organization is mentioned within 5 lines of the service, return null or empty string",
        "- Do NOT infer or guess organizations",
        "- Do NOT use external knowledge about organizations",
        "",
        "PURPOSE EXTRACTION (STRICT SUMMARY RULE):",
        "- Extract purpose description ONLY from the chunk text",
        "- MUST be 20-50 characters summary (not 1-2 sentences)",
        "- MUST be a summary, NOT a direct copy of the chunk",
        "- Describe what the service does, what problem it solves, what value it provides to citizens",
        "- Use your own words to summarize the purpose concisely",
        "- Example: '국지성 집중호우를 예측하여 홍수 위험을 사전에 안내한다.'",
        "- If no clear purpose description exists, return null or empty string",
        "- Do NOT use '목적 정보 없음'",
        "- Do NOT copy evaluative sentences (혁신적, 모범 사례, 높은 평가, 매우, 훌륭, 우수, 성공)",
        "- Do NOT copy CONTENTS, page numbers, source references, or image captions",
        "- Do NOT copy the entire chunk or large sections of text",
        "- Do NOT include symbols like ▶, ●, ○, ----",
        "",
        "OUTPUT FORMAT:",
        "- Return ONLY a valid JSON array",
        "- Each object must have: chunk_id, organization, service_name, purpose",
        "- chunk_id: the chunk number where the service was found",
        "- organization: the operating organization (can be null or empty string)",
        "- service_name: the exact service name from the text",
        "- purpose: a 1-2 sentence summary (can be null or empty string if not available)",
        "- NO markdown formatting, NO explanations, NO additional text",
        "",
        "DOCUMENT CHUNKS:"
    ]
    
    # Add all chunks to the prompt
    for chunk in chunks:
        prompt_parts.append(f"\n--- Chunk {chunk['chunk_id']} (from {chunk['source']}) ---")
        prompt_parts.append(chunk["content"])
        prompt_parts.append("")
    
    prompt_parts.append("\nExtract ALL AI services from the above chunks and return ONLY a JSON array:")
    
    return "\n".join(prompt_parts)


def build_extraction_prompt(chunk: dict, topic: str) -> str:
    """Build prompt for AI service extraction with strict rules (per-chunk).
    
    Args:
        chunk: Single chunk with content.
        topic: Research topic.
        
    Returns:
        Prompt string for extraction.
    """
    prompt_parts = [
        "You are an information extraction system.",
        "",
        "Your primary task is to extract EVERY AI service mentioned in the provided document chunk.",
        "",
        "STRICT RULES:",
        "- Extract ONLY actual AI services operated by public organizations",
        "- DO NOT extract AI model names, APIs, or technical components",
        "- DO NOT extract research projects, support projects, or development projects",
        "- DO NOT extract features, functions, or technical capabilities",
        "- DO NOT extract systems under construction or datasets",
        "- If no AI service exists, return empty array []",
        "",
        "CRITICAL: DO NOT INCLUDE IN JSON:",
        "- AI model names (GPT, Gemma, Llama, qwen, Claude, etc.)",
        "- API names or endpoints",
        "- Contact Center, 관제센터, 주민센터, 공무원",
        "- Research projects (연구사업, 지원사업)",
        "- Technical terms (기술, 플랫폼, 시스템 구축 사업, ISP, 데이터셋)",
        "- Features (연구, 기능, 음성인식, 상담 기능, 검색 기능)",
        "- Generic terms (AI 음성상담, 행복이음, 공무원, 주민센터, 관제센터)",
        "- If it's not a service, DO NOT put it in the JSON array",
        "",
        "WHAT CONSTITUTES AN AI SERVICE:",
        "- Must be a NAMED independent service used by citizens or public officials",
        "- Must have a clear service name (not a model name or technical term)",
        "- Must use AI technology (machine learning, deep learning, NLP, etc.)",
        "- Must be implemented or operated by a public organization",
        "- Must have a clear function or purpose",
        "- Examples: AI 홍수 예보시스템, AI 운전자격확인시스템, AI 다국어 동시통역 민원안내",
        "",
        "EXTRACTION CONSTRAINTS:",
        "- ONLY use information present in the chunk text",
        "- Do NOT use any external knowledge",
        "- Do NOT infer or hallucinate information",
        "- Do NOT modify service names from the text",
        "- Do NOT create information that doesn't exist",
        "- Do NOT summarize or generate descriptions",
        "",
        "ORGANIZATION EXTRACTION:",
        "- Extract ONLY the organization that actually operates or provides the service",
        "- Do NOT extract participating organizations, cooperation organizations, executing organizations, or supporting organizations",
        "- Do NOT extract document-writing organizations, API-providing organizations, or model-developing organizations",
        "- Do NOT extract 주민센터, 공무원, 상담원, 관제센터 as organizations",
        "- If no operating organization is mentioned near the service, return null or empty string",
        "- Do NOT infer or guess organizations",
        "- Do NOT use external knowledge about organizations",
        "",
        "PURPOSE EXTRACTION:",
        "- Extract purpose description ONLY from the chunk text",
        "- Describe what the service does, what problem it solves, what value it provides to citizens",
        "- Use 1-2 sentences maximum",
        "- Use the exact sentence or phrase from the text, or summarize from service description, first sentence, slogan, or explanation",
        "- If no purpose is mentioned, return null or empty string",
        "- Do NOT use '목적 정보 없음'",
        "- Do NOT summarize or generate purpose descriptions",
        "",
        "OUTPUT FORMAT:",
        "Return ONLY a JSON array with this structure:",
        "",
        "[",
        '  {',
        '    "organization": "actual operating organization from text or empty string",',
        '    "service_name": "exact AI service name from text",',
        '    "purpose": "exact purpose sentence from text or empty string",',
        '    "chunk_id": 1',
        '  }',
        "]",
        "",
        "CRITICAL OUTPUT REQUIREMENTS:",
        "- Return ONLY the JSON array",
        "- Do NOT explain",
        "- Do NOT use markdown formatting",
        "- Do NOT wrap with ```json",
        "- No introductory text",
        "- No closing remarks",
        "- Output must start with '[' and end with ']'",
        "- No additional text before or after the JSON array",
        "- If no services found, return []",
        "- DO NOT include non-services in the JSON array",
        "",
        "----------------------------------",
        "Document Chunk",
        "----------------------------------",
        "",
        f"Chunk {chunk['chunk_id']} (from {chunk['source']}):",
        "",
        chunk["content"],
        ""
    ]
    
    return "\n".join(prompt_parts)


def robust_json_parse(response: str) -> list:
    """Robust JSON parser with multiple recovery attempts.
    
    Args:
        response: Raw LLM response.
        
    Returns:
        Parsed JSON list or empty list if all attempts fail.
    """
    import json
    
    # Attempt 1: Direct parse
    try:
        data = json.loads(response)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    
    # Attempt 2: Remove markdown code blocks
    cleaned = response.strip()
    if cleaned.startswith('```'):
        lines = cleaned.split('\n')
        # Remove first line (```json or ```)
        if lines[0].startswith('```'):
            lines = lines[1:]
        # Remove last line (```)
        if lines[-1].strip() == '```':
            lines = lines[:-1]
        cleaned = '\n'.join(lines).strip()
    
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    
    # Attempt 3: Extract content between first '[' and last ']'
    first_bracket = cleaned.find('[')
    last_bracket = cleaned.rfind(']')
    
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        extracted = cleaned[first_bracket:last_bracket + 1]
        try:
            data = json.loads(extracted)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    
    # Attempt 4: Try json_repair if available
    try:
        import json_repair
        repaired = json_repair.repair_json(cleaned)
        data = json.loads(repaired)
        if isinstance(data, list):
            return data
    except ImportError:
        pass
    except Exception:
        pass
    
    # Attempt 5: Try dirtyjson if available
    try:
        import dirtyjson
        data = dirtyjson.loads(cleaned)
        if isinstance(data, list):
            return data
    except ImportError:
        pass
    except Exception:
        pass
    
    # All attempts failed
    LOGGER.error(f"[JSON Parser] All parsing attempts failed")
    LOGGER.debug(f"[JSON Parser] Raw response (first 500 chars): {response[:500]}")
    return []


def extract_organization_nearby(content: str, service_name: str, chunk_id: int) -> str:
    """Extract actual operating organization name near the service name.
    
    Args:
        content: Chunk content.
        service_name: Name of the AI service.
        chunk_id: Chunk ID for logging.
        
    Returns:
        Extracted operating organization name.
    """
    # Find service name position in content
    service_pos = content.lower().find(service_name.lower())
    if service_pos == -1:
        return ''
    
    # Extract ±150 characters around the service
    start = max(0, service_pos - 150)
    end = min(len(content), service_pos + len(service_name) + 150)
    context = content[start:end]
    
    # Look for headings (lines that end with specific patterns or are all caps)
    lines = content.split('\n')
    service_line_idx = -1
    for i, line in enumerate(lines):
        if service_name.lower() in line.lower():
            service_line_idx = i
            break
    
    if service_line_idx != -1:
        # Look for heading before service (within 5 lines)
        for i in range(max(0, service_line_idx - 5), service_line_idx):
            line = lines[i].strip()
            # Check if line looks like a heading (short, ends with specific patterns)
            if len(line) < 50 and any(line.endswith(suffix) for suffix in ['부', '청', '공단', '공사', '원', '센터', '재단', '처', '위원회']):
                return line
    
    # Strict organization patterns (only specific suffixes)
    org_patterns = [
        r'(기후에너지환경부|고용노동부|보건복지부|행정안전부|법제처|경찰청|국방부|교육부|과학기술정보통신부|산업통상자원부|해양수산부|농림축산식품부|여성가족부|문화체육관광부|국토교통부|국세청|관세청|조달청|통계청|기상청|방송통신위원회|금융위원회|공정거래위원회|국민권익위원회|개인정보보호위원회|원자력안전위원회)',
        r'(한국지능정보사회진흥원|한국사회보장정보원|근로복지공단|국민건강보험공단|국민연금공단|산업안전보건공단|고용노동공단|한국산업기술진흥원|한국과학기술기획평가원|한국연구재단)',
        r'([가-힣]{2,4}부|[가-힣]{2,4}청|[가-힣]{2,4}공단|[가-힣]{2,4}공사|[가-힣]{2,4}원|[가-힣]{2,4}센터|[가-힣]{2,4}재단)',
    ]
    
    # Exclude patterns (words that should NOT be treated as organizations)
    exclude_patterns = [
        r'안부', r'상담', r'복지', r'서비스', r'시스템', r'플랫폼', r'모델',
        r'지원', r'관리', r'운영', r'개발', r'연구', r'분석', r'조사',
        r'정보', r'데이터', r'기술', r'사업', r'정책', r'전략', r'계획',
        r'협력', r'참여', r'수행', r'지원기관', r'협력기관', r'관제센터',
    ]
    
    # Find organizations in context
    organizations = []
    for pattern in org_patterns:
        matches = re.findall(pattern, context)
        for match in matches:
            # Exclude matches that contain excluded patterns
            is_excluded = False
            for exclude in exclude_patterns:
                if re.search(exclude, match):
                    is_excluded = True
                    break
            if not is_excluded and match not in organizations:
                organizations.append(match)
    
    if organizations:
        # Return the first (closest) organization found
        return organizations[0]
    
    return ''


def post_process_services(services: list[dict], settings: Settings) -> list[dict]:
    """Post-process extracted services with normalization and LLM enhancement.
    
    Args:
        services: List of extracted services from rule-based extraction.
        settings: Application settings with Ollama configuration.
        
    Returns:
        List of post-processed services.
    """
    processed = []
    
    for service in services:
        # Separate technology from service name
        clean_service_name, technologies = separate_technology_from_service(service['service'])
        
        # Resolve organization into main/operating
        org_resolution = resolve_organization(service['organization'])
        
        # Summarize purpose using LLM
        summarized_purpose = summarize_purpose_llm(
            service['purpose'], 
            clean_service_name, 
            settings
        )
        
        processed.append({
            'chunk': service['chunk'],
            'service': clean_service_name,
            'technologies': technologies,
            'organization_main': org_resolution['main'],
            'organization_operating': org_resolution['operating'],
            'purpose': summarized_purpose
        })
    
    return processed


def extract_ai_services_rule_based(selected_chunks: list[dict]) -> list[dict]:
    """Extract AI services using rule-based patterns (fallback).
    
    Args:
        selected_chunks: Selected chunks with content.
        
    Returns:
        List of extracted services.
    """
    services = []
    
    # Very specific AI service patterns - only match complete service names
    ai_service_patterns = [
        # Specific known service patterns from the document
        r'AI\s*홍수\s*예보\s*시스템',
        r'AI\s*홍수예보\s*시스템',
        r'AI\s*기반\s*산재조사\s*분류\s*모델',
        r'AI\s*산재조사\s*분류\s*모델',
        r'AI\s*돌봄\s*서비스',
        r'AI\s*근로감독관\s*지원\s*시스템',
        r'AI\s*근로감독관\s*AI\s*지원\s*시스템',
        r'AI\s*법령검색\s*서비스',
        r'AI\s*법령\s*검색\s*서비스',
        r'AI\s*특허심사\s*지원\s*시스템',
        r'AI\s*민원\s*상담\s*시스템',
        r'AI\s*노인말벗\s*서비스',
        r'AI\s*노인말벗',
        r'AI\s*복지\s*상담\s*서비스',
        r'AI\s*복지상담\s*서비스',
        r'AI\s*안전망\s*시스템',
        r'AI\s*모니터링\s*시스템',
        r'AI\s*예보\s*플랫폼',
        r'AI\s*ARS\s*시스템',
        r'AI\s*챗봇',
        r'AI\s*상담\s*봇',
        r'AI\s*음성상담',
        r'AI\s*스마트\s*정수장',
        r'AI\s*기반\s*스마트\s*정수장',
        # Generic but complete patterns
        r'AI\s*[가-힣a-zA-Z0-9\s]{3,15}\s*(시스템|서비스|플랫폼)',
        r'[가-힣a-zA-Z0-9\s]{3,15}\s*AI\s*(시스템|서비스|플랫폼)',
    ]
    
    # Exclude terms (not actual services)
    exclude_terms = [
        '사업', '지원사업', 'R&D', '실증', '개발', '연구', '과제', '프로젝트',
        'API', 'endpoint', 'interface', 'sdk',
        'GPT', 'Claude', 'Llama', 'Qwen', 'Gemma', 'Mistral', 'Phi', 'Falcon', 'Yi', 'Baichuan', 'InternLM',
        '플랫폼', 'platform', 'hub', 'portal'
    ]
    
    for chunk in selected_chunks:
        content = chunk['content']
        chunk_id = chunk['chunk_id']
        
        # Try to find AI services
        ai_services = []
        for pattern in ai_service_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                service_name = match.strip()
                # Normalize whitespace
                service_name = re.sub(r'\s+', ' ', service_name)
                
                # Check if service name contains exclude terms
                is_excluded = False
                for exclude in exclude_terms:
                    if exclude.lower() in service_name.lower():
                        is_excluded = True
                        break
                
                if is_excluded:
                    continue
                
                # Only add if it's a complete service name (has at least 3 meaningful parts)
                if len(service_name) >= 5 and service_name not in ai_services:
                    ai_services.append(service_name)
        
        # If found, add to services with nearby organization
        for service in ai_services:
            # Extract organization near the service
            org = extract_organization_nearby(content, service, chunk_id)
            
            # Try to extract purpose from context around the service
            purpose = extract_purpose_from_context(content, service)
            
            services.append({
                'chunk': str(chunk_id),
                'organization': org,
                'service': service,
                'purpose': purpose
            })
    
    return services


def summarize_purpose_llm(purpose: str, service_name: str, settings: Settings) -> str:
    """Summarize purpose using LLM to 20-40 characters.
    
    Args:
        purpose: Original purpose text.
        service_name: Service name for context.
        settings: Application settings with Ollama configuration.
        
    Returns:
        Summarized purpose (20-40 characters).
    """
    try:
        llm = get_llm(settings)
    except Exception as e:
        LOGGER.warning(f"Failed to initialize Ollama for purpose summarization: {e}")
        return purpose[:40] if len(purpose) > 40 else purpose
    
    prompt = f"""Summarize the following AI service purpose to 20-40 Korean characters.

Service: {service_name}
Original purpose: {purpose}

Requirements:
- 20-40 characters
- Focus on core function
- Remove promotional language
- Concise and clear

Output ONLY the summarized purpose. No explanation."""

    try:
        response = llm.invoke(prompt)
        summarized = response.strip()
        
        # Clean up any markdown or extra text
        summarized = re.sub(r'[`\'"()]', '', summarized)
        
        # Ensure length constraint
        if len(summarized) > 40:
            summarized = summarized[:40]
        elif len(summarized) < 20 and len(purpose) > 20:
            # If too short, use first 40 chars of original
            summarized = purpose[:40]
        
        return summarized
    except Exception as e:
        LOGGER.warning(f"LLM purpose summarization failed: {e}")
        return purpose[:40] if len(purpose) > 40 else purpose


def separate_technology_from_service(service_name: str) -> tuple[str, list[str]]:
    """Separate technology terms from service name.
    
    Args:
        service_name: Original service name.
        
    Returns:
        Tuple of (clean_service_name, technologies)
    """
    technology_keywords = [
        'ARS', 'LLM', 'RAG', 'OCR', 'AI', '멀티모달', '딥러닝', 
        '머신러닝', '자연어처리', 'NLP', '컴퓨터비전', '음성인식',
        '텍스트마이닝', '데이터마이닝', '클라우드', '빅데이터'
    ]
    
    technologies = []
    clean_name = service_name
    
    for tech in technology_keywords:
        if tech in service_name:
            technologies.append(tech)
            # Remove from service name if it's a standalone technology term
            # Keep it if it's part of the service name (like "AI 홍수예보시스템")
            if tech not in ['AI']:  # Keep AI in service names
                clean_name = clean_name.replace(tech, '').strip()
    
    # Clean up extra spaces and parentheses
    clean_name = re.sub(r'\s+', ' ', clean_name)
    clean_name = re.sub(r'\(\s*\)', '', clean_name)
    
    return clean_name, technologies


def extract_purpose_from_context(content: str, service_name: str) -> str:
    """Extract purpose description from service description (summarize from chunk, exclude evaluation).
    
    Args:
        content: Chunk content.
        service_name: Name of the AI service.
        
    Returns:
        Extracted purpose description (1-2 sentences).
    """
    # Find service name position in content
    service_pos = content.lower().find(service_name.lower())
    if service_pos == -1:
        return ''
    
    # Extract ±400 characters around the service for context
    start = max(0, service_pos - 200)
    end = min(len(content), service_pos + len(service_name) + 200)
    context = content[start:end]
    
    # Look for service description, first sentence, slogan, or explanation
    purpose_indicators = [
        r'{}[^.]*?(?:서비스|시스템|플랫폼)[^.]*?\.'.format(re.escape(service_name)),
        r'{}[^.]*?(?:제공|지원|해결|운영|관리)[^.]*?\.'.format(re.escape(service_name)),
        r'{}[^.]*?(?:목적|용도|기능|역할)[^.]*?\.'.format(re.escape(service_name)),
    ]
    
    for pattern in purpose_indicators:
        match = re.search(pattern, context, re.IGNORECASE)
        if match:
            purpose = match.group(0).strip()
            # Clean up the purpose
            purpose = re.sub(r'[•·\-\*]', '', purpose)
            # Extract 1-2 sentences
            sentences = re.split(r'[.!?]', purpose)
            if sentences:
                first_sentence = sentences[0].strip()
                # Exclude evaluation text
                evaluation_terms = ['혁신적', '모범 사례', '높은 평가', '매우', '훌륭', '우수', '성공']
                if not any(term in first_sentence for term in evaluation_terms) and len(first_sentence) > 10:
                    result = first_sentence + '.'
                    # Add second sentence if available and relevant
                    if len(sentences) > 1 and len(sentences[1].strip()) > 10:
                        result += ' ' + sentences[1].strip() + '.'
                    return result
    
    # If no specific purpose found, look for sentences around service name
    sentences = re.split(r'[.!?]', context)
    for i, sentence in enumerate(sentences):
        if service_name.lower() in sentence.lower():
            # Return the current sentence if it describes purpose
            if any(word in sentence.lower() for word in ['제공', '지원', '해결', '서비스', '운영', '관리', '상담', '예보', '탐지', '분석', '검색', '확인', '안내']):
                cleaned = sentence.strip()
                # Exclude evaluation text
                evaluation_terms = ['혁신적', '모범 사례', '높은 평가', '매우', '훌륭', '우수', '성공']
                if not any(term in cleaned for term in evaluation_terms) and len(cleaned) > 10:
                    return cleaned + '.'
            # Return the next sentence if it exists
            if i + 1 < len(sentences):
                next_sentence = sentences[i + 1].strip()
                evaluation_terms = ['혁신적', '모범 사례', '높은 평가', '매우', '훌륭', '우수', '성공']
                if not any(term in next_sentence for term in evaluation_terms) and len(next_sentence) > 10:
                    return next_sentence + '.'
    
    return ''


def clean_pdf_residual_text(text: str) -> str:
    """Remove PDF residual text patterns (page numbers, sources, etc.).
    
    Args:
        text: Text to clean.
        
    Returns:
        Cleaned text.
    """
    # Remove page numbers
    text = re.sub(r'Page\s*\d+', '', text)
    text = re.sub(r'페이지\s*\d+', '', text)
    
    # Remove source references
    text = re.sub(r'<출처\s*:\s*[^>]*>', '', text)
    text = re.sub(r'<출처:.*?>', '', text)
    text = re.sub(r'출처\s*:\s*[^<\n]*', '', text)
    
    # Remove standalone numbers (likely page numbers)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'\n\s*\d{2,3}\s*\n', '\n', text)  # 2-3 digit numbers on their own lines
    text = re.sub(r'\n\s*\d{4}\s*\n', '\n', text)  # 4-digit numbers (years) on their own lines
    
    # Remove diagram/chart artifacts
    text = re.sub(r'<.*?체계도.*?>', '', text)
    text = re.sub(r'<.*?소개자료.*?>', '', text)
    text = re.sub(r'출처\s*:\s*[^\n]+', '', text)
    
    # Remove multiple consecutive newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def resolve_organization(organization: str) -> dict[str, str]:
    """Resolve organization into main and operating agencies.
    
    Args:
        organization: Original organization string.
        
    Returns:
        Dict with 'main' and 'operating' keys.
    """
    # Check for parentheses format: "기관(하위기관)"
    if '(' in organization and ')' in organization:
        main_part = organization[:organization.index('(')].strip()
        operating_part = organization[organization.index('(')+1:organization.index(')')].strip()
        return {'main': main_part, 'operating': operating_part}
    
    # Check for known ministry-affiliate relationships
    ministry_affiliates = {
        '고용노동부': ['근로복지공단', '한국공인노무사회'],
        '보건복지부': ['한국사회보장정보원', '국민건강보험공단'],
        '국토교통부': ['한국교통안전공단', '국토교통과학기술진흥원'],
        '행정안전부': ['국립과학수사연구원', '한국지능정보사회진흥원'],
        '과학기술정보통신부': ['한국과학기술기획평가원', '한국연구재단'],
    }
    
    for ministry, affiliates in ministry_affiliates.items():
        for affiliate in affiliates:
            if affiliate in organization:
                return {'main': ministry, 'operating': affiliate}
    
    # If no special format, treat as main agency only
    return {'main': organization, 'operating': ''}


def analyze_chunk_coverage(chunk: dict, services: list[dict]) -> dict[str, str]:
    """Analyze why a chunk has 0 services extracted.
    
    Args:
        chunk: Chunk with content.
        services: Services extracted from this chunk.
        
    Returns:
        Dict with analysis results.
    """
    content = chunk['content']
    chunk_id = chunk['chunk_id']
    
    if services:
        return {'status': 'success', 'count': len(services), 'reason': ''}
    
    # Analyze why no services were found
    reasons = []
    
    # Check if chunk contains only technology terms
    tech_terms = ['AI', 'LLM', 'RAG', 'OCR', '딥러닝', '머신러닝', '자연어처리', 'NLP']
    has_tech = any(term in content for term in tech_terms)
    has_service = any(term in content for term in ['시스템', '서비스', '플랫폼'])
    
    if has_tech and not has_service:
        reasons.append('only_technology')
    
    # Check if chunk is only explanatory/introductory
    intro_terms = ['소개', '개요', '배경', '현황', '추세', '현황']
    if any(term in content for term in intro_terms):
        reasons.append('only_explanation')
    
    # Check if chunk is too short
    if len(content) < 100:
        reasons.append('too_short')
    
    # Check if chunk contains only general AI mentions
    if 'AI' in content and not any(term in content for term in ['시스템', '서비스', '플랫폼', '모델']):
        reasons.append('only_general_ai')
    
    # Check if chunk is duplicate content
    if '동일' in content or '중복' in content:
        reasons.append('duplicate')
    
    if not reasons:
        reasons.append('unknown')
    
    return {
        'status': 'no_services',
        'count': 0,
        'reason': ','.join(reasons),
        'content_preview': content[:100] + '...' if len(content) > 100 else content
    }


def normalize_service_name(service_name: str) ->(str, str):
    """Normalize service name for duplicate detection with enhanced logic.
    
    Args:
        service_name: Original service name.
        
    Returns:
        Tuple of (normalized_name, canonical_name)
    """
    # Remove leading/trailing whitespace
    normalized = service_name.strip()
    
    # Remove consecutive whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Normalize AI기반 ↔ AI 기반
    normalized = normalized.replace('AI기반', 'AI 기반')
    
    # Normalize AI당직 ↔ AI 당직
    normalized = normalized.replace('AI당직', 'AI 당직')
    
    # Remove special characters (keep Korean, English, numbers, spaces)
    normalized = re.sub(r'[^\w\s가-힣]', '', normalized)
    
    # Remove parentheses and their contents for base normalization
    base_name = re.sub(r'\([^)]*\)', '', normalized)
    
    # Remove hyphens
    base_name = base_name.replace('-', '')
    
    # Remove common suffixes for fuzzy matching
    suffixes = ['시스템', '서비스', '플랫폼', '모델', '지원', '관리', '운영']
    for suffix in suffixes:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
    
    # Handle specific variants
    if '당직' in base_name:
        base_name = 'ai 당직'
    
    if '노동법' in base_name and '상담' in base_name:
        base_name = 'ai 노동법상담'
    
    if '돌봄' in base_name:
        base_name = 'ai 돌봄'
    
    if '홍수' in base_name and '예보' in base_name:
        base_name = 'ai 홍수예보'
    
    if '산재' in base_name and '조사' in base_name:
        base_name = 'ai 산재조사'
    
    # Normalize service name if > 30 characters
    if len(service_name) > 30:
        # Extract core name by removing common prefixes/suffixes
        core_name = service_name
        # Remove common prefixes
        prefixes = ['AI 기반 ', 'AI ', '인공지능 기반 ', '인공지능 ', '디지털 기반 ', '개인 맞춤형 ']
        for prefix in prefixes:
            if core_name.startswith(prefix):
                core_name = core_name[len(prefix):]
        # Remove common suffixes
        suffixes = [' 시스템', ' 서비스', ' 플랫폼', ' 모델', ' 업무처리 시스템', ' 추천 AI 시스템']
        for suffix in suffixes:
            if core_name.endswith(suffix):
                core_name = core_name[:-len(suffix)]
        # If core name is reasonable length, use it
        if len(core_name) > 5 and len(core_name) < 30:
            normalized = 'AI ' + core_name if 'AI' not in core_name else core_name
            base_name = core_name.lower().strip()
    
    return base_name.lower().strip(), normalized.strip()


def classify_service_type(service: dict) -> str:
    """Classify service type based on name and content with strict exclusion rules.
    
    Args:
        service: Service dict with name, organization, purpose, chunk.
        
    Returns:
        Service type: SERVICE, PROJECT, MODEL, API, OTHER
    """
    service_name = service['service'].lower()
    purpose = service.get('purpose', '').lower() if service.get('purpose') else ''
    
    # Strict exclusion list - these are NOT services
    exclusion_terms = [
        # Model names
        'gpt', 'gemma', 'llama', 'qwen', 'claude', 'mistral', 'phi', 'falcon', 'yi', 'baichuan', 'internlm',
        # Technical terms
        'api', 'endpoint', 'interface', 'sdk', 'framework', 'isp', '데이터셋', 'dataset',
        # Generic terms
        'contact center', '관제센터', '주민센터', '공무원', '상담원',
        # Project/business terms
        '연구사업', '지원사업', '시스템 구축 사업', '연구', '기술', '플랫폼',
        # Feature/function terms
        '음성인식', '상담 기능', '검색 기능', '기능', 'ai 음성상담',
        # System names that are not services
        '행복이음', '행복이음 시스템',
        # Generic AI system patterns
        '추천 ai 시스템', '업무처리 시스템', '디지털 기반 신규 업무처리 시스템',
    ]
    
    for term in exclusion_terms:
        if term in service_name:
            return 'OTHER'
    
    # Check for generic service names (only AI, 시스템, 플랫폼)
    generic_patterns = [r'^ai$', r'^ai 시스템$', r'^ai 플랫폼$', r'^시스템$', r'^플랫폼$']
    for pattern in generic_patterns:
        if re.match(pattern, service_name):
            return 'OTHER'
    
    # Check for description sentences used as service names
    if len(service_name) > 50 or ' 제공 ' in service_name or ' 지원 ' in service_name:
        return 'OTHER'
    
    # Check for LLM model names (strict patterns)
    model_patterns = [
        r'gpt[-\s]?[\d.]+', r'claude[-\s]?[\d.]+', r'llama[-\s]?[\d.]+', 
        r'qwen[-\s]?[\d.]+', r'gemma[-\s]?[\d.]+', r'mistral[-\s]?[\d.]+',
        r'phi[-\s]?[\d.]+', r'falcon[-\s]?[\d.]+', r'yi[-\s]?[\d.]+',
        r'baichuan[-\s]?[\d.]+', r'internlm[-\s]?[\d.]+',
        r'\d+b', r'\d+\.?\d*', r'oss'
    ]
    for pattern in model_patterns:
        if re.search(pattern, service_name):
            return 'MODEL'
    
    # Check for API names (strict patterns)
    api_keywords = ['api', 'endpoint', 'interface', 'sdk', 'framework']
    for keyword in api_keywords:
        if keyword in service_name and '서비스' not in service_name and '시스템' not in service_name:
            return 'API'
    
    # Check for project/business names (strict patterns)
    project_keywords = ['사업', '지원사업', 'r&d', '실증', '개발', '연구', '과제', '프로젝트']
    for keyword in project_keywords:
        if keyword in service_name and '서비스' not in service_name and '시스템' not in service_name:
            return 'PROJECT'
    
    # Keep as SERVICE if it has clear service indicators
    service_keywords = ['시스템', '서비스', '지원 서비스', '상담', '행정 서비스', '플랫폼', '관리', '운영', '확인', '예보', '탐지', '분석', '검색', '발굴', '심사', '상담', '번역', '통역', '생성', '합성', '인식', '처리', '제어', '자동화']
    for keyword in service_keywords:
        if keyword in service_name:
            return 'SERVICE'
    
    # Check if it's a service based on purpose description
    if purpose:
        service_purpose_keywords = ['제공', '지원', '서비스', '운영', '관리', '상담', '예보', '탐지', '분석', '검색']
        for keyword in service_purpose_keywords:
            if keyword in purpose:
                return 'SERVICE'
    
    # If no clear indicators, classify as OTHER
    return 'OTHER'


def validate_and_filter_services(services: list[dict]) -> tuple[list[dict], dict]:
    """Validate and filter services to keep only actual AI services with detailed logging.
    
    Args:
        services: List of extracted services.
        
    Returns:
        Tuple of (filtered_services, validation_stats)
    """
    validation_stats = {
        'SERVICE': 0,
        'PROJECT': 0,
        'MODEL': 0,
        'API': 0,
        'OTHER': 0,
        'total': len(services),
        'removed_details': []
    }
    
    filtered_services = []
    
    for service in services:
        service_type = classify_service_type(service)
        service['service_type'] = service_type
        validation_stats[service_type] += 1
        
        # Only keep SERVICE type
        if service_type == 'SERVICE':
            filtered_services.append(service)
        else:
            # Log removal details
            reason = service_type
            if service_type == 'MODEL':
                reason = 'MODEL'
            elif service_type == 'PROJECT':
                reason = 'PROJECT'
            elif service_type == 'API':
                reason = 'API'
            else:
                reason = 'OTHER'
            
            validation_stats['removed_details'].append({
                'service': service['service'],
                'type': service_type,
                'reason': reason
            })
    
    validation_stats['filtered'] = validation_stats['total'] - len(filtered_services)
    
    return filtered_services, validation_stats


def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings using character overlap and sequence matching.
    
    Args:
        str1: First string.
        str2: Second string.
        
    Returns:
        Similarity ratio (0.0 to 1.0).
    """
    str1_lower = str1.lower().replace(' ', '')
    str2_lower = str2.lower().replace(' ', '')
    
    if not str1_lower or not str2_lower:
        return 0.0
    
    # Character overlap ratio
    set1 = set(str1_lower)
    set2 = set(str2_lower)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    char_overlap = intersection / union if union > 0 else 0.0
    
    # Sequence matching (longest common substring)
    from difflib import SequenceMatcher
    seq_matcher = SequenceMatcher(None, str1_lower, str2_lower)
    seq_ratio = seq_matcher.ratio()
    
    # Weighted combination (70% sequence, 30% character overlap)
    return (seq_ratio * 0.7) + (char_overlap * 0.3)


def merge_duplicate_services(services: list[dict]) -> tuple[list[dict], dict]:
    """Merge duplicate services from multiple chunks with 80% similarity logic.
    
    Args:
        services: List of extracted services.
        
    Returns:
        Tuple of (merged_services, merge_stats)
    """
    merged = {}
    merge_stats = {
        'original_count': len(services),
        'merged_count': 0,
        'merge_details': []
    }
    
    for service in services:
        # Normalize for duplicate detection (returns tuple)
        normalized_key, canonical_name = normalize_service_name(service['service'])
        
        # Check for 85% similarity with existing services (moderate threshold)
        found_similar = False
        for existing_key in list(merged.keys()):
            similarity = calculate_similarity(normalized_key, existing_key)
            if similarity >= 0.85:
                # Merge into existing service
                merge_stats['merge_details'].append({
                    'merged_from': service['service'],
                    'merged_to': merged[existing_key]['service'],
                    'similarity': round(similarity, 2)
                })
                normalized_key = existing_key
                found_similar = True
                break
        
        # Additional check: merge AI prefix/suffix variations (aggressive for target cases)
        if not found_similar:
            for existing_key in list(merged.keys()):
                key_without_ai = normalized_key.replace('ai', '').replace('AI', '').replace(' ', '').strip()
                existing_without_ai = existing_key.replace('ai', '').replace('AI', '').replace(' ', '').strip()
                # Only merge if base names are identical AND non-empty
                if key_without_ai == existing_without_ai and key_without_ai and len(key_without_ai) > 3:
                    # Check if one has AI and the other doesn't
                    has_ai_new = 'ai' in normalized_key.lower()
                    has_ai_existing = 'ai' in existing_key.lower()
                    # Only merge if exactly one has AI
                    if has_ai_new != has_ai_existing:
                        # Prefer the AI version as canonical
                        if has_ai_new:
                            merged[normalized_key] = merged.pop(existing_key)
                            normalized_key = normalized_key
                        else:
                            normalized_key = existing_key
                        merge_stats['merge_details'].append({
                            'merged_from': service['service'],
                            'merged_to': merged[normalized_key]['service'],
                            'similarity': 1.0,
                            'reason': 'AI prefix variation'
                        })
                        found_similar = True
                        break
        
        if normalized_key not in merged:
            merged[normalized_key] = {
                'service': canonical_name,  # Use canonical name
                'organizations': set(),
                'roles': set(),
                'l1_categories': set(),
                'l2_categories': set(),
                'chunks': set()
            }
        else:
            merge_stats['merged_count'] += 1
        
        # Merge organizations with filtering (exclude non-operating agencies)
        if service.get('organizations'):
            for org in service['organizations']:
                # Filter out participating/cooperation/supporting agencies
                exclude_terms = ['협력', '참여', '수행', '지원기관', '협력기관', '관제센터', '연구소', '개발', '주민센터', '공무원', '상담원']
                is_excluded = any(term in org for term in exclude_terms)
                if not is_excluded:
                    merged[normalized_key]['organizations'].add(org)
        
        # Merge roles (keep all unique roles)
        if service.get('role'):
            merged[normalized_key]['roles'].add(service['role'])
        
        # Merge L1 categories (if different, keep the most specific one)
        if service.get('l1_category'):
            merged[normalized_key]['l1_categories'].add(service['l1_category'])
        
        # Merge L2 categories (if different, keep the most specific one)
        if service.get('l2_category'):
            merged[normalized_key]['l2_categories'].add(service['l2_category'])
        
        # Merge chunks (keep all chunk references)
        if service.get('chunks'):
            for chunk in service['chunks']:
                merged[normalized_key]['chunks'].add(chunk)
    
    # Convert sets to lists and select most specific categories
    result = []
    for key, data in merged.items():
        # Select most specific role (longest)
        roles = list(data['roles'])
        selected_role = max(roles, key=len) if roles else 'AI 기반 서비스'
        
        # Select L1 category (prefer non-보류)
        l1_categories = list(data['l1_categories'])
        selected_l1 = '보류'
        for l1 in l1_categories:
            if l1 != '보류':
                selected_l1 = l1
                break
        
        # Select L2 category (prefer non-보류)
        l2_categories = list(data['l2_categories'])
        selected_l2 = '보류'
        for l2 in l2_categories:
            if l2 != '보류':
                selected_l2 = l2
                break
        
        result.append({
            'service': data['service'],
            'organizations': list(data['organizations']),
            'role': selected_role,
            'l1_category': selected_l1,
            'l2_category': selected_l2,
            'chunks': list(data['chunks'])
        })
    
    return result, merge_stats


def generate_report_from_extraction(merged_services: list[dict], topic: str, search_results: list[dict]) -> str:
    """Generate report based on extracted services table.
    
    Args:
        merged_services: Merged extracted services.
        topic: Research topic.
        search_results: Web search results for supplementary info.
        
    Returns:
        Generated report.
    """
    report = f"# {topic}\n\n"
    
    # Executive Summary
    report += "## Executive Summary\n\n"
    report += f"총 {len(merged_services)}개의 AI 서비스가 식별되었습니다. "
    report += "공공기관에서 다양한 분야에 걸쳐 AI 기술을 도입하여 서비스 품질을 향상하고 운영 효율성을 높이고 있습니다.\n\n"
    
    # Key Content - by organization
    report += "## 핵심 내용\n\n"
    
    # Group by organization
    org_groups = {}
    for service in merged_services:
        for org in service['organizations']:
            if org not in org_groups:
                org_groups[org] = []
            org_groups[org].append(service)
    
    for org, services in org_groups.items():
        report += f"### {org}\n\n"
        for service in services:
            # Use role and categories from merged data
            role = service.get('role', 'AI 기반 서비스')
            l1 = service.get('l1_category', '보류')
            l2 = service.get('l2_category', '보류')
            report += f"- **{service['service']}**: {role} ({l1} > {l2})\n"
        report += "\n"
    
    # Major Cases
    report += "## 주요 사례\n\n"
    for i, service in enumerate(merged_services, 1):
        report += f"{i}. **{service['service']}**\n"
        report += f"   - 운영 기관: {', '.join(service['organizations'])}\n"
        
        # Use role and categories from merged data
        role = service.get('role', 'AI 기반 서비스')
        l1 = service.get('l1_category', '보류')
        l2 = service.get('l2_category', '보류')
        report += f"   - 역할: {role}\n"
        report += f"   - 카테고리: {l1} > {l2}\n"
        
        report += f"   - 출처 청크: {', '.join(service['chunks'])}\n\n"
    
    # Insights
    report += "## 시사점\n\n"
    if len(merged_services) > 0:
        report += "- 공공기관이 다양한 분야에 AI를 적용하고 있음\n"
        report += "- 서비스 품질 향상과 운영 효율성 개선이 주요 목적\n"
        report += "- 여러 부처에서 AI 도입이 확산되는 추세\n"
        report += f"- 총 {len(org_groups)}개 기관에서 {len(merged_services)}개 서비스 운영\n\n"
    
    # References
    report += "## References\n\n"
    report += "### Local PDF\n"
    report += "- data/AI 정부 서비스 사례집_배포용.pdf\n\n"
    
    if search_results:
        report += "### Web\n\n"
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "알 수 없음")
            url = result.get("url", "알 수 없음")
            report += f"{i}. [{title}]({url})\n"
    
    return report


def generate_fallback_report(topic: str, search_results: list[dict], local_documents: list[dict], document_chunks: list[dict]) -> str:
    """Generate fallback report without Ollama, using improved chunk selection.
    
    Args:
        topic: Research topic.
        search_results: Search results from Tavily MCP.
        local_documents: Local documents metadata.
        document_chunks: Chunked content.
        
    Returns:
        Simple Markdown report.
    """
    report = f"# {topic}\n\n"
    
    # Select relevant chunks using the same scoring algorithm
    selected_chunks = select_relevant_chunks(topic, document_chunks, top_k=15)
    
    # Summary section
    report += "## Executive Summary\n\n"
    total_sources = len(search_results) + len(local_documents)
    report += f"총 {total_sources}개의 자료를 수집했습니다 (웹 검색: {len(search_results)}, 로컬 문서: {len(local_documents)}, 총 청크: {len(document_chunks)}).\n\n"
    
    # Key content section (from selected chunks)
    report += "## 핵심 내용\n\n"
    if selected_chunks:
        for chunk in selected_chunks:
            content = chunk.get("content", "")[:500]
            report += f"### Chunk {chunk['chunk_id']} (from {chunk['source']})\n\n"
            report += f"{content}...\n\n"
    elif document_chunks:
        # Fallback to first chunks if selection failed
        for chunk in document_chunks[:3]:
            content = chunk.get("content", "")[:300]
            report += f"- {content}...\n\n"
    
    # Major cases section
    report += "## 주요 사례\n\n"
    if local_documents:
        for doc in local_documents:
            path = doc.get("path", "알 수 없음")
            report += f"- {path}\n\n"
    
    # Insights section
    report += "## 시사점\n\n"
    report += "자료를 기반으로 한 분석이 필요합니다.\n\n"
    
    # Local references section
    if local_documents:
        report += "## 로컬 참고자료\n\n"
        for doc in local_documents:
            path = doc.get("path", "알 수 없음")
            doc_type = doc.get("type", "알 수 없음")
            report += f"- {path} ({doc_type})\n"
        report += "\n"
    
    # Used chunks section
    if selected_chunks:
        report += "## 사용된 PDF 청크\n\n"
        for chunk in selected_chunks:
            report += f"- Chunk #{chunk['chunk_id']} (from {chunk['source']})\n"
        report += "\n"
    
    # Web references section
    if search_results:
        report += "## 웹 참고자료\n\n"
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "알 수 없음")
            url = result.get("url", "알 수 없음")
            report += f"{i}. [{title}]({url})\n"
    
    return report
