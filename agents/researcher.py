"""Research agent with Tavily web search integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.state import ResearchState
from config.settings import Settings
from tools.ollama_client import get_llm
from tools.prompt_loader import load_prompt
from tools.tavily_search import TavilySearchError, search_web

if TYPE_CHECKING:
    from config.settings import Settings as SettingsType



LOGGER = logging.getLogger(__name__)


def conduct_research(state: ResearchState, settings: Settings) -> dict[str, object]:
    """Perform web searches using Tavily based on the research plan.

    Args:
        state: Workflow state containing topic and research plan.
        settings: Application settings with Tavily configuration.

    Returns:
        State update containing search queries, results, and structured data.
    """
    _ = load_prompt("researcher.txt")
    topic = state["topic"]
    plan = state["plan"]
    
    # Initialize LLM client for service name extraction (with fallback)
    try:
        llm = get_llm(settings)
    except Exception as e:
        LOGGER.warning(f"Failed to initialize LLM for service name extraction: {e}. Using fallback only.")
        llm = None

    # Generate 3-5 search queries based on the research plan
    search_queries = generate_search_queries(topic, plan)
    LOGGER.info("Generated %d search queries for topic: %s", len(search_queries), topic)
    for i, query in enumerate(search_queries, 1):
        LOGGER.debug("Query %d: %s", i, query)

    # Perform web searches for each query
    all_results = []
    for query in search_queries:
        try:
            LOGGER.info("Starting Tavily search for query: %s", query)
            search_response = search_web(query, settings)
            all_results.extend(search_response["results"])
            LOGGER.info("Tavily search completed for query: %s, results: %d", query, len(search_response["results"]))
        except TavilySearchError as e:
            LOGGER.error("Tavily search failed for query '%s': %s", query, str(e))
            # Continue with remaining queries as per requirements
            continue

    # Remove duplicate URLs
    seen_urls = set()
    unique_results = []
    for result in all_results:
        url = result["url"]
        if url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)

    duplicates_removed = len(all_results) - len(unique_results)
    LOGGER.info("Removed %d duplicate URLs, %d unique results remain", duplicates_removed, len(unique_results))

    # Log source breakdown
    source_breakdown = {}
    for result in unique_results:
        source_type = result["source_type"]
        source_breakdown[source_type] = source_breakdown.get(source_type, 0) + 1
    LOGGER.info("Source breakdown: %s", source_breakdown)

    # Convert search results to legacy sources format for compatibility
    sources = convert_to_sources_format(unique_results, llm)

    # Generate findings from search results
    findings = generate_findings_from_results(unique_results, topic)

    LOGGER.info(
        "Researcher completed: %d search queries, %d unique results, %d sources, %d findings",
        len(search_queries),
        len(unique_results),
        len(sources),
        len(findings),
    )

    return {
        "search_queries": search_queries,
        "search_results": unique_results,
        "sources": sources,
        "findings": findings,
    }


def generate_search_queries(topic: str, plan: list[str]) -> list[str]:
    """Generate 3-5 search queries based on research plan.

    Args:
        topic: Research topic.
        plan: Research questions from planner.

    Returns:
        List of search query strings.
    """
    queries = [topic]  # Start with the main topic

    # Add queries based on research plan items
    for i, question in enumerate(plan[:2]):  # Take first 2 questions to avoid too many queries
        # Extract key terms from the question
        if "AI" in question or "인공지능" in question:
            query = f"{topic} 구축 사례"
            if query not in queries:
                queries.append(query)
        if "서비스" in question or "service" in question.lower():
            query = f"{topic} 실제 서비스 구현"
            if query not in queries:
                queries.append(query)
        if "성숙도" in question or "maturity" in question.lower():
            query = f"{topic} 운영 현황"
            if query not in queries:
                queries.append(query)

    # Add specific queries for actual service implementations
    specific_queries = [
        f"{topic} 챗봇 구축 사례",
        f"{topic} AI 서비스 도입 성공 사례"
    ]
    
    for query in specific_queries:
        if query not in queries and len(queries) < 5:
            queries.append(query)

    # Ensure we have 3-5 unique queries
    while len(queries) < 3:
        query = f"{topic} 실제 적용 사례"
        if query not in queries:
            queries.append(query)
        else:
            queries.append(f"{topic} 구현 현황")

    return queries[:5]  # Limit to 5 queries


def clean_text(text: str) -> str:
    """Clean text by removing markdown formatting and special characters.
    
    Args:
        text: Raw text.
        
    Returns:
        Cleaned text.
    """
    import re
    
    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove markdown bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep Korean and basic punctuation
    text = re.sub(r'[^\w\s\.,?!-:;()가-힣]', '', text)
    
    return text.strip()


def extract_service_name_llm(title: str, content: str, llm) -> str:
    """Extract actual AI service name from content using LLM.
    
    Args:
        title: Full title.
        content: Content that may contain actual service names.
        llm: Ollama LLM instance for inference.
        
    Returns:
        Extracted service name.
    """
    if not content:
        return extract_service_name_fallback(title)
    
    # Use LLM to extract actual service names
    prompt = f"""다음 텍스트에서 공공기관이 실제로 구축한 AI 서비스의 구체적인 이름 하나만 추출하세요.
예: "민원상담 AI 어시스턴트", "챗경북", "RAG 플랫폼" 같은 구체적인 서비스 이름만 추출하세요.
일반적인 용어(공공, 정부, AI, 생성형, 챗봇 등)나 문장 조각은 제외하세요.
여러 서비스 이름이 있으면 가장 처음에 나오는 하나만 추출하세요.

제목: {title}
내용: {content[:300]}

서비스 이름 하나만 출력하세요. 찾을 수 없으면 "알 수 없음"이라고 출력하세요."""
    
    try:
        response = llm.invoke(prompt)
        service_name = response.strip()
        
        # Clean and validate the response
        if service_name and service_name not in ['알 수 없음', '공공', '정부', '국내', '해외', '생성형', 'AI', '인공지능', '챗봇']:
            # Further clean the response
            cleaned = clean_text(service_name)
            # Remove any commas or separators that might indicate multiple services
            cleaned = cleaned.split(',')[0].split('、')[0].strip()
            if len(cleaned) >= 2 and len(cleaned) <= 20:
                return cleaned
    except Exception as e:
        LOGGER.warning(f"LLM service name extraction failed: {e}")
    
    return extract_service_name_fallback(title)


def extract_service_name_fallback(title: str) -> str:
    """Fallback service name extraction using title only.
    
    Args:
        title: Full title.
        
    Returns:
        Cleaned service name.
    """
    import re
    
    # Remove common suffixes/prefixes that are not part of the service name
    patterns_to_remove = [
        r'\s*-\s*[가-힣A-Za-z\s]+$',  # Remove trailing "- Publisher Name"
        r'\s*\.\.\.$',  # Remove trailing "..."
        r'\s*블로그$',  # Remove "블로그"
        r'\s*뉴스$',  # Remove "뉴스"
        r'\s*보도$',  # Remove "보도"
        r'\s*기사$',  # Remove "기사"
        r'\s*리포트$',  # Remove "리포트"
        r'\s*연구$',  # Remove "연구"
        r'\s*보고서$',  # Remove "보고서"
        r'\s*현황$',  # Remove "현황"
        r'\s*분석$',  # Remove "분석"
        r'\s*구축$',  # Remove "구축"
        r'\s*사례$',  # Remove "사례"
    ]
    
    cleaned = title
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned)
    
    # If title is too long, try to extract the main part
    if len(cleaned) > 50:
        # Look for common patterns and extract the meaningful part
        # Remove everything after common separators
        for separator in [' - ', ' | ', ' : ', ' > ']:
            if separator in cleaned:
                parts = cleaned.split(separator)
                # Take the first part as it's usually the main title
                cleaned = parts[0]
                break
    
    # Apply general cleaning
    cleaned = clean_text(cleaned)
    
    # Limit length
    if len(cleaned) > 40:
        cleaned = cleaned[:40] + "..."
    
    return cleaned if cleaned else title


def convert_to_sources_format(search_results: list[dict], llm) -> list[dict[str, str]]:
    """Convert search results to legacy sources format.

    Args:
        search_results: Normalized search results from Tavily.
        llm: Ollama client for LLM-based service name extraction (can be None).

    Returns:
        List of source records in legacy format.
    """
    sources = []
    for result in search_results:
        # Extract more specific information from the search result
        title = result["title"]
        content = result["content"]
        url = result["url"]
        source_type = result["source_type"]
        
        # Try to extract more specific service type from content
        service_type = extract_service_type(title, content)
        
        # Try to extract use case from content
        use_case = extract_use_case(content)
        
        # Try to extract maturity level from content
        maturity = extract_maturity(content)
        
        # Use LLM-based extraction for actual service names if available, otherwise fallback
        if llm:
            service_name = extract_service_name_llm(title, content, llm)
        else:
            service_name = extract_service_name_fallback(title)
        
        source = {
            "title": service_name,
            "url": url,
            "institution": extract_institution(title, content),
            "service_type": service_type,
            "use_case": use_case,
            "maturity": maturity,
        }
        sources.append(source)
    return sources


def extract_institution(title: str, content: str = "") -> str:
    """Extract institution name from title and content.

    Args:
        title: Source title.
        content: Source content.

    Returns:
        Institution name or default.
    """
    # Try to extract from title first
    text = title + " " + content
    
    # Known Korean public institutions (exact matches)
    known_institutions = [
        "국회도서관", "국민연금공단", "한국수력원자력", "기획재정부", 
        "행정안전부", "과학기술정보통신부", "보건복지부", "교육부",
        "한국지능정보사회진흥원", "한국IDC", "SPRi", "소프트웨어정책연구소",
        "KISDI", "한국일보", "위키백과", "CIO", "산업종합저널",
        "한국정책브리핑", "정책브리핑", "국가전략포털", "NIA",
        "한국지역정보개발원", "한국정보화진흥원", "남부발전", "한국전력거래소",
        "AI매터스", "폴라리스오피스", "코리아딥", "인더스트리저널"
    ]
    
    for institution in known_institutions:
        if institution in text:
            return institution
    
    # Try to extract from URL patterns or common patterns (more specific)
    import re
    
    # Look for specific government entity patterns with minimum length
    gov_patterns = [
        r"([가-힣]{2,}청)", r"([가-힣]{2,}부)", r"([가-힣]{2,}도청)", r"([가-힣]{2,}시청)", 
        r"([가-힣]{2,}군청)", r"([가-힣]{2,}구청)", r"([가-힣]{2,}공사)", r"([가-힣]{2,}공단)",
        r"([가-힣]{2,}원)", r"([가-힣]{2,}센터)", r"([가-힣]{2,}연구소)", r"([가-힣]{2,}대학)",
        r"([가-힣]{2,}발전)", r"([가-힣]{2,}전력)", r"([가-힣]{2,}진흥원)"
    ]
    
    # Filter out common non-institution words
    exclude_words = [
        "수시", "예도", "무엇", "어떤", "이것", "저것", "그것", "방법", "방향", "현황", "사례", "분석", "연구",
        "제안요청", "공공부", "남부", "폴라리스", "오피스", "블로그", "인스타그램", "클릭", "스마트",
        "국내", "해외", "글로벌", "세계", "전체", "전체글", "목록", "카테고리", "보기", "상세", "다운로드",
        "인공지능", "생성형", "서비스", "도입", "구축", "사례", "현황", "분석", "연구", "보고서"
    ]
    
    for pattern in gov_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Filter out excluded words and return the longest valid match
            valid_matches = [m for m in matches if m not in exclude_words and len(m) >= 2]
            if valid_matches:
                return max(valid_matches, key=len)
    
    # Try to extract from common media/news sources
    media_patterns = [
        r"([가-힣]{2,}일보)", r"([가-힣]{2,}신문)", r"([가-힣]{2,}뉴스)", r"([가-힣]{2,}타임즈)",
        r"([가-힣]{2,}저널)", r"([가-힣]{2,}매거진)", r"([가-힣]{2,}리포트)"
    ]
    
    for pattern in media_patterns:
        matches = re.findall(pattern, text)
        if matches:
            valid_matches = [m for m in matches if m not in exclude_words and len(m) >= 2]
            if valid_matches:
                return max(valid_matches, key=len)
    
    # If still no match, try to return a generic but meaningful name based on content
    if "공공" in text:
        return "공공기관"
    if "정부" in text:
        return "정부기관"
    if "연구" in text:
        return "연구기관"
    
    return "알 수 없음"


def extract_service_type(title: str, content: str) -> str:
    """Extract service type from title and content.

    Args:
        title: Source title.
        content: Source content.

    Returns:
        Service type classification.
    """
    text = (title + " " + content).lower()
    
    service_keywords = {
        "챗봇": ["챗봇", "chatbot", "대화형", "상담"],
        "문서 처리": ["문서", "document", "분류", "triage", "자동화"],
        "분석": ["분석", "analysis", "데이터", "data", "인사이트", "insight"],
        "번역": ["번역", "translation", "통번역"],
        "생성": ["생성", "generation", "작성", "writing"],
    }
    
    for service_type, keywords in service_keywords.items():
        for keyword in keywords:
            if keyword in text:
                return service_type
    
    return "AI 서비스"


def clean_text(text: str) -> str:
    """Clean text by removing markdown formatting and special characters.
    
    Args:
        text: Raw text.
        
    Returns:
        Cleaned text.
    """
    import re
    
    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove markdown bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep Korean and basic punctuation
    text = re.sub(r'[^\w\s\.,?!-:;()가-힣]', '', text)
    
    return text.strip()


def extract_use_case(content: str) -> str:
    """Extract use case from content.

    Args:
        content: Source content.

    Returns:
        Use case description.
    """
    if not content:
        return "알 수 없음"
    
    # Clean the content first
    cleaned = clean_text(content)
    
    # Return first 100 characters as use case
    return cleaned[:100] if len(cleaned) > 100 else cleaned


def extract_maturity(content: str) -> str:
    """Extract maturity level from content.

    Args:
        content: Source content.

    Returns:
        Maturity level classification.
    """
    if not content:
        return "알 수 없음"
    
    text = content.lower()
    
    maturity_keywords = {
        "운영 중": ["운영", "operation", "서비스", "service", "배포", "deployed"],
        "파일럿": ["파일럿", "pilot", "시범", "test"],
        "프로토타입": ["프로토타입", "prototype", "개발", "development"],
    }
    
    for maturity, keywords in maturity_keywords.items():
        for keyword in keywords:
            if keyword in text:
                return maturity
    
    return "알 수 없음"


def generate_findings_from_results(search_results: list[dict], topic: str) -> list[str]:
    """Generate findings from search results.

    Args:
        search_results: Normalized search results from Tavily.
        topic: Research topic.

    Returns:
        List of finding statements.
    """
    findings = [
        f"{topic}에 대한 웹 검색을 통해 {len(search_results)}개의 관련 결과를 수집했습니다.",
    ]

    # Add source type findings
    source_types = set(result["source_type"] for result in search_results)
    if source_types:
        findings.append(f"수집된 출처 유형: {', '.join(source_types)}")

    # Add content-based findings
    if search_results:
        findings.append("검색 결과는 다양한 공공기관의 AI 서비스 도입 사례와 현황을 포함하고 있습니다.")

    return findings
