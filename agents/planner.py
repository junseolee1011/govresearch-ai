"""Planner agent for task decomposition using Ollama with structured JSON output."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from config.settings import Settings
from config.service_constants import SERVICE_SEARCH_QUERIES
from models.schemas import ResearchPlan, SubTask
from tools.logger import log_node_execution
from tools.ollama_client import get_llm


LOGGER = logging.getLogger(__name__)


# Few-shot decomposition examples
FEW_SHOT_EXAMPLES = """
Example 1:
Topic: "AI services used by public institutions"
Output:
{
  "main_goal": "Identify and analyze AI services currently deployed across various public sector institutions",
  "sub_tasks": [
    {
      "description": "Search for government reports and case studies on AI implementation in public sector",
      "priority": "high",
      "dependencies": [],
      "estimated_complexity": "moderate"
    },
    {
      "description": "Identify specific AI services by department (healthcare, transportation, administration)",
      "priority": "high",
      "dependencies": ["Search for government reports and case studies on AI implementation in public sector"],
      "estimated_complexity": "complex"
    },
    {
      "description": "Analyze the impact and effectiveness of identified AI services",
      "priority": "medium",
      "dependencies": ["Identify specific AI services by department (healthcare, transportation, administration)"],
      "estimated_complexity": "moderate"
    },
    {
      "description": "Compile findings into a comprehensive report with recommendations",
      "priority": "medium",
      "dependencies": ["Analyze the impact and effectiveness of identified AI services"],
      "estimated_complexity": "simple"
    }
  ],
  "total_estimated_steps": 8
}

Example 2:
Topic: "Climate change adaptation strategies for coastal cities"
Output:
{
  "main_goal": "Research and document adaptation strategies that coastal cities are implementing to address climate change",
  "sub_tasks": [
    {
      "description": "Identify major coastal cities vulnerable to climate change",
      "priority": "high",
      "dependencies": [],
      "estimated_complexity": "simple"
    },
    {
      "description": "Research sea-level rise projections for identified cities",
      "priority": "high",
      "dependencies": ["Identify major coastal cities vulnerable to climate change"],
      "estimated_complexity": "moderate"
    },
    {
      "description": "Document infrastructure adaptation measures (seawalls, elevated buildings)",
      "priority": "high",
      "dependencies": ["Research sea-level rise projections for identified cities"],
      "estimated_complexity": "complex"
    },
    {
      "description": "Analyze policy and governance frameworks for adaptation",
      "priority": "medium",
      "dependencies": ["Document infrastructure adaptation measures (seawalls, elevated buildings)"],
      "estimated_complexity": "moderate"
    },
    {
      "description": "Evaluate economic costs and funding sources for adaptation projects",
      "priority": "medium",
      "dependencies": ["Analyze policy and governance frameworks for adaptation"],
      "estimated_complexity": "complex"
    }
  ],
  "total_estimated_steps": 12
}
"""


def plan_research(state: ResearchState, settings: Optional[Settings] = None) -> dict[str, list[str]]:
    """Generate structured research plan with task decomposition using schema-constrained prompting.

    Args:
        state: Current workflow state containing a research topic.
        settings: Application settings for LLM configuration.

    Returns:
        State update containing search queries derived from the research plan.
    """
    topic = state["topic"]
    
    if settings is None:
        # Fallback to simple keyword generation if no settings provided
        LOGGER.warning("No settings provided to planner, using simple keyword generation")
        keywords = [topic]
        LOGGER.info("Planner generated search keywords: %s", keywords)
        return {"search_queries": keywords}
    
    # Implement verify-and-reprompt loop for sufficient decomposition depth
    max_retries = 3
    research_plan = None
    
    for attempt in range(max_retries):
        try:
            research_plan = generate_structured_plan(topic, settings, attempt)
            
            # Verify decomposition depth
            if len(research_plan.sub_tasks) >= 3 and research_plan.total_estimated_steps >= len(research_plan.sub_tasks):
                LOGGER.info("Research plan validation passed on attempt %d", attempt + 1)
                break
            else:
                LOGGER.warning(
                    "Insufficient decomposition depth on attempt %d: %d sub-tasks, %d steps",
                    attempt + 1, len(research_plan.sub_tasks), research_plan.total_estimated_steps
                )
                if attempt < max_retries - 1:
                    continue
                    
        except Exception as e:
            LOGGER.error("Failed to generate valid research plan on attempt %d: %s", attempt + 1, str(e))
            if attempt < max_retries - 1:
                continue
    
    if research_plan is None:
        LOGGER.error("Failed to generate valid research plan after %d attempts, using fallback", max_retries)
        keywords = [topic]
        return {"search_queries": keywords}
    
    # Extract search queries from the research plan
    search_queries = extract_search_queries_from_plan(
        research_plan, topic, max_queries=settings.max_search_queries
    )
    
    LOGGER.info(
        "Planner generated structured plan with %d sub-tasks and %d search queries",
        len(research_plan.sub_tasks),
        len(search_queries)
    )
    
    return {"search_queries": search_queries}


def generate_structured_plan(topic: str, settings: Settings, attempt: int) -> ResearchPlan:
    """Generate structured research plan using LLM with schema validation.
    
    Args:
        topic: Research topic.
        settings: Application settings.
        attempt: Current attempt number (for prompt adjustment).
        
    Returns:
        Validated ResearchPlan object.
        
    Raises:
        ValueError: If generated plan fails schema validation.
    """
    LOGGER.info("Attempting to generate structured plan (attempt %d) for topic: %s", attempt + 1, topic)
    llm = get_llm(settings)
    LOGGER.info("LLM initialized successfully with model: %s", settings.ollama_model)
    
    # Adjust prompt based on attempt number
    depth_instruction = ""
    if attempt > 0:
        depth_instruction = (
            "\nIMPORTANT: Your previous attempt had insufficient decomposition depth. "
            "Please ensure you generate AT LEAST 3-5 sub-tasks with clear dependencies. "
            "Break down complex tasks into smaller, manageable components."
        )
    
    prompt = f"""You are a research planning expert. Your task is to decompose a research topic into a structured plan with specific sub-tasks.

Research Topic: {topic}

{depth_instruction}

Your output must be a valid JSON object with the following structure:
{{
  "main_goal": "Clear statement of the primary research goal",
  "sub_tasks": [
    {{
      "description": "Detailed description of what this sub-task accomplishes",
      "priority": "high|medium|low",
      "dependencies": ["List of sub-task descriptions that must be completed first"],
      "estimated_complexity": "simple|moderate|complex"
    }}
  ],
  "total_estimated_steps": "Total number of estimated steps across all sub-tasks"
}}

Requirements:
- Generate at least 3-5 sub-tasks for comprehensive coverage
- Each sub-task should have a clear, actionable description
- Prioritize sub-tasks logically (high priority first)
- Identify dependencies between sub-tasks
- Estimate complexity realistically
- Total steps should be at least equal to the number of sub-tasks

{FEW_SHOT_EXAMPLES}

Now generate a research plan for the topic: {topic}

Output only valid JSON, no additional text:"""

    try:
        LOGGER.info("Invoking LLM with prompt...")
        response = llm.invoke(prompt)
        LOGGER.info("LLM response received. Length: %d characters", len(response))
        LOGGER.debug("LLM response preview: %s", response[:200])
        
        # Parse JSON response
        try:
            # Extract JSON from response if there's extra text
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                LOGGER.info("Extracted JSON from response (positions %d to %d)", json_start, json_end)
            else:
                json_str = response
                LOGGER.warning("Could not find JSON delimiters, using full response")
            
            plan_dict = json.loads(json_str)
            LOGGER.info("JSON parsed successfully. Keys: %s", list(plan_dict.keys()))
            
            # Validate against Pydantic schema
            research_plan = ResearchPlan(**plan_dict)
            LOGGER.info("Successfully generated and validated research plan with %d sub-tasks", len(research_plan.sub_tasks))
            return research_plan
            
        except json.JSONDecodeError as e:
            LOGGER.error("Failed to parse JSON response: %s", str(e))
            LOGGER.error("Response content: %s", response)
            raise ValueError(f"Invalid JSON response: {e}")
        except Exception as e:
            LOGGER.error("Schema validation failed: %s", str(e))
            LOGGER.error("Plan dict: %s", plan_dict if 'plan_dict' in locals() else "N/A")
            raise ValueError(f"Schema validation failed: {e}")
            
    except Exception as e:
        LOGGER.error("LLM invocation failed: %s", str(e), exc_info=True)
        raise


def extract_search_queries_from_plan(
    research_plan: ResearchPlan,
    original_topic: str,
    max_queries: int = 70,
) -> list[str]:
    """Extract search queries from the structured research plan.
    
    Args:
        research_plan: Validated research plan.
        original_topic: Original research topic for language detection.
        max_queries: Upper bound on queries for Korean public-sector AI topics
            (the main goal plus the predefined service-specific queries).
        
    Returns:
        List of search queries derived from sub-tasks.
    """
    queries = []
    
    # Detect if topic is about Korean public institution AI services
    # If so, use actual service names for better search results
    korean_ai_keywords = ['공공기관', 'AI 서비스', '정부', '한국', '공공']
    is_korean_public_ai = any(keyword in original_topic for keyword in korean_ai_keywords)
    
    if is_korean_public_ai:
        # Use predefined service-specific search queries for better results
        LOGGER.info("[Planner] Using service-specific search queries for Korean public AI services")
        queries = SERVICE_SEARCH_QUERIES.copy()
        
        # Add the main goal as a general query
        queries.insert(0, research_plan.main_goal)
        
        # Limit the number of queries (configurable via MAX_SEARCH_QUERIES)
        if len(queries) > max_queries:
            queries = queries[:max_queries]
            
        return queries
    
    # Original logic for other topics
    # Detect if topic is in Korean
    is_korean_topic = bool(any('\u3131' <= char <= '\u3163' or '\uAC00' <= char <= '\uD7A3' for char in original_topic))
    
    # Extract main goal as a query
    main_goal = research_plan.main_goal
    if is_korean_topic and not any('\u3131' <= char <= '\u3163' or '\uAC00' <= char <= '\uD7A3' for char in main_goal):
        # If topic is Korean but main goal is English, translate key terms to Korean
        main_goal = translate_to_korean_keywords(main_goal)
    queries.append(main_goal)
    
    # Extract high-priority sub-tasks as queries
    for sub_task in research_plan.sub_tasks:
        description = sub_task.description
        if is_korean_topic and not any('\u3131' <= char <= '\u3163' or '\uAC00' <= char <= '\uD7A3' for char in description):
            # If topic is Korean but description is English, translate key terms to Korean
            description = translate_to_korean_keywords(description)
        
        if sub_task.priority == "high":
            queries.append(description)
        elif sub_task.priority == "medium" and len(queries) < 5:
            queries.append(description)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_queries = []
    for query in queries:
        if query not in seen:
            seen.add(query)
            unique_queries.append(query)
    
    return unique_queries


def translate_to_korean_keywords(text: str) -> str:
    """Translate English keywords to Korean for Korean topics.
    
    Args:
        text: English text to translate.
        
    Returns:
        Text with Korean keywords substituted.
    """
    # Simple keyword mapping for common AI/government terms
    keyword_map = {
        "AI services": "AI 서비스",
        "artificial intelligence": "인공지능",
        "public institutions": "공공기관",
        "government": "정부",
        "public sector": "공공 부문",
        "implementation": "도입",
        "deployment": "구축",
        "case studies": "사례",
        "examples": "예시",
        "research": "연구",
        "analysis": "분석",
        "report": "보고서",
        "policy": "정책",
        "strategy": "전략",
        "services": "서비스",
        "identify": "식별",
        "analyze": "분석",
        "search": "검색",
        "find": "찾기",
        "government reports": "정부 보고서",
        "case study": "사례 연구",
        "healthcare": "보건 의료",
        "transportation": "교통",
        "administration": "행정",
        "impact": "영향",
        "effectiveness": "효과성",
        "efficiency": "효율성",
    }
    
    result = text
    for eng, kor in keyword_map.items():
        result = result.replace(eng, kor)
    
    return result

