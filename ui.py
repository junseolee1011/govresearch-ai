"""Simple Streamlit UI for PoC."""

import streamlit as st
from app.workflow import run_research
from config.settings import get_settings


def main():
    """Main Streamlit app."""
    st.title("GovResearch-AI PoC")
    st.markdown("간단한 연구 보고서 생성 데모")
    
    # Question input
    topic = st.text_input("연구 주제를 입력하세요:", placeholder="예: 공공기관 AI 서비스 도입 사례")
    
    # Run button
    if st.button("실행"):
        if not topic:
            st.warning("연구 주제를 입력해주세요.")
            return
        
        # Show loading state
        with st.spinner("연구 중..."):
            try:
                settings = get_settings()
                result = run_research(topic, settings)
                report = result.get("report", "")
                
                # Display Markdown output
                st.markdown(report)
            except Exception as e:
                st.error(f"오류 발생: {e}")


if __name__ == "__main__":
    main()
