"""Command-line entry point for GovResearch-AI."""

from __future__ import annotations

import argparse
import sys

from app.workflow import run_research
from config.settings import get_settings
from tools.logger import configure_logging


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace containing optional topic input.
    """
    parser = argparse.ArgumentParser(description="Run a GovResearch-AI workflow.")
    parser.add_argument("--topic", help="Public-sector AI topic to research.")
    return parser.parse_args()


def main() -> None:
    """Execute the Planner → Research → Report workflow and print its report."""
    # Set UTF-8 encoding for Windows console
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, 'strict')
    
    settings = get_settings()
    configure_logging(settings.log_level)
    arguments = parse_arguments()
    topic = arguments.topic or settings.default_research_topic
    result = run_research(topic, settings)
    print(result["report"])


if __name__ == "__main__":
    main()
