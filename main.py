from __future__ import annotations

import argparse
import sys

from utils.logging_setup import setup_logging
from config import DOCS_DIR, GROQ_API_KEY
from rag.ingestion import DocumentIngestor
from rag.retriever import DocumentRetriever
from agent.orchestrator import AgenticRAG
from cli.interface import run_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic RAG System — LangChain + Chroma + Groq",
    )
    parser.add_argument(
        "--ingest",
        metavar="PATH",
        help="Path to a file or directory to ingest before starting.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def validate_env() -> None:
    if not GROQ_API_KEY:
        print(
            "ERROR: GROQ_API_KEY is not set.\n"
            "Make sure your .env file has:\n  GROQ_API_KEY=gsk_..."
        )
        sys.exit(1)


def build_pipeline() -> tuple[AgenticRAG, DocumentIngestor]:
    ingestor = DocumentIngestor()
    retriever = DocumentRetriever(ingestor.vector_store)
    agent = AgenticRAG(retriever)
    return agent, ingestor


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    validate_env()

    agent, ingestor = build_pipeline()

    if args.ingest:
        from pathlib import Path
        p = Path(args.ingest)
        if p.is_dir():
            ingestor.ingest_directory(p)
        else:
            ingestor.ingest_file(p)

    elif DOCS_DIR.exists() and ingestor.collection_size() == 0:
        files = list(DOCS_DIR.iterdir())
        if files:
            print(f"Auto-ingesting documents from '{DOCS_DIR}' ...")
            ingestor.ingest_directory(DOCS_DIR)

    run_cli(agent, ingestor)


if __name__ == "__main__":
    main()