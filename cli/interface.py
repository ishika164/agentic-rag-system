from __future__ import annotations

import logging
import sys
from pathlib import Path

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    _COLOR = True
except ImportError:
    _COLOR = False
    class _Dummy:
        def __getattr__(self, _):
            return ""
    Fore = Style = _Dummy()

from agent.orchestrator import AgenticRAG
from rag.ingestion import DocumentIngestor

logger = logging.getLogger(__name__)

BANNER = """
╔═══════════════════════════════════════════════╗
║         Agentic RAG System  •  v1.0           ║
║  Type /help for commands  •  /quit to exit    ║
╚═══════════════════════════════════════════════╝
"""

HELP_TEXT = """
Available commands:
  /ingest <path>   Ingest a file or directory
  /status          Show number of stored chunks
  /reset           Clear conversation memory
  /help            Show this message
  /quit | /exit    Exit
"""


def _header(text: str) -> str:
    width = 52
    line = "─" * width
    return f"{Fore.CYAN}┌─ {text} {line[:width - len(text) - 3]}{Style.RESET_ALL}"


def _row(label: str, value: str) -> str:
    return f"{Fore.CYAN}│{Style.RESET_ALL}  {Fore.YELLOW}{label:<12}{Style.RESET_ALL}: {value}"


def _footer() -> str:
    return f"{Fore.CYAN}└{'─' * 53}{Style.RESET_ALL}"


def _print_response(resp) -> None:
    sources_str = ", ".join(Path(s).name for s in resp.sources) if resp.sources else "—"
    retrieval_str = (
        f"{Fore.GREEN}Yes{Style.RESET_ALL}" if resp.retrieval_used
        else f"{Fore.MAGENTA}No{Style.RESET_ALL}"
    )
    routing_color = Fore.GREEN if resp.routing_decision == "RAG" else Fore.MAGENTA

    print()
    print(_header("Answer"))
    for line in resp.answer.strip().splitlines():
        print(f"{Fore.CYAN}│{Style.RESET_ALL}  {line}")
    print(_header("Metadata"))
    print(_row("Routing", f"{routing_color}{resp.routing_decision}{Style.RESET_ALL}"))
    print(_row("Retrieval", retrieval_str))
    print(_row("Sources", sources_str))
    print(_footer())
    print()


def _handle_ingest(args: str, ingestor: DocumentIngestor) -> None:
    path = Path(args.strip())
    if not path.exists():
        print(f"{Fore.RED}Error: path not found — {path}{Style.RESET_ALL}")
        return
    try:
        if path.is_dir():
            n = ingestor.ingest_directory(path)
        else:
            n = ingestor.ingest_file(path)
        print(f"{Fore.GREEN}✓ Ingested {n} chunk(s) from '{path.name}'{Style.RESET_ALL}")
    except Exception as exc:
        print(f"{Fore.RED}Ingestion failed: {exc}{Style.RESET_ALL}")


def _handle_status(ingestor: DocumentIngestor) -> None:
    count = ingestor.collection_size()
    print(f"{Fore.CYAN}Knowledge base: {count} chunk(s) stored.{Style.RESET_ALL}")


def run_cli(agent: AgenticRAG, ingestor: DocumentIngestor) -> None:
    print(Fore.CYAN + BANNER + Style.RESET_ALL)

    while True:
        try:
            raw = input(f"{Fore.YELLOW}You ▶{Style.RESET_ALL} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not raw:
            continue

        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                print("Goodbye!")
                sys.exit(0)
            elif cmd == "/help":
                print(HELP_TEXT)
            elif cmd == "/reset":
                agent.reset_memory()
                print(f"{Fore.GREEN}✓ Memory cleared.{Style.RESET_ALL}")
            elif cmd == "/status":
                _handle_status(ingestor)
            elif cmd == "/ingest":
                if not args:
                    print(f"{Fore.RED}Usage: /ingest <path>{Style.RESET_ALL}")
                else:
                    _handle_ingest(args, ingestor)
            else:
                print(f"{Fore.RED}Unknown command. Type /help.{Style.RESET_ALL}")
            continue

        try:
            resp = agent.ask(raw)
            _print_response(resp)
        except Exception as exc:
            logger.exception("Error processing query")
            print(f"{Fore.RED}Error: {exc}{Style.RESET_ALL}")