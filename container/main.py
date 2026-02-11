#!/usr/bin/env python3
"""RLM Document Analysis CLI - Analyze long documents using Recursive Language Models."""

import argparse
import os
import sys
from pathlib import Path

import dspy

from pdf_loader import load_pdf
from qa import ProductionRLMQA, SimpleRLMQA


def load_document(path: str) -> str:
    """Load document content based on file extension."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .txt, or .md")


def configure_lm(provider: str, model: str | None = None) -> None:
    """Configure dspy with the appropriate language model."""
    provider = provider.lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        model_name = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
        lm = dspy.LM(f"openai/{model_name}", api_key=api_key)

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        model_name = model or os.getenv("MODEL_NAME", "claude-3-5-sonnet-20241022")
        lm = dspy.LM(f"anthropic/{model_name}", api_key=api_key)

    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        model_name = model or os.getenv("MODEL_NAME", "claude-3-5-sonnet-20241022")
        lm = dspy.LM(f"openrouter/openrouter/{model_name}", api_key=api_key)

    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
        api_key = os.getenv("OLLAMA_API_KEY")
        if not api_key:
            raise ValueError("OLLAMA_API_KEY environment variable not set")
        model_name = model or os.getenv("MODEL_NAME", "gpt-oss:120b-cloud")
        lm = dspy.LM(f"ollama/{model_name}", api_base=base_url, api_key=api_key)
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. Use openai, anthropic, or ollama"
        )

    dspy.configure(lm=lm)
    print(f"Configured {provider} with model: {model_name}")


def detect_provider() -> str:
    """Auto-detect provider based on available environment variables."""
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    elif os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    elif os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_BASE_URL"):
        return "ollama"
    elif os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    else:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or OLLAMA_API_KEY, OPENROUTER_API_KEY"
        )


def run_interactive(qa_module, context: str) -> None:
    """Run interactive Q&A session."""
    print("\n" + "=" * 60)
    print("Interactive RLM Document Analysis")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nAnalyzing...")
        result = qa_module(context=context, question=question)

        print(f"\nAnswer: {result.answer}")
        if hasattr(result, "method"):
            print(f"(Method: {result.method})")
        print()


def run_single_query(qa_module, context: str, question: str) -> None:
    """Run a single query and print the result."""
    print(f"\nQuestion: {question}")
    print("Analyzing...")

    result = qa_module(context=context, question=question)

    print(f"\nAnswer: {result.answer}")
    if hasattr(result, "method"):
        print(f"(Method: {result.method})")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze documents using RLM (Recursive Language Models)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py document.pdf "What is the main argument?"
  python main.py document.pdf --interactive
  python main.py document.txt -i --provider anthropic
  python main.py paper.md "Summarize the findings" --model gpt-4o

Environment Variables:
  OPENAI_API_KEY      OpenAI API key
  ANTHROPIC_API_KEY   Anthropic API key  
  OLLAMA_API_KEY      Ollama Cloud API key
  OLLAMA_BASE_URL     Ollama Cloud URL (default: https://ollama.com/)
  MODEL_NAME          Override default model name
        """,
    )

    parser.add_argument("document", help="Path to document file (.pdf, .txt, or .md)")
    parser.add_argument("query", nargs="?", help="Question to ask about the document")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Run in interactive Q&A mode"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "ollama", "openrouter"],
        help="LLM provider (auto-detected from env vars if not specified)",
    )
    parser.add_argument(
        "--model", help="Model name to use (overrides MODEL_NAME env var)"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use SimpleRLMQA instead of ProductionRLMQA",
    )

    args = parser.parse_args()

    if not args.interactive and not args.query:
        parser.error("Either provide a query or use --interactive mode")

    try:
        print(f"Loading document: {args.document}")
        context = load_document(args.document)
        print(f"Loaded {len(context):,} characters")

        provider = args.provider or detect_provider()
        configure_lm(provider, args.model)

        if args.simple:
            qa_module = SimpleRLMQA()
            print("Using SimpleRLMQA")
        else:
            qa_module = ProductionRLMQA()
            print("Using ProductionRLMQA")

        if args.interactive:
            run_interactive(qa_module, context)
        else:
            run_single_query(qa_module, context, args.query)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
