#!/usr/bin/env python
import argparse
import logging
import json
from typing import Dict, Any

# Import the processor
from src.processors.rdl_qwen_processor import RDLQwenQueryProcessor

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_query(
    query: str,
    endpoint: str = "https://data.posccaesar.org/rdl/sparql",
    quantization: str = "4bit",
    format_type: str = "text",
) -> Dict[str, Any]:
    """
    Process a query using the RDLQwenQueryProcessor.

    Args:
        query: Query string
        endpoint: SPARQL endpoint URL
        quantization: Quantization level for Qwen model
        format_type: Output format (text or json)

    Returns:
        Response dictionary
    """
    try:
        print("Creating processor instance...")
        # Create processor instance
        processor = RDLQwenQueryProcessor(endpoint=endpoint)

        print("Processing query...")
        # Process the query
        response = processor.process_query(query)

        print(f"Response type: {type(response)}")
        response_keys = (
            response.keys() if isinstance(response, dict) else "Not a dictionary"
        )
        print(f"Response keys: {response_keys}")

        return response
    except Exception as e:
        print(f"Debug - Error in process_query: {e}")
        print(f"Error type: {type(e)}")
        import traceback

        traceback.print_exc()
        raise


def process_bare_qwen_query(
    query: str,
    endpoint: str = "https://data.posccaesar.org/rdl/sparql",
    quantization: str = "4bit",
) -> Dict[str, Any]:
    """
    Process a query using only bare Qwen LLM without RAG.

    Args:
        query: Query string
        endpoint: SPARQL endpoint URL (not used but kept for consistency)
        quantization: Quantization level for Qwen model

    Returns:
        Response dictionary
    """
    try:
        print("Creating processor instance for bare Qwen...")
        # Create processor instance
        processor = RDLQwenQueryProcessor(endpoint=endpoint)

        print("Processing query with bare Qwen...")
        # Process the query with bare Qwen only
        response = processor.process_bare_qwen_query(query)

        return response
    except Exception as e:
        print(f"Debug - Error in process_bare_qwen_query: {e}")
        print(f"Error type: {type(e)}")
        import traceback

        traceback.print_exc()
        raise


def process_comparison(
    query: str,
    endpoint: str = "https://data.posccaesar.org/rdl/sparql",
    quantization: str = "4bit",
    format_type: str = "text",
) -> Dict[str, Any]:
    """
    Process a query with both RAG and bare Qwen for comparison.

    Args:
        query: Query string
        endpoint: SPARQL endpoint URL
        quantization: Quantization level for Qwen model
        format_type: Output format (text or json)

    Returns:
        Dictionary with both responses
    """
    print("=" * 80)
    print("COMPARISON MODE: RAG vs Bare Qwen")
    print("=" * 80)

    # Get RAG response
    print("\n1. Processing with RAG system...")
    rag_response = process_query(query, endpoint, quantization, format_type)

    # Get bare Qwen response
    print("\n2. Processing with bare Qwen...")
    bare_response = process_bare_qwen_query(query, endpoint, quantization)

    return {
        "query": query,
        "rag_response": rag_response,
        "bare_qwen_response": bare_response,
    }


def format_response(response: Dict[str, Any], format_type: str = "text") -> str:
    """
    Format the processor response for display.

    Args:
        response: Response dictionary
        format_type: Output format (text or json)

    Returns:
        Formatted response as a string
    """
    if format_type == "json":
        return json.dumps(response, indent=2)

    # Format as text
    lines = []

    # Add question
    question = response.get("question", "")
    lines.append(f"Question: {question}")
    lines.append("-" * 80)

    # Add entity matching details
    context = response.get("context", {})
    print(f"Context: {context}")
    entity = context.get("entity", "")
    matched_term = context.get("matched_term", "")
    confidence = context.get("confidence", 0)

    lines.append("Entity Matching:")
    if matched_term:
        lines.append(f"- '{entity}' → '{matched_term}' (confidence: {confidence:.1f}%)")
    else:
        lines.append(f"- No match found for '{entity}'")

    lines.append("-" * 80)

    # Add LLM answer
    llm_answer = response.get("llm_answer", "")
    lines.append("Answer:")
    lines.append(llm_answer)

    return "\n".join(lines)


def format_comparison(comparison: Dict[str, Any], format_type: str = "text") -> str:
    """
    Format the comparison response for display.

    Args:
        comparison: Comparison dictionary with both responses
        format_type: Output format (text or json)

    Returns:
        Formatted comparison as a string
    """
    if format_type == "json":
        return json.dumps(comparison, indent=2)

    # Format as text comparison
    lines = []

    query = comparison.get("query", "")
    rag_response = comparison.get("rag_response", {})
    bare_response = comparison.get("bare_qwen_response", {})

    lines.append("=" * 80)
    lines.append(f"QUERY COMPARISON: {query}")
    lines.append("=" * 80)

    # RAG Response
    lines.append("\n🔍 RAG SYSTEM RESPONSE:")
    lines.append("-" * 50)

    # Add entity matching for RAG
    context = rag_response.get("context", {})
    entity = context.get("entity", "")
    matched_term = context.get("matched_term", "")
    confidence = context.get("confidence", 0)
    query_type = context.get("query_type", "")

    if matched_term:
        lines.append(f"Entity Match: '{entity}' → '{matched_term}' ({confidence:.1f}%)")
        lines.append(f"Query Type: {query_type}")
    else:
        lines.append("Entity Match: No match found")

    lines.append("\nRAG Answer:")
    lines.append(rag_response.get("llm_answer", "No answer"))

    # Bare Qwen Response
    lines.append("\n" + "=" * 50)
    lines.append("🤖 BARE QWEN RESPONSE:")
    lines.append("-" * 50)
    lines.append("Method: Direct LLM only (no RAG)")
    lines.append("\nBare Qwen Answer:")
    lines.append(bare_response.get("llm_answer", "No answer"))

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def main():
    """
    Main entry point for command-line usage.
    """
    parser = argparse.ArgumentParser(
        description="Query ISO 15926 RDL data using Qwen LLM"
    )
    parser.add_argument("query", nargs="?", help="Natural language query")
    parser.add_argument(
        "--endpoint",
        default="https://data.posccaesar.org/rdl/sparql",
        help="SPARQL endpoint URL",
    )
    parser.add_argument(
        "--quantization",
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="Quantization level for Qwen model",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (text or json)",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare RAG vs bare Qwen responses"
    )
    parser.add_argument(
        "--bare-only", action="store_true", help="Use only bare Qwen (no RAG)"
    )

    args = parser.parse_args()

    if args.interactive:
        mode_desc = (
            "Comparison" if args.compare else ("Bare Qwen" if args.bare_only else "RAG")
        )
        print(f"RDL-Qwen Query System ({mode_desc} Mode)")
        print("Enter 'exit' or 'quit' to end the session")
        print("-" * 60)

        while True:
            query = input("\nEnter your query: ")
            if query.lower() in ["exit", "quit"]:
                break

            try:
                if args.compare:
                    comparison = process_comparison(
                        query=query,
                        endpoint=args.endpoint,
                        quantization=args.quantization,
                    )
                    print("\n" + format_comparison(comparison, args.format))
                elif args.bare_only:
                    response = process_bare_qwen_query(
                        query=query,
                        endpoint=args.endpoint,
                        quantization=args.quantization,
                    )
                    print("\n" + format_response(response, args.format))
                else:
                    response = process_query(
                        query=query,
                        endpoint=args.endpoint,
                        quantization=args.quantization,
                    )
                    print("\n" + format_response(response, args.format))
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"Error: {e}")

    elif args.query:
        try:
            if args.compare:
                comparison = process_comparison(
                    query=args.query,
                    endpoint=args.endpoint,
                    quantization=args.quantization,
                )
                print(format_comparison(comparison, args.format))
            elif args.bare_only:
                response = process_bare_qwen_query(
                    query=args.query,
                    endpoint=args.endpoint,
                    quantization=args.quantization,
                )
                print(format_response(response, args.format))
            else:
                response = process_query(
                    query=args.query,
                    endpoint=args.endpoint,
                    quantization=args.quantization,
                )
                print(format_response(response, args.format))
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Error: {e}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
