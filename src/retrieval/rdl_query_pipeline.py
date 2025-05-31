#!/usr/bin/env python
"""
Pipeline for processing RDL queries.
"""
import logging
from typing import Dict, Any

from SPARQLWrapper import SPARQLWrapper, JSON

from src.retrieval.rdl_query_generator import RDLQueryGenerator
from src.retrieval.e5_term_matcher import E5TermMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RDLQueryPipeline:
    """Pipeline for processing RDL queries."""

    def __init__(
        self,
        endpoint: str = "https://data.posccaesar.org/rdl/sparql",
        term_matcher=None,
        query_generator=None,
    ):
        """
        Initialize the pipeline.

        Args:
            endpoint: SPARQL endpoint URL
            term_matcher: Term matcher instance
            query_generator: Query generator instance
        """
        self.endpoint = endpoint
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)

        # Initialize term matcher if not provided
        if term_matcher is None:
            self.term_matcher = E5TermMatcher()
        else:
            self.term_matcher = term_matcher

        # Initialize query generator if not provided
        if query_generator is None:
            self.query_generator = RDLQueryGenerator(term_matcher=self.term_matcher)
        else:
            self.query_generator = query_generator

        # LLM reference - will be set by the processor
        self.llm = None

    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a question and return the results.

        Args:
            question: Question string

        Returns:
            Dictionary with results
        """
        logger.info(f"Processing question: {question}")

        # Extract entities from question
        entity, confidence, uri = self.term_matcher.match_term(question)
        logger.info(f"Matched term: '{question}' → '{entity}' ({confidence:.1f}%)")

        # Generate SPARQL query
        query, query_type = self.query_generator.generate_query(question, entity, uri)

        # Execute query
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            bindings = results["results"]["bindings"]
            logger.info(f"Query returned {len(bindings)} results")
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            bindings = []

        # Return results
        return {
            "question": question,
            "results": bindings,
            "context": {
                "entity": entity,
                "matched_term": entity,
                "confidence": confidence,
                "uri": uri,
                "query_type": query_type,
                "sparql_query": query,
            },
        }

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a SPARQL query directly.

        Args:
            query: SPARQL query string

        Returns:
            Dictionary with results
        """
        logger.info(f"Processing SPARQL query: {query[:100]}...")

        # Execute query
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            bindings = results["results"]["bindings"]
            logger.info(f"Query returned {len(bindings)} results")
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            bindings = []

        # Return results
        return {
            "query": query,
            "results": bindings,
            "context": {
                "query_type": "custom",
            },
        }

    def format_results(
        self, response: Dict[str, Any], format_type: str = "text"
    ) -> str:
        """
        Format the query results for display.

        Args:
            response: Response dictionary from process_question
            format_type: Output format (text or json)

        Returns:
            Formatted results as a string
        """
        if format_type == "json":
            return json.dumps(response, indent=2)

        question = response["question"]
        results = response["results"]
        context = response["context"]

        # Format as text
        lines = []
        lines.append(f"Question: {question}")
        lines.append("-" * 80)

        # Entity matching
        matched_term = context.get("matched_term", "")
        confidence = context.get("confidence", 0)

        lines.append("Entity Matching:")
        if matched_term:
            lines.append(
                f"- '{question}' → '{matched_term}' (confidence: {confidence:.1f}%)"
            )
        else:
            lines.append("- No match found")

        lines.append("-" * 80)

        # Results section
        lines.append("Results:")
        if results:
            for result in results[:10]:  # Limit display to 10 results
                values = []
                for key, val in result.items():
                    values.append(f"{key}: {val.get('value', '')}")
                lines.append("- " + ", ".join(values))

            if len(results) > 10:
                lines.append(f"... and {len(results) - 10} more results")
        else:
            lines.append("No results found.")

        return "\n".join(lines)
