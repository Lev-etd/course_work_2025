#!/usr/bin/env python
"""
Module for generating SPARQL queries from RDL terms.
"""
import logging
import re
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RDLQueryGenerator:
    """Generate SPARQL queries for the RDL."""

    def __init__(self, term_matcher=None):
        """
        Initialize the query generator.

        Args:
            term_matcher: Term matcher instance
        """
        self.term_matcher = term_matcher

    def generate_query(self, question: str, term: str, uri: str) -> Tuple[str, str]:
        """
        Generate a SPARQL query for an entity.

        Args:
            question: Original question
            term: Entity term
            uri: Entity URI

        Returns:
            (query, query_type) tuple
        """
        # Check if it's a hierarchical query
        if self._is_hierarchical_query(question):
            return self._generate_hierarchical_query(term, uri, question)

        # Default to a definition query if not hierarchical
        return self._generate_definition_query(term, uri), "definition"

    def _is_hierarchical_query(self, question: str) -> bool:
        """
        Check if a question is asking for hierarchical relationships.

        Args:
            question: Question string

        Returns:
            True if the question is asking for hierarchical relationships
        """
        # Define patterns for hierarchical queries
        subtype_patterns = [
            r"(?:all|list|show|give\s+me|what\s+are)\s+(?:the\s+)?(?:sub|child)",
            r"(?:all|list|show|give\s+me|what\s+are)\s+(?:the\s+)?(?:types|kinds)",
            r"what\s+(?:sub|child)",
        ]

        supertype_patterns = [
            r"(?:all|list|show|give\s+me|what\s+are)\s+(?:the\s+)?(?:super|parent)",
            r"(?:sub|child)(?:type|class)(?:es)?\s+of\s+what",
            r"what\s+(?:super|parent)",
        ]

        question_lower = question.lower()

        # Check if the question matches any subtype pattern
        for pattern in subtype_patterns:
            if re.search(pattern, question_lower):
                return True

        # Check if the question matches any supertype pattern
        for pattern in supertype_patterns:
            if re.search(pattern, question_lower):
                return True

        return False

    def _is_supertype_query(self, question: str) -> bool:
        """
        Check if a question is asking for supertypes.

        Args:
            question: Question string

        Returns:
            True if the question is asking for supertypes
        """
        supertype_keywords = [
            "supertype",
            "superclass",
            "parent",
            "super",
            "base type",
            "base class",
            "parent type",
            "parent class",
        ]

        question_lower = question.lower()
        for keyword in supertype_keywords:
            if keyword in question_lower:
                return True

        return False

    def _generate_hierarchical_query(
        self, term: str, uri: str, question: str
    ) -> Tuple[str, str]:
        """
        Generate a SPARQL query for a hierarchical relationship.

        Args:
            term: Entity term
            uri: Entity URI
            question: Original question

        Returns:
            (query, query_type) tuple
        """
        # Determine if we're looking for supertypes or subtypes
        if self._is_supertype_query(question):
            query_type = "supertypes"
            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            
            SELECT ?supertype ?label
            WHERE {{
              <{uri}> rdfs:subClassOf ?supertype .
              OPTIONAL {{ ?supertype rdfs:label ?label . }}
            }}
            """
        else:
            query_type = "subtypes"
            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            
            SELECT ?subtype ?label
            WHERE {{
              ?subtype rdfs:subClassOf <{uri}> .
              OPTIONAL {{ ?subtype rdfs:label ?label . }}
            }}
            """

        return query, query_type

    def _generate_definition_query(self, term: str, uri: str) -> str:
        """
        Generate a SPARQL query to get the definition of an entity.

        Args:
            term: Entity term
            uri: Entity URI

        Returns:
            SPARQL query string
        """
        return f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?predicate ?predicateLabel ?object ?objectLabel
        WHERE {{
          <{uri}> ?predicate ?object .
          OPTIONAL {{ ?predicate rdfs:label ?predicateLabel . }}
          OPTIONAL {{ ?object rdfs:label ?objectLabel . }}
        }}
        """
