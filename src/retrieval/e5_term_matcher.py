#!/usr/bin/env python
import json
import logging
import os
from typing import List, Tuple, Optional
from difflib import SequenceMatcher


from src.embeddings.e5_embeddings import E5EmbeddingsHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths for term mapping files
TERM_MAPPING_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "rdl_term_mapping.json",
)

TERMS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "rdl_terms.json",
)


class E5TermMatcher:
    """
    Matches terms from natural language questions to RDL terms
    using semantic embeddings from the E5 model.
    """

    def __init__(
        self,
        term_mapping_file: str = TERM_MAPPING_FILE,
        terms_file: str = TERMS_FILE,
        embeddings_handler: Optional[E5EmbeddingsHandler] = None,
    ):
        """
        Initialize the E5 term matcher with mapping data.

        Args:
            term_mapping_file: Path to term mapping JSON file
            terms_file: Path to terms JSON file
            embeddings_handler: Optional E5EmbeddingsHandler instance
        """
        # Initialize storage for term data
        self.term_mapping = {}
        self.terms_data = {}
        self.uri_to_label = {}
        self.label_to_uri = {}  # Map from label to URI
        self.all_terms = []
        self.match_cache = {}
        self.hierarchy_cache = {}  # Cache for hierarchy queries

        # Hierarchy relationships
        self.subtype_map = {}  # Map from URI to list of subtype URIs
        self.supertype_map = {}  # Map from URI to list of supertype URIs
        self.entity_type_map = {}  # Map from URI to entity type

        # Set up embeddings handler
        self.embeddings_handler = embeddings_handler or E5EmbeddingsHandler(
            device="cpu"
        )

        # Load term mapping
        if os.path.exists(term_mapping_file):
            with open(term_mapping_file, "r") as f:
                self.term_mapping = json.load(f)
                logger.info(f"Loaded {len(self.term_mapping)} term mappings")

        # Load terms data
        if os.path.exists(terms_file):
            with open(terms_file, "r") as f:
                self.terms_data = json.load(f)
                logger.info(f"Loaded {len(self.terms_data)} terms")

        # Process terms data
        self._process_terms_data()

        # Build hierarchy relationships
        self._build_hierarchy_relationships()

        # Generate embeddings for all terms
        self._generate_term_embeddings()

    def _process_terms_data(self):
        """Process terms data and build internal data structures."""
        self.uri_to_label = {}
        self.label_to_uri = {}
        self.all_terms = []

        # Handle different term data formats
        if isinstance(next(iter(self.terms_data.values()), None), str):
            logger.info(
                "Converting terms data from {term: uri} format to internal structure"
            )
            for term, uri in self.terms_data.items():
                self.uri_to_label[uri] = term
                self.label_to_uri[term.lower()] = uri
                self.all_terms.append(term)
        else:
            for uri, term_data in self.terms_data.items():
                if isinstance(term_data, dict) and "label" in term_data:
                    label = term_data["label"]
                    self.uri_to_label[uri] = label
                    self.label_to_uri[label.lower()] = uri
                    self.all_terms.append(label)
                elif isinstance(term_data, list) and len(term_data) > 0:
                    for item in term_data:
                        if isinstance(item, dict) and "label" in item:
                            label = item["label"]
                            self.uri_to_label[uri] = label
                            self.label_to_uri[label.lower()] = uri
                            self.all_terms.append(label)
                            break
                else:
                    logger.warning(
                        f"Unexpected term data format for URI {uri}: {type(term_data)}"
                    )

    def _build_hierarchy_relationships(self):
        """Build hierarchical relationships between terms."""
        logger.info("Building hierarchical relationships...")

        # Initialize maps
        self.subtype_map = {}
        self.supertype_map = {}
        self.entity_type_map = {}

        # Process each term to extract relationships
        for uri, term_data in self.terms_data.items():
            if isinstance(term_data, dict):
                # Extract entity type
                if "type" in term_data:
                    self.entity_type_map[uri] = term_data["type"]

                # Extract subtypes
                if "subtypes" in term_data and isinstance(term_data["subtypes"], list):
                    self.subtype_map[uri] = term_data["subtypes"]

                    # Update corresponding supertypes
                    for subtype_uri in term_data["subtypes"]:
                        if subtype_uri not in self.supertype_map:
                            self.supertype_map[subtype_uri] = []
                        self.supertype_map[subtype_uri].append(uri)

                # Extract supertypes
                if "supertypes" in term_data and isinstance(
                    term_data["supertypes"], list
                ):
                    self.supertype_map[uri] = term_data["supertypes"]

                    # Update corresponding subtypes
                    for supertype_uri in term_data["supertypes"]:
                        if supertype_uri not in self.subtype_map:
                            self.subtype_map[supertype_uri] = []
                        self.subtype_map[supertype_uri].append(uri)

            # Handle alternative formats
            elif isinstance(term_data, list):
                for item in term_data:
                    if isinstance(item, dict):
                        # Extract entity type
                        if "type" in item:
                            self.entity_type_map[uri] = item["type"]

                        # Extract subtypes
                        if "subtypes" in item and isinstance(item["subtypes"], list):
                            self.subtype_map[uri] = item["subtypes"]

                            # Update corresponding supertypes
                            for subtype_uri in item["subtypes"]:
                                if subtype_uri not in self.supertype_map:
                                    self.supertype_map[subtype_uri] = []
                                self.supertype_map[subtype_uri].append(uri)

                        # Extract supertypes
                        if "supertypes" in item and isinstance(
                            item["supertypes"], list
                        ):
                            self.supertype_map[uri] = item["supertypes"]

                            # Update corresponding subtypes
                            for supertype_uri in item["supertypes"]:
                                if supertype_uri not in self.subtype_map:
                                    self.subtype_map[supertype_uri] = []
                                self.subtype_map[supertype_uri].append(uri)

        logger.info(
            f"Built hierarchy relationships: {len(self.subtype_map)} terms with subtypes, "
            f"{len(self.supertype_map)} terms with supertypes"
        )

    def _generate_term_embeddings(self):
        """Generate embeddings for all terms."""
        logger.info(f"Generating embeddings for {len(self.all_terms)} terms...")
        # Use batch processing for efficiency
        term_embeddings = self.embeddings_handler.batch_get_embeddings(self.all_terms)
        logger.info(f"Generated embeddings for {len(term_embeddings)} terms")

    def _character_level_similarity(self, str1: str, str2: str) -> float:
        """
        Compute character-level similarity between two strings.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score between 0 and 1
        """
        # Convert to lowercase for case-insensitive matching
        str1 = str1.lower()
        str2 = str2.lower()

        # Use SequenceMatcher for character-level similarity
        return SequenceMatcher(None, str1, str2).ratio()

    def match_term(self, term: str) -> Tuple[str, float, str]:
        """
        Match a term from a question to an RDL term using semantic similarity.

        Args:
            term: Term to match

        Returns:
            Tuple of (matched_term, confidence_score, uri)
        """
        # Skip empty terms
        if not term or term.strip() == "":
            logger.warning("Empty term provided for matching")
            return term, 0.0, ""

        # Check cache first
        if term.lower() in self.match_cache:
            return self.match_cache[term.lower()]

        # Try exact match first (case-insensitive)
        term_lower = term.lower()
        for i, candidate in enumerate(self.all_terms):
            if candidate.lower() == term_lower:
                # Find the URI for this term
                uri = ""
                for u, label in self.uri_to_label.items():
                    if label.lower() == term_lower:
                        uri = u
                        break
                logger.info(f"Exact match: '{term}' → '{candidate}' (100%)")
                self.match_cache[term_lower] = (candidate, 100.0, uri)
                return candidate, 100.0, uri

        # If not an exact match, use semantic similarity with E5
        top_matches = self.embeddings_handler.find_most_similar(
            term, self.all_terms, top_k=3
        )

        if not top_matches:
            return term, 0.0, ""

        best_match, similarity = top_matches[0]

        # Convert similarity to percentage confidence
        confidence = min(similarity * 100, 100.0)

        # Find the URI for the best match
        uri = ""
        for u, label in self.uri_to_label.items():
            if label == best_match:
                uri = u
                break

        # Log match details
        log_level = logging.INFO if confidence > 50 else logging.DEBUG
        logger.log(
            log_level,
            f"E5 semantic match: '{term}' → '{best_match}' "
            f"(confidence: {confidence:.1f}%, semantic: {similarity:.2f})",
        )

        # Cache the result
        self.match_cache[term_lower] = (best_match, confidence, uri)
        return best_match, confidence, uri

    def _get_uri_for_term(self, term: str) -> str:
        """Get URI for a term, with matching if necessary."""
        # Check exact match first
        term_lower = term.lower()
        if term_lower in self.label_to_uri:
            return self.label_to_uri[term_lower]

        # Otherwise, match the term
        _, _, uri = self.match_term(term)
        return uri

    def get_entity_type(self, entity: str) -> str:
        """
        Get entity type for a given entity.

        Args:
            entity: Entity label or URI

        Returns:
            Entity type string or empty string if not found
        """
        # Get URI for the entity
        if entity.startswith("http://") or entity.startswith("https://"):
            uri = entity
        else:
            uri = self._get_uri_for_term(entity)

        if not uri:
            return ""

        # Return entity type if available
        return self.entity_type_map.get(uri, "")

    def get_supertypes(self, entity: str) -> List[str]:
        """
        Get supertypes (parent classes) for a given entity.

        Args:
            entity: Entity label or URI

        Returns:
            List of supertype labels
        """
        # Check cache first
        cache_key = f"super_{entity}"
        if cache_key in self.hierarchy_cache:
            return self.hierarchy_cache[cache_key]

        # Get URI for the entity
        if entity.startswith("http://") or entity.startswith("https://"):
            uri = entity
        else:
            uri = self._get_uri_for_term(entity)

        if not uri or uri not in self.supertype_map:
            return []

        # Get supertype URIs
        supertype_uris = self.supertype_map.get(uri, [])

        # Convert URIs to labels
        supertype_labels = []
        for supertype_uri in supertype_uris:
            label = self.uri_to_label.get(supertype_uri)
            if label:
                supertype_labels.append(label)

        # Cache result
        self.hierarchy_cache[cache_key] = supertype_labels
        return supertype_labels

    def get_subtypes(self, entity: str) -> List[str]:
        """
        Get subtypes (subclasses) for a given entity.

        Args:
            entity: Entity label or URI

        Returns:
            List of subtype labels
        """
        # Check cache first
        cache_key = f"sub_{entity}"
        if cache_key in self.hierarchy_cache:
            return self.hierarchy_cache[cache_key]

        # Get URI for the entity
        if entity.startswith("http://") or entity.startswith("https://"):
            uri = entity
        else:
            uri = self._get_uri_for_term(entity)

        if not uri or uri not in self.subtype_map:
            return []

        # Get subtype URIs
        subtype_uris = self.subtype_map.get(uri, [])

        # Convert URIs to labels
        subtype_labels = []
        for subtype_uri in subtype_uris:
            label = self.uri_to_label.get(subtype_uri)
            if label:
                subtype_labels.append(label)

        # Cache result
        self.hierarchy_cache[cache_key] = subtype_labels
        return subtype_labels
