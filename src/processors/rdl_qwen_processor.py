#!/usr/bin/env python
"""
Query processor for the RDL QA system using the Qwen LLM.
"""
import json
import logging
from typing import Dict, List, Any, Optional

# Import required components
from src.llm.qwen_model import QwenModel
from src.retrieval.rdl_query_pipeline import RDLQueryPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RDLQwenQueryProcessor:
    """
    Component for processing queries about POSC Caesar RDL data using Qwen LLM.
    Integrates the RDL query pipeline with Qwen LLM to provide enhanced answers
    following IDO (Information and Data Object) principles.
    """

    def __init__(
        self,
        qwen_model: Optional[QwenModel] = None,
        endpoint: str = "https://data.posccaesar.org/rdl/sparql",
    ):
        """
        Initialize the RDL-Qwen query processor.

        Args:
            qwen_model: Optional pre-initialized QwenModel instance
            endpoint: SPARQL endpoint URL for RDL (default is POSC Caesar public endpoint)
        """
        # Initialize RDL pipeline
        self.rdl_pipeline = RDLQueryPipeline(endpoint=endpoint)

        # Ensure E5 term matcher is using CPU
        if (
            not hasattr(self.rdl_pipeline.term_matcher, "embeddings_handler")
            or self.rdl_pipeline.term_matcher.embeddings_handler.device != "cpu"
        ):
            # Replace term matcher with one that explicitly uses CPU
            from src.retrieval.e5_term_matcher import E5TermMatcher
            from src.embeddings.e5_embeddings import E5EmbeddingsHandler

            # Create embeddings handler with CPU device
            embeddings_handler = E5EmbeddingsHandler(device="cpu")
            # Create term matcher with CPU embeddings handler
            term_matcher = E5TermMatcher(embeddings_handler=embeddings_handler)
            self.rdl_pipeline.term_matcher = term_matcher
            logger.info("Replaced term matcher with CPU-specific version")

        # Initialize or use provided Qwen model
        if qwen_model:
            self.llm = qwen_model
        else:
            # Default to 4-bit quantization for efficiency
            self.llm = QwenModel(quantization="4bit")
            logger.info("Initialized Qwen model with 4-bit quantization")

    def _is_hierarchical_query(self, query: str) -> bool:
        """
        Check if a query is asking for hierarchical relationships.

        Args:
            query: Query string

        Returns:
            True if the query is asking for hierarchical relationships
        """
        # Use Qwen to determine if this is a hierarchical query
        prompt = f"""
        You are an ontology expert. Determine if this query is asking about hierarchical relationships 
        (like subtypes, supertypes, parent-child class relationships, types of, etc.).
        
        Query: "{query}"
        
        Respond with only "True" or "False".
        """

        try:
            response = self.llm.generate(prompt).strip()
            return response.lower() == "true"
        except Exception as e:
            logger.warning(f"Error determining if hierarchical query: {e}")
            # Default to False if we can't determine
            return False

    def _extract_key_term(self, query: str) -> str:
        """
        Extract the key term from a hierarchical query.

        Args:
            query: Query string

        Returns:
            Key term
        """
        # Use Qwen to extract the key term
        prompt = f"""
        You are an ontology expert. Extract the main entity or term for which hierarchical
        relationships (subtypes, supertypes, etc.) are being queried.
        
        For example:
        - From "What are the types of pump?" → extract "pump"
        - From "Show me subtypes of centrifugal pump" → extract "centrifugal pump"
        - From "List all the classes that the valve belongs to" → extract "valve"
        
        Query: "{query}"
        
        Respond with only the extracted term (a single word or phrase), nothing else.
        """

        try:
            response = self.llm.generate(prompt).strip()

            # Clean up the response to extract just the term
            # First try to find the term at the end of the response
            lines = [line.strip() for line in response.split("\n") if line.strip()]
            if lines:
                # Take the last non-empty line as the term
                term = lines[-1]

                # Remove any assistant/user prefixes
                if "assistant" in term.lower():
                    parts = term.split("assistant", 1)
                    if len(parts) > 1:
                        term = parts[1].strip()

                # Clean up any remaining quotes or punctuation
                term = term.strip("\"'.,;:")

                logger.info(f"Extracted key term: '{term}'")
                return term

            return response
        except Exception as e:
            logger.warning(f"Error extracting key term: {e}")
            return ""

    def _is_supertype_query(self, query: str) -> bool:
        """
        Check if a query is asking for supertypes.

        Args:
            query: Query string

        Returns:
            True if the query is asking for supertypes
        """
        # Ask Qwen to determine if this is a supertype query
        prompt = f"""
        You are an ontology expert. Determine if this query is asking for superclasses/supertypes/parent classes.
        
        Query: "{query}"
        
        Respond with only "True" or "False".
        """

        try:
            response = self.llm.generate(prompt).strip()
            return response.lower() == "true"
        except Exception as e:
            logger.warning(f"Error determining if supertype query: {e}")
            # Default to subtypes if we can't determine
            return False

    def _process_hierarchical_query(self, query: str) -> Dict[str, Any]:
        """
        Process a hierarchical query.

        Args:
            query: Query string

        Returns:
            Response dictionary
        """
        # Extract the key term from the query
        key_term = self._extract_key_term(query)
        if not key_term:
            return {
                "question": query,
                "llm_answer": "I couldn't identify the term to find hierarchical relationships for. Please specify the term more clearly.",
                "context": {
                    "matched_term": "",
                    "confidence": 0.0,
                    "hierarchical_query": True,
                    "query_type": "hierarchical",
                },
            }

        # Match the key term to an RDL term
        term_matcher = self.rdl_pipeline.term_matcher
        matched_term, confidence, uri = term_matcher.match_term(key_term)

        # Check if we're asking for supertypes or subtypes
        is_supertype_query = self._is_supertype_query(query)

        # Log the query direction determination
        logger.debug(f"Query '{query}' -> is_supertype_query: {is_supertype_query}")

        # Set relationship types based on direction
        if is_supertype_query:
            # Get supertypes
            related_terms = term_matcher.get_supertypes(matched_term)
            relationship_type = "supertypes"
            relationship_description = "parent classes"
            relationship_direction = "superclass"
        else:
            # Default to subtypes
            related_terms = term_matcher.get_subtypes(matched_term)
            relationship_type = "subtypes"
            relationship_description = "subclasses"
            relationship_direction = "subclass"

        # Generate response
        if related_terms:
            terms_list = "\n".join([f"- {term}" for term in related_terms])
            answer = f"Here are the {relationship_description} of {matched_term}:\n\n{terms_list}"
        else:
            answer = f"No {relationship_description} found for {matched_term}."

        # Return the response
        return {
            "question": query,
            "llm_answer": answer,
            "context": {
                "matched_term": matched_term,
                "confidence": confidence,
                "hierarchical_query": True,
                "query_type": "hierarchical",
                "relationship_type": relationship_type,
                "relationship_direction": relationship_direction,
                "related_terms": related_terms,
            },
        }

    def _analyze_query_with_qwen(
        self, query: str, entity: str = "", uri: str = ""
    ) -> Dict[str, Any]:
        """
        Use Qwen to analyze the query and generate a SPARQL query if appropriate.

        Args:
            query: Natural language query
            entity: Matched entity name (optional)
            uri: Entity URI (optional)

        Returns:
            Dictionary with query analysis, including query type and SPARQL query
        """
        prompt = self._get_query_analysis_prompt(query, entity, uri)

        try:
            response = self.llm.generate(prompt)
            logger.debug(f"Qwen analysis response: {response}")

            # Log the full response for debugging
            logger.debug(f"Full Qwen query analysis response:\n{response}")

            # Log the raw response before any processing
            logger.info(f"Raw Qwen response for query '{query}':\n{response}")

            # Find ALL JSON objects in the response and pick the LAST one
            # This ensures we get Qwen's actual response, not examples from the prompt
            # Use a more robust regex that properly matches complete JSON objects
            # This pattern matches balanced braces and handles nested structures
            import re

            # First try to find JSON in code blocks
            code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
            code_matches = re.findall(code_block_pattern, response, re.DOTALL)

            if code_matches:
                # Take the last JSON from code blocks
                analysis_json = code_matches[-1]
                logger.info(
                    f"Found {len(code_matches)} JSON objects in code blocks, using the last one for query '{query}': {analysis_json[:200]}..."
                )
            else:
                # If no code blocks, look for standalone JSON objects
                # This pattern looks for objects that start with { and have "query_type" field
                json_pattern = (
                    r'\{\s*"[^"]*"\s*:\s*"[^"]*"[^}]*"query_type"\s*:\s*"[^"]*"[^}]*\}'
                )
                json_matches = re.findall(json_pattern, response, re.DOTALL)

                if json_matches:
                    # Always take the LAST match as it's Qwen's actual response
                    analysis_json = json_matches[-1]
                    logger.info(
                        f"Found {len(json_matches)} JSON objects with query_type, using the last one for query '{query}': {analysis_json[:200]}..."
                    )
                else:
                    logger.warning(
                        "Could not extract any valid JSON from Qwen response"
                    )
                    # Log full response for debugging
                    logger.debug(f"Full Qwen response: {response}")

                    # Give up and return a fallback analysis
                    logger.warning(
                        f"Failed to extract any JSON from Qwen response for query '{query}'"
                    )
                    return self._fallback_query_analysis(query)

            # Clean up JSON string
            # Replace single quotes with double quotes if needed
            if "'" in analysis_json and '"' not in analysis_json:
                analysis_json = analysis_json.replace("'", '"')

            # Remove trailing commas before closing brackets/braces
            analysis_json = re.sub(r",\s*}", "}", analysis_json)
            analysis_json = re.sub(r",\s*]", "]", analysis_json)

            # Basic cleanup complete, try to parse JSON
            try:
                analysis = json.loads(analysis_json)

                # Print raw analysis for debugging
                logger.info(f"Parsed JSON analysis: {analysis}")

                # Log the query type identified by Qwen
                original_query_type = analysis.get("query_type", "unknown")
                logger.info(
                    f"Qwen classified query '{query}' as type: {original_query_type}"
                )

                # Log the query type identified by Qwen (no overrides)
                logger.info(f"Qwen classified query as: {original_query_type}")

                # Check if this should have been hierarchical
                hierarchical_keywords = [
                    "subclasses",
                    "subtypes",
                    "types of",
                    "children",
                    "parent classes",
                    "superclasses",
                    "supertypes",
                ]
                found_keywords = [
                    kw for kw in hierarchical_keywords if kw in query.lower()
                ]
                if found_keywords and original_query_type != "Hierarchical":
                    logger.warning(
                        f"CLASSIFICATION ERROR: Query '{query}' contains hierarchical keywords {found_keywords} but was classified as '{original_query_type}' instead of 'Hierarchical'"
                    )

                # For property exploration queries, generate a comprehensive property query
                if analysis.get("query_type") == "PropertyExploration" and uri:
                    logger.info("Processing as PropertyExploration query")
                    property_query = self._generate_property_exploration_sparql(uri)
                    analysis["sparql_query"] = property_query
                    logger.info("Generated SPARQL for property exploration query")
                    return analysis

                # For hierarchical queries, verify the SPARQL is correct
                if analysis.get("query_type") == "Hierarchical" and uri:
                    logger.info("Processing as Hierarchical query")

                    # Ensure relationship_direction exists and is valid
                    if (
                        "relationship_direction" not in analysis
                        or not analysis["relationship_direction"]
                    ):
                        # Default to subclass for "types of" queries
                        if "types of" in query.lower():
                            analysis["relationship_direction"] = "subclass"
                        else:
                            analysis["relationship_direction"] = "superclass"
                        logger.info(
                            f"Setting relationship direction to: {analysis['relationship_direction']}"
                        )

                    # Generate or verify SPARQL
                    corrected_query = self._verify_hierarchical_sparql(
                        analysis["relationship_direction"],
                        uri,
                        analysis.get("sparql_query", ""),
                    )

                    analysis["sparql_query"] = corrected_query
                    logger.info(
                        "Generated/corrected SPARQL query for hierarchical relationship"
                    )

                return analysis

            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing JSON: {e}")
                logger.debug(f"Problematic JSON: {analysis_json}")
                logger.warning(f"JSON parsing failed for query '{query}': {e}")
                logger.warning(f"Problematic JSON for query '{query}': {analysis_json}")
                return self._fallback_query_analysis(query)

        except Exception as e:
            logger.error(f"Error analyzing query with Qwen: {e}")
            return self._fallback_query_analysis(query)

    def _verify_hierarchical_sparql(
        self, relationship_direction: str, uri: str, sparql_query: str
    ) -> str:
        """
        Verify and correct a SPARQL query for hierarchical relationships.

        Args:
            relationship_direction: "subclass" or "superclass"
            uri: Entity URI
            sparql_query: Original SPARQL query

        Returns:
            Corrected SPARQL query
        """
        # Make sure we have the right prefixes - use iso: to match Qwen's output
        prefixes = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX iso: <http://rds.posccaesar.org/2008/02/OWL/ISO-15926-2_2003#>
        """

        # Build the appropriate query based on relationship direction
        if relationship_direction == "subclass":
            # To find children/subtypes, the resource has the entity as superclass
            query_pattern = f"""
            SELECT ?resource ?label WHERE {{
              ?resource iso:hasSuperclass <{uri}> .
              OPTIONAL {{ ?resource rdfs:label ?label . }}
            }}
            """
        else:
            # To find parents/supertypes, the entity has the resource as superclass
            query_pattern = f"""
            SELECT ?resource ?label WHERE {{
              <{uri}> iso:hasSuperclass ?resource .
              OPTIONAL {{ ?resource rdfs:label ?label . }}
            }}
            """

        # Combine prefixes and pattern
        corrected_query = prefixes.strip() + "\n\n" + query_pattern.strip()

        return corrected_query

    def _generate_property_exploration_sparql(self, uri: str) -> str:
        """
        Generate a SPARQL query to explore all properties related to an entity in both directions,
        including properties of entities that relate TO the target entity.

        Args:
            uri: Entity URI

        Returns:
            SPARQL query for comprehensive property exploration
        """
        return f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX iso: <http://rds.posccaesar.org/2008/02/OWL/ISO-15926-2_2003#>
        
        SELECT DISTINCT ?property ?propertyLabel ?direction ?relatedEntity ?relatedLabel 
                        ?relatedProperty ?relatedPropertyLabel ?relatedValue ?relatedValueLabel
        WHERE {{
          {{
            # Properties FROM the entity (outgoing relations)
            <{uri}> ?property ?relatedEntity .
            FILTER(?property != rdfs:label)
            FILTER(?property != iso:hasSuperclass)
            BIND("FROM entity" as ?direction)
            
            # Get properties of the related entity (optional)
            OPTIONAL {{
              ?relatedEntity ?relatedProperty ?relatedValue .
              FILTER(?relatedProperty != rdfs:label)
              FILTER(?relatedProperty != iso:hasSuperclass)
              OPTIONAL {{ ?relatedProperty rdfs:label ?relatedPropertyLabel }}
              OPTIONAL {{ 
                FILTER(isURI(?relatedValue))
                ?relatedValue rdfs:label ?relatedValueLabel 
              }}
            }}
          }}
          UNION
          {{
            # Properties TO the entity (incoming relations)
            ?relatedEntity ?property <{uri}> .
            FILTER(?property != rdfs:label)
            FILTER(?property != iso:hasSuperclass)
            BIND("TO entity" as ?direction)
            
            # Get properties of the entity that relates TO our target
            OPTIONAL {{
              ?relatedEntity ?relatedProperty ?relatedValue .
              FILTER(?relatedProperty != rdfs:label)
              FILTER(?relatedProperty != iso:hasSuperclass)
              OPTIONAL {{ ?relatedProperty rdfs:label ?relatedPropertyLabel }}
              OPTIONAL {{ 
                FILTER(isURI(?relatedValue))
                ?relatedValue rdfs:label ?relatedValueLabel 
              }}
            }}
          }}
          
          OPTIONAL {{ ?property rdfs:label ?propertyLabel }}
          OPTIONAL {{ ?relatedEntity rdfs:label ?relatedLabel }}
        }}
        ORDER BY ?direction ?property
        """

    def _fallback_query_analysis(self, query: str) -> Dict[str, Any]:
        """
        Perform a fallback query analysis when all parsing methods fail.

        Args:
            query: The original query string

        Returns:
            A basic query analysis
        """
        # Use Qwen to determine the most likely query type
        prompt = f"""
        You are an ontology expert. Analyze this query and determine its type.
        
        Query: "{query}"
        
        Select the most appropriate type from these options and respond with ONLY that type:
        - Hierarchical (asking about subtypes, supertypes, class/subclass relationships)
        - PropertyExistence (asking if an entity has a specific property/parameter)
        - PropertyBasedClassSearch (asking for classes with a specific property)
        - PropertyExploration (asking for information, properties, or relationships of an entity)
        """

        try:
            query_type = self.llm.generate(prompt).strip()
        except Exception as e:
            logger.error(f"Error in Qwen-based fallback classification: {e}")
            # Default to Definition as safest fallback
            query_type = "Definition"

        # Extract the key term using our existing method
        target_term = self._extract_key_term(query)

        # Set appropriate relationship direction for hierarchical queries
        relationship_direction = ""
        if query_type == "Hierarchical":
            relationship_direction = (
                "superclass" if self._is_supertype_query(query) else "subclass"
            )

        # Generate SPARQL query if we have a term URI
        sparql_query = ""
        uri_match = None

        # Use term matcher to find URI for the target term
        if target_term:
            _, _, uri_match = self.rdl_pipeline.term_matcher.match_term(target_term)

        if uri_match:
            if query_type == "Hierarchical":
                sparql_query = self._verify_hierarchical_sparql(
                    relationship_direction, uri_match, ""  # No original query to verify
                )
            elif query_type == "PropertyExploration":
                sparql_query = self._generate_property_exploration_sparql(uri_match)
            else:
                # Default to PropertyExploration for any other query type
                sparql_query = self._generate_property_exploration_sparql(uri_match)

        return {
            "query_type": query_type,
            "target_term": target_term,
            "relationship_direction": relationship_direction,
            "sparql_query": sparql_query,
            "error": "Generated via fallback analysis",
        }

    def _get_query_analysis_prompt(
        self, query: str, entity: str = "", uri: str = ""
    ) -> str:
        """
        Generate a prompt for Qwen to analyze a query and generate SPARQL.

        Args:
            query: Natural language query
            entity: Entity identified by term matcher
            uri: URI of the entity

        Returns:
            Prompt for Qwen
        """
        # Construct entity information section
        entity_info = ""
        if entity and uri:
            entity_info = f"""
        ENTITY INFORMATION:
        - Matched entity: {entity}
        - Entity URI: {uri}
        Use this entity and URI for your SPARQL query when appropriate.
        """

        # Determine the target entity to use in examples
        example_entity = entity.lower() if entity else "ENTITY_NAME"
        example_uri = uri if uri else "ENTITY_URI"

        # Log the query being analyzed for debugging
        logger.info(f"Analyzing query for classification: '{query}'")

        # Check for hierarchical keywords explicitly
        hierarchical_keywords = [
            "subclasses",
            "subtypes",
            "types of",
            "children",
            "parent classes",
            "superclasses",
            "supertypes",
        ]
        found_keywords = [kw for kw in hierarchical_keywords if kw in query.lower()]
        logger.info(f"Found hierarchical keywords in query: {found_keywords}")

        # Build the prompt for SPARQL query analysis
        return f"""
        You are an expert in RDL (Reference Data Library) ontology and SPARQL queries. 
        Your task is to analyze the following question and generate an appropriate SPARQL query:

        QUESTION: "{query}"
        {entity_info}
        
        STEP 1: CLASSIFY THE QUERY TYPE
        
        If the question asks for "information", "properties", "what is", "describe" → It is PropertyExploration
        
        CRITICAL EXAMPLES:
        - "List me all the subclasses of TANK" → Contains "subclasses" → HIERARCHICAL
        - "What are the types of pump?" → Contains "types of" → HIERARCHICAL  
        - "Show me all subtypes of valve" → Contains "subtypes" → HIERARCHICAL
        - "Tell me about TANK" → Contains "about" → PropertyExploration
        - "What is a pump?" → Contains "what is" → PropertyExploration
        
        
        STEP 2: GENERATE SPARQL
        
        Now identify what type of query this is:
        1. Hierarchical - Asking about class/subclass relationships (types, subtypes, subclasses, superclasses)
        2. PropertyExistence - Asking if an entity has a specific property/parameter
        3. PropertyBasedClassSearch - Asking for classes with a specific property/parameter
        4. PropertyExploration - Asking for information, properties, or relationships of a class

        Then, extract key information and generate a valid SPARQL query for the RDL ontology.

        For the RDL ontology:
        - URIs are in the form http://data.posccaesar.org/rdl/RDS[ID]
        - Class hierarchies use iso:hasSubclass and iso:hasSuperclass (NOT rdfs:subClassOf)
        - Properties are typically connected via predicates like 'has parameter'
        - Entity labels are stored with rdfs:label
        - Descriptions/definitions use rdfs:comment or hasDefinition
        - If you are not sure about what the query is asking for, query the entity's predicates and objects to get all its properties

        Make sure to use all the necessary prefixes in your SPARQL query:
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX iso: <http://rds.posccaesar.org/2008/02/OWL/ISO-15926-2_2003#>

        IMPORTANT EXAMPLES for different query types:
        
        1. For PROPERTY EXPLORATION queries to get information, properties, and relationships of an entity:
        ```json
        {{
          "query_type": "PropertyExploration",
          "target_term": "{example_entity}",
          "target_property": "",
          "relationship_direction": "",
          "sparql_query": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX iso15926-2: <http://rds.posccaesar.org/2008/02/OWL/ISO-15926-2_2003#> SELECT DISTINCT ?property ?propertyLabel ?direction ?relatedEntity ?relatedLabel WHERE {{ {{ <{example_uri}> ?property ?relatedEntity . FILTER(?property != rdfs:label) BIND(\\"FROM entity\\" as ?direction) }} UNION {{ ?relatedEntity ?property <{example_uri}> . FILTER(?property != rdfs:label) BIND(\\"TO entity\\" as ?direction) }} OPTIONAL {{ ?property rdfs:label ?propertyLabel }} OPTIONAL {{ ?relatedEntity rdfs:label ?relatedLabel }} }} ORDER BY ?direction ?property"
        }}
        ```
        
        2. For HIERARCHICAL queries to find SUBCLASSES of an entity (children):
        ```json
        {{
          "query_type": "Hierarchical",
          "target_term": "{example_entity}",
          "target_property": "",
          "relationship_direction": "subclass",
          "sparql_query": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX iso15926-2: <http://rds.posccaesar.org/2008/02/OWL/ISO-15926-2_2003#> SELECT ?resource ?label WHERE {{ ?resource iso15926-2:hasSuperclass <{example_uri}> . OPTIONAL {{ ?resource rdfs:label ?label . }} }}"
        }}
        ```
        
        3. For HIERARCHICAL queries to find SUPERCLASSES of an entity (parents):
        ```json
        {{
          "query_type": "Hierarchical",
          "target_term": "{example_entity}",
          "target_property": "",
          "relationship_direction": "superclass",
          "sparql_query": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX iso15926-2: <http://rds.posccaesar.org/2008/02/OWL/ISO-15926-2_2003#> SELECT ?resource ?label WHERE {{ <{example_uri}> iso15926-2:hasSuperclass ?resource . OPTIONAL {{ ?resource rdfs:label ?label . }} }}"
        }}
        ```
        


        Return your analysis as a JSON object with the following structure, similar to the examples above.

        IMPORTANT: 
        1. The JSON response must be valid with no syntax errors
        2. Use empty strings for fields that don't apply to the query type
        3. For relationship_direction, use ONLY "subclass" or "superclass" without any additional text
        4. The JSON must be properly formatted with double quotes around field names and string values
        5. Do not include comments or explanations in the JSON object
        6. Ensure the SPARQL query is valid and enclosed in double quotes
        7. Include ALL the necessary prefixes in your SPARQL query
        8. If an entity and URI are provided, use them in your SPARQL query instead of placeholder values
        9. For hierarchical queries, use iso15926-2:hasSuperclass correctly:
            - To find subclasses (children): ?resource iso15926-2:hasSuperclass <EntityURI>
            - To find superclasses (parents): <EntityURI> iso15926-2:hasSuperclass ?resource
        10. For definition queries, query the entity's predicates and objects to get all its properties
        11. CRITICAL: Use the actual matched entity from the ENTITY INFORMATION section as your target_term, NOT the examples above

        Return your analysis only as a valid JSON object enclosed in triple backticks like this:
        ```json
        {{
          "query_type": "PropertyExploration",
          "target_term": "{example_entity}",
          "target_property": "",
          "relationship_direction": "",
          "sparql_query": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX iso15926-2: <http://rds.posccaesar.org/2008/02/OWL/ISO-15926-2_2003#> SELECT DISTINCT ?property ?propertyLabel ?direction ?relatedEntity ?relatedLabel WHERE {{ {{ <{example_uri}> ?property ?relatedEntity . FILTER(?property != rdfs:label) BIND(\\"FROM entity\\" as ?direction) }} UNION {{ ?relatedEntity ?property <{example_uri}> . FILTER(?property != rdfs:label) BIND(\\"TO entity\\" as ?direction) }} OPTIONAL {{ ?property rdfs:label ?propertyLabel }} OPTIONAL {{ ?relatedEntity rdfs:label ?relatedLabel }} }} ORDER BY ?direction ?property"
        }}
        ```
        """

    def _get_resource_data(
        self, uris: List[str], max_resources: int = 150
    ) -> List[Dict[str, Any]]:
        """
        Get detailed data for a list of URIs.

        Args:
            uris: List of resource URIs to query
            max_resources: Maximum number of resources to query (default: 150)

        Returns:
            List of resource data dictionaries with label, type, and other properties
        """
        if not uris:
            return []

        # Limit to max_resources to avoid excessive querying
        limited_uris = uris[:max_resources]

        # Initialize combined results
        all_resource_data = []

        # Set batch size to avoid URI too long errors
        batch_size = 25

        # Process URIs in batches
        for i in range(0, len(limited_uris), batch_size):
            batch_uris = limited_uris[i : i + batch_size]

            # Build a SPARQL query for this batch
            values_clause = " ".join([f"<{uri}>" for uri in batch_uris])
            resource_query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            
            SELECT ?resource ?label ?type ?typeLabel ?comment
            WHERE {{
              VALUES ?resource {{ {values_clause} }}
              OPTIONAL {{ ?resource rdfs:label ?label . }}
              OPTIONAL {{ ?resource rdf:type ?type . 
                         OPTIONAL {{ ?type rdfs:label ?typeLabel . }} }}
              OPTIONAL {{ ?resource rdfs:comment ?comment . }}
            }}
            """

            try:
                # Execute query for this batch
                self.rdl_pipeline.sparql.setQuery(resource_query)
                results = self.rdl_pipeline.sparql.query().convert()

                # Process bindings into a more useful format
                bindings = results.get("results", {}).get("bindings", [])

                # Group bindings by resource
                resource_map = {}
                for binding in bindings:
                    resource_uri = binding.get("resource", {}).get("value", "")
                    if not resource_uri:
                        continue

                    if resource_uri not in resource_map:
                        resource_map[resource_uri] = {
                            "uri": resource_uri,
                            "labels": [],
                            "types": [],
                            "type_labels": [],
                            "comments": [],
                        }

                    # Add label if present
                    if "label" in binding:
                        label = binding["label"].get("value", "")
                        if label and label not in resource_map[resource_uri]["labels"]:
                            resource_map[resource_uri]["labels"].append(label)

                    # Add type if present
                    if "type" in binding:
                        type_uri = binding["type"].get("value", "")
                        if (
                            type_uri
                            and type_uri not in resource_map[resource_uri]["types"]
                        ):
                            resource_map[resource_uri]["types"].append(type_uri)

                    # Add type label if present
                    if "typeLabel" in binding:
                        type_label = binding["typeLabel"].get("value", "")
                        if (
                            type_label
                            and type_label
                            not in resource_map[resource_uri]["type_labels"]
                        ):
                            resource_map[resource_uri]["type_labels"].append(type_label)

                    # Add comment if present
                    if "comment" in binding:
                        comment = binding["comment"].get("value", "")
                        if (
                            comment
                            and comment not in resource_map[resource_uri]["comments"]
                        ):
                            resource_map[resource_uri]["comments"].append(comment)

                # Convert map to list and add to combined results
                batch_resource_data = [data for _, data in resource_map.items()]
                all_resource_data.extend(batch_resource_data)

                logger.info(
                    f"Retrieved data for batch of {len(batch_uris)} resources ({i+1} to {min(i+batch_size, len(limited_uris))} of {len(limited_uris)})"
                )

            except Exception as e:
                logger.error(
                    f"Error retrieving resource data for batch {i//batch_size + 1}: {e}"
                )
                # Continue with next batch instead of failing completely
                continue

        logger.info(
            f"Retrieved detailed data for {len(all_resource_data)} resources in total"
        )
        return all_resource_data

    def _execute_sparql(self, sparql_query: str) -> Dict[str, Any]:
        """
        Execute a SPARQL query against the RDL endpoint.

        Args:
            sparql_query: SPARQL query string

        Returns:
            Dictionary with results and resource_data
        """
        try:
            # Validate query structure - ensure balanced braces
            open_braces = sparql_query.count("{")
            close_braces = sparql_query.count("}")

            # If imbalanced, attempt to fix by adding missing closing braces
            if open_braces > close_braces:
                missing_braces = open_braces - close_braces
                sparql_query = sparql_query + " " + "}" * missing_braces
                logger.info(
                    f"Fixed imbalanced SPARQL query by adding {missing_braces} closing braces"
                )

            # Check if query length is reasonable (many SPARQL endpoints have a limit)
            if len(sparql_query) > 8000:
                logger.warning(
                    f"SPARQL query is very long ({len(sparql_query)} chars), which may cause URItoolong errors"
                )

            # Set query to SPARQL endpoint
            self.rdl_pipeline.sparql.setQuery(sparql_query)

            # Execute query
            results = self.rdl_pipeline.sparql.query().convert()

            # Extract bindings
            bindings = results.get("results", {}).get("bindings", [])
            logger.info(f"SPARQL query returned {len(bindings)} results")

            # If we have results without labels, get resource data for context
            resource_data = []
            if bindings and len(bindings) > 0:
                # Check if we have resources without labels
                uris_without_labels = []
                for binding in bindings:
                    if "resource" in binding and (
                        "label" not in binding or not binding["label"].get("value", "")
                    ):
                        uri = binding["resource"].get("value", "")
                        if uri:
                            uris_without_labels.append(uri)

                # If we have URIs without labels, get their data
                if uris_without_labels:
                    max_resources = (
                        150
                        if len(uris_without_labels) > 150
                        else len(uris_without_labels)
                    )
                    logger.info(
                        f"Getting resource data for {max_resources} out of {len(uris_without_labels)} URIs without labels"
                    )
                    resource_data = self._get_resource_data(
                        uris_without_labels, max_resources
                    )

            return {"bindings": bindings, "resource_data": resource_data}
        except Exception as e:
            error_msg = str(e)
            if "URItoolong" in error_msg or "414" in error_msg:
                logger.error(
                    "URI too long error when executing SPARQL query. The query contains too many terms or is too complex."
                )
                # Return empty results but with an error message
                return {
                    "bindings": [],
                    "resource_data": [],
                    "error": "Query too complex: The SPARQL endpoint returned a URI too long error. Try limiting your search or using more specific terms.",
                }
            else:
                logger.error(f"Error executing SPARQL query: {e}")
                return {"bindings": [], "resource_data": []}

    def _format_results(self, results: Any, query_type: str) -> str:
        """
        Format SPARQL query results based on query type.

        Args:
            results: Results from _execute_sparql (dict with bindings and resource_data)
            query_type: Type of query (Hierarchical, PropertyExistence, etc.)

        Returns:
            Formatted results as string
        """
        # Handle both list (legacy) and dict (new) format
        if isinstance(results, list):
            bindings = results
            resource_data = []
        else:
            bindings = results.get("bindings", [])
            resource_data = results.get("resource_data", [])

        if not bindings:
            return "No results found."

        if query_type == "Hierarchical":
            # Format hierarchy results
            labels = []

            # Try to get labels from results first
            for binding in bindings:
                if "label" in binding:
                    label = binding["label"].get("value", "")
                    if label:
                        labels.append(label)

            # If no labels found in results, try to look them up from URIs
            if (
                not labels
                and self.rdl_pipeline
                and hasattr(self.rdl_pipeline, "term_matcher")
            ):
                term_matcher = self.rdl_pipeline.term_matcher

                for binding in bindings:
                    if "resource" in binding:
                        uri = binding["resource"].get("value", "")
                        if uri in term_matcher.uri_to_label:
                            label = term_matcher.uri_to_label[uri]
                            labels.append(label)

                # If we still don't have labels but we have resource data, use that
                if not labels and resource_data:
                    for resource in resource_data:
                        if resource.get("labels"):
                            for label in resource["labels"]:
                                labels.append(label)
                        # If no label, use URI as fallback
                        elif resource.get("uri"):
                            uri = resource["uri"]
                            labels.append(f"{uri.split('/')[-1]}")

            if labels:
                return "• " + "\n• ".join(labels)
            elif resource_data:
                # We have resource data but no labels, provide summary
                return (
                    f"Found {len(resource_data)} types of pump, but they don't have labels. URIs include: "
                    + ", ".join([r["uri"].split("/")[-1] for r in resource_data[:5]])
                    + (
                        f" and {len(resource_data) - 5} more"
                        if len(resource_data) > 5
                        else ""
                    )
                )
            else:
                return "No labeled results found. Retrieved URIs without labels."

        elif query_type == "PropertyExistence":
            # Format property existence results (typically yes/no)
            if len(results) > 0:
                return "Yes, the entity has this property."
            else:
                return "No, the entity does not appear to have this property."

        elif query_type == "PropertyBasedClassSearch":
            # Format classes with property
            classes = []
            for result in results:
                if "class" in result and "classLabel" in result:
                    uri = result["class"].get("value", "")
                    label = result["classLabel"].get("value", "")
                    if label:
                        classes.append(f"{label} ({uri})")
                    else:
                        classes.append(uri)

            if classes:
                return "Classes with the specified property:\n• " + "\n• ".join(classes)
            else:
                return "No classes found with the specified property."

        elif query_type == "PropertyExploration":
            # Format property exploration results with nested properties
            from_properties = {}
            to_properties = {}

            for binding in bindings:
                direction = binding.get("direction", {}).get("value", "")
                property_uri = binding.get("property", {}).get("value", "")
                property_label = binding.get("propertyLabel", {}).get("value", "")
                related_entity = binding.get("relatedEntity", {}).get("value", "")
                related_label = binding.get("relatedLabel", {}).get("value", "")

                # Additional property information
                related_property_uri = binding.get("relatedProperty", {}).get(
                    "value", ""
                )
                related_property_label = binding.get("relatedPropertyLabel", {}).get(
                    "value", ""
                )
                related_value = binding.get("relatedValue", {}).get("value", "")
                related_value_label = binding.get("relatedValueLabel", {}).get(
                    "value", ""
                )

                # Use property label if available, otherwise use URI fragment
                prop_name = (
                    property_label
                    if property_label
                    else property_uri.split("/")[-1].split("#")[-1]
                )

                # Use related entity label if available, otherwise use URI fragment
                related_name = (
                    related_label
                    if related_label
                    else (related_entity.split("/")[-1] if related_entity else "")
                )

                # Create main property info
                property_key = (
                    f"{prop_name} → {related_name}" if related_name else prop_name
                )

                # Handle nested properties
                if related_property_uri:
                    related_prop_name = (
                        related_property_label
                        if related_property_label
                        else related_property_uri.split("/")[-1].split("#")[-1]
                    )

                    related_val_name = (
                        related_value_label
                        if related_value_label
                        else (related_value.split("/")[-1] if related_value else "")
                    )

                    nested_property = f"{related_prop_name}"
                    if related_val_name:
                        nested_property += f" → {related_val_name}"
                else:
                    nested_property = None

                # Group by direction
                if direction == "FROM entity":
                    if property_key not in from_properties:
                        from_properties[property_key] = set()
                    if nested_property:
                        from_properties[property_key].add(nested_property)
                elif direction == "TO entity":
                    if property_key not in to_properties:
                        to_properties[property_key] = set()
                    if nested_property:
                        to_properties[property_key].add(nested_property)

            # Format the results
            result_parts = []

            if from_properties:
                result_parts.append("Properties FROM the entity:")
                count = 0
                # for prop_key, nested_props in list(from_properties.items())[:15]:
                for prop_key, nested_props in list(from_properties.items())[:]:
                    result_parts.append(f"  • {prop_key}")
                    if nested_props:
                        for nested_prop in list(nested_props)[
                            # :3
                            :
                        ]:  # Limit nested properties
                            result_parts.append(f"    - {nested_prop}")
                        # if len(nested_props) > 3:
                        #     result_parts.append(
                        #         f"    - ... and {len(nested_props) - 3} more properties"
                        #     )
                    count += 1

                # if len(from_properties) > 15:
                #     result_parts.append(
                #         f"  ... and {len(from_properties) - 15} more main properties"
                #     )

            if to_properties:
                if result_parts:
                    result_parts.append("")  # Add blank line
                result_parts.append("Properties TO the entity:")
                count = 0
                # for prop_key, nested_props in list(to_properties.items())[:15]:
                for prop_key, nested_props in list(to_properties.items())[:]:
                    result_parts.append(f"  • {prop_key}")
                    if nested_props:
                        for nested_prop in list(nested_props)[
                            # :3
                            :
                        ]:  # Limit nested properties
                            result_parts.append(f"    - {nested_prop}")
                        # if len(nested_props) > 3:
                        #     result_parts.append(
                        #         f"    - ... and {len(nested_props) - 3} more properties"
                        #     )
                    count += 1

                #  if len(to_properties) > 15:
                #     result_parts.append(
                #         f"  ... and {len(to_properties) - 15} more main properties"
                #     )

            if result_parts:
                return "\n".join(result_parts)
            else:
                return "No properties found for this entity."

        # Generic formatting for other query types
        # return json.dumps(results[:5], indent=2) + (
        #     f"\n... and {len(results) - 5} more results" if len(results) > 5 else ""
        # )
        return json.dumps(results, indent=2)

    def _generate_answer_with_qwen(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        results: str,
        type_analysis: Dict = None,
        error: str = None,
    ) -> str:
        """
        Use Qwen to generate a natural language answer based on query results and type analysis.

        Args:
            query: Original query
            query_analysis: Query analysis from Qwen
            results: Formatted query results
            type_analysis: Optional type-based analysis results
            error: Optional error message to include in prompt

        Returns:
            Natural language answer
        """
        # Start with standard prompt
        prompt = f"""
        You are an expert in industrial equipment and ontology, particularly the ISO 15926 / POSC Caesar ontology.
        
        Answer the following question based on the provided query results:
        
        QUESTION: {query}
        
        QUERY TYPE: {query_analysis.get("query_type", "Unknown")}
        
        TARGET TERM: {query_analysis.get("target_term", "")}
        
        {f'TARGET PROPERTY: {query_analysis.get("target_property", "")}' if query_analysis.get("target_property") else ""}
        
        {f'RELATIONSHIP DIRECTION: {query_analysis.get("relationship_direction", "")}' if query_analysis.get("relationship_direction") else ""}
        """

        # Add error information if present
        if error:
            prompt += f"""
            ERROR: {error}
            
            Despite this error, please try to provide the best answer possible based on available information.
            """

        # Add type analysis information if available
        if type_analysis and type_analysis.get("selected_resources"):
            prompt += "\nTYPE ANALYSIS SUMMARY:\n"

            for type_key, resources in type_analysis["selected_resources"].items():
                # Extract simple type name for readability
                if " (" in type_key:
                    type_name = type_key.split(" (")[0]
                else:
                    type_name = type_key.split("/")[-1]

                prompt += f"\n{type_name} ({len(resources)} resources):\n"

                # Add information about each resource
                for i, resource in enumerate(
                    # resources[:3]
                    resources
                ):  # Limit to 3 resources per type
                    # Format with ID first, followed by label if available
                    resource_label = resource.get(
                        "label", resource.get("id", "Unknown")
                    )

                    prompt += f"- {resource_label} (ID: {resource.get('id', '')})"

                    # Add definition if available
                    if "definition" in resource and resource["definition"]:
                        # Truncate definition if too long
                        definition = resource["definition"]
                        if len(definition) > 100:
                            definition = definition[:100] + "..."
                        prompt += f"\n  Definition: {definition}"

                    # Add key properties if available
                    if "properties" in resource:
                        # Select most relevant properties
                        relevant_props = [
                            "hasSuperclass",
                            "hasSubclass",
                            "label",
                            "comment",
                            "hasDefinition",
                        ]
                        added_props = 0

                        for prop in relevant_props:
                            if prop in resource["properties"] and added_props < 3:
                                values = resource["properties"][prop]
                                value_str = ", ".join(values[:2])
                                if len(values) > 2:
                                    value_str += f" and {len(values) - 2} more"
                                prompt += f"\n  {prop}: {value_str}"
                                added_props += 1

                        # Add other properties if we haven't reached the limit yet
                        other_props = [
                            p for p in resource["properties"] if p not in relevant_props
                        ]
                        for prop in other_props[:2]:
                            if added_props < 4:
                                values = resource["properties"][prop]
                                value_str = ", ".join(values[:2])
                                if len(values) > 2:
                                    value_str += f" and {len(values) - 2} more"
                                prompt += f"\n  {prop}: {value_str}"
                                added_props += 1

                    prompt += "\n"

                # Indicate if there are more resources
                if len(resources) > 3:
                    prompt += (
                        f"... and {len(resources) - 3} more resources of this type\n"
                    )

        # Add the original results
        prompt += f"""
        RESULTS:
        {results}
        
        Provide a clear, concise answer based on these results. If the results are empty or don't directly answer the question, say so clearly.
        If appropriate, structure your response as a list for easier reading. Select only the names or any useful information related to the question, without service information like IDs or URIs that does not make sense to human.
        Do not cut your answer short, even if it is too long. I can guarantee that there are no duplicates in the context. Do not overthink too much, but still think for some time.
        """

        try:
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Error generating answer with Qwen: {e}")
            return f"I encountered an error processing the results: {str(e)}"

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query using Qwen for SPARQL generation and execution.

        Args:
            query: Query string

        Returns:
            Response dictionary
        """
        logger.info(f"Processing query: {query}")

        # Make sure we have an initialized pipeline
        if self.rdl_pipeline is None:
            logger.error("RDL pipeline not initialized")
            return {
                "question": query,
                "llm_answer": "RDL pipeline not initialized",
                "context": {},
            }

        # First, use term matcher to identify the entity
        entity, confidence, uri = self.rdl_pipeline.term_matcher.match_term(query)
        logger.info(f"Term matcher found: '{query}' → '{entity}' ({confidence:.1f}%)")

        # Get top matches and best string match
        matching_results = self._get_top_matches(query)
        semantic_matches = matching_results["semantic_matches"]
        best_string_match = matching_results["best_string_match"]

        # If we have an exact match, use it instead of the semantic match
        if best_string_match["exact_match"]:
            logger.info(
                f"Using exact string match '{best_string_match['term']}' instead of "
                f"semantic match '{entity}'"
            )
            entity = best_string_match["term"]
            confidence = best_string_match["confidence"]
            uri = best_string_match["uri"]

        # Step 1: Have Qwen analyze the query and generate SPARQL
        query_analysis = self._analyze_query_with_qwen(query, entity, uri)

        # Step 2: Check if we got a valid analysis with SPARQL query
        if "sparql_query" in query_analysis and query_analysis["sparql_query"]:
            logger.info(
                f"Using Qwen-generated SPARQL for query type: "
                f"{query_analysis.get('query_type')}"
            )

            # Step 3: Execute the generated SPARQL
            results = self._execute_sparql(query_analysis["sparql_query"])

            # Check for errors in SPARQL execution
            error_message = results.get("error", None)

            # Step 4: Perform type-based analysis
            type_analysis = {}
            if isinstance(results, dict) and results.get("bindings"):
                # Extract resource URIs from results
                resource_uris = []
                for binding in results["bindings"]:
                    if "resource" in binding:
                        uri = binding["resource"].get("value", "")
                        if uri:
                            resource_uris.append(uri)

                if resource_uris:
                    # 4.1: Get type information for resources
                    resource_types = self._get_resource_types(resource_uris)

                    # 4.2: Group resources by type
                    type_groups = self._group_resources_by_type(resource_types)

                    # 4.3: Analyze which types are relevant for the query
                    relevant_types = self._analyze_relevant_types(query, type_groups)

                    # 4.4: Select resources for each relevant type
                    selected_resources = self._select_resources_by_type(
                        relevant_types, type_groups, resource_types
                    )

                    # 4.5: Enrich selected resources with full property details
                    enriched_resources = self._enrich_selected_resources(
                        selected_resources
                    )

                    # Store analysis results
                    type_analysis = {
                        "resource_types": resource_types,
                        "type_groups": type_groups,
                        "relevant_types": relevant_types,
                        "selected_resources": enriched_resources,
                    }

                    logger.info(
                        f"Completed type-based analysis with {len(resource_types)} resources and {len(type_groups)} types"
                    )

            # Step 5: Format results based on query type
            formatted_results = self._format_results(
                results, query_type=query_analysis["query_type"]
            )

            # Step 6: Have Qwen generate a natural language answer using enhanced context
            answer = self._generate_answer_with_qwen(
                query=query,
                query_analysis=query_analysis,
                results=formatted_results,
                type_analysis=type_analysis,
                error=error_message,
            )

            # Step 7: Return the full response with both Qwen analysis and entity matching info
            return {
                "question": query,
                "llm_answer": answer,
                "context": {
                    # Entity matching information from term matcher
                    "entity": entity,
                    "matched_term": entity,
                    "confidence": confidence,
                    "uri": uri,
                    "semantic_matches": semantic_matches,
                    "best_string_match": best_string_match,
                    # Qwen analysis information
                    "query_type": query_analysis["query_type"],
                    "target_term": query_analysis.get("target_term", ""),
                    "target_property": query_analysis.get("target_property", ""),
                    "relationship_direction": query_analysis.get(
                        "relationship_direction", ""
                    ),
                    "sparql_query": query_analysis["sparql_query"],
                    "raw_results": (
                        results.get("bindings", [])[:5]
                        if isinstance(results, dict)
                        and len(results.get("bindings", [])) > 5
                        else results.get("bindings", [])
                    ),
                    "resource_data": (
                        results.get("resource_data", [])[:5]
                        if isinstance(results, dict)
                        and len(results.get("resource_data", [])) > 5
                        else results.get("resource_data", [])
                    ),
                    # Type analysis information
                    "type_analysis": type_analysis,
                },
            }

        # Fall back to the existing pipeline if Qwen didn't generate a valid SPARQL query
        logger.info("Falling back to standard query processing")

        # Process the query with fallback, but include matching results
        fallback_response = self._process_query_with_fallback(query)
        fallback_response["context"]["semantic_matches"] = semantic_matches
        fallback_response["context"]["best_string_match"] = best_string_match

        # If we have an exact match, update the main entity in the response
        if best_string_match["exact_match"]:
            fallback_response["context"]["entity"] = best_string_match["term"]
            fallback_response["context"]["matched_term"] = best_string_match["term"]
            fallback_response["context"]["confidence"] = best_string_match["confidence"]
            fallback_response["context"]["uri"] = best_string_match["uri"]

        return fallback_response

    def _process_query_with_fallback(self, query: str) -> Dict[str, Any]:
        """
        Process a query using the standard RDL pipeline as fallback.

        Args:
            query: Query string

        Returns:
            Response dictionary
        """
        # Process the query with RDL pipeline
        rdl_response = self.rdl_pipeline.process_question(query)

        # Extract relevant information for LLM prompt
        entity = rdl_response["context"].get("entity", "")
        matched_term = rdl_response["context"].get("matched_term", "")
        matched_uri = rdl_response["context"].get("uri", "")
        query_type = rdl_response["context"].get("query_type", "")
        confidence = rdl_response["context"].get("confidence", 0)

        # Format results for LLM context
        results_context = self._format_results_for_llm(rdl_response)

        # Generate LLM prompt with contextual information
        prompt = self._generate_ido_prompt(
            query=query,
            entity=entity,
            matched_term=matched_term,
            matched_uri=matched_uri,
            query_type=query_type,
            confidence=confidence,
            results_context=results_context,
        )

        # Get LLM to interpret the results
        llm_answer = self.llm.generate(prompt)

        # Add LLM answer to response
        rdl_response["llm_answer"] = llm_answer

        return rdl_response

    def _format_results_for_llm(self, response: Dict[str, Any]) -> str:
        """
        Format query results in a structured way for the LLM prompt.

        Args:
            response: Response dictionary from RDL pipeline

        Returns:
            Formatted results as a string
        """
        results = response["results"]
        query_type = response["context"].get("query_type", "unknown")

        if not results:
            return "No results found."

        # Format differently based on query type
        if query_type == "subtypes":
            subtype_labels = [
                r.get("label", {}).get("value", "No label") for r in results[:15]
            ]
            return f"Subtypes: {', '.join(subtype_labels)}" + (
                f" and {len(results) - 15} more" if len(results) > 15 else ""
            )

        elif query_type == "supertypes":
            supertype_labels = [
                r.get("label", {}).get("value", "No label") for r in results
            ]
            return f"Supertypes: {', '.join(supertype_labels)}"

        elif query_type == "properties":
            property_descriptions = []
            for r in results:
                prop = r.get("propertyLabel", {}).get("value", "Unknown property")
                dim = r.get("dimensionLabel", {}).get("value", "")
                space = r.get("spaceLabel", {}).get("value", "")

                if dim or space:
                    property_descriptions.append(f"{prop} ({dim or space})")
                else:
                    property_descriptions.append(prop)

            return f"Properties: {'; '.join(property_descriptions[:10])}" + (
                f" and {len(property_descriptions) - 10} more"
                if len(property_descriptions) > 10
                else ""
            )

        elif query_type == "definition":
            definition_parts = []

            # Group by predicates
            predicates = {}
            for r in results:
                pred = r.get("predicateLabel", {}).get(
                    "value", r.get("predicate", {}).get("value", "")
                )
                obj = r.get("objectLabel", {}).get(
                    "value", r.get("object", {}).get("value", "")
                )

                if pred not in predicates:
                    predicates[pred] = []

                predicates[pred].append(obj)

            # Format each predicate and its objects
            for pred, objs in predicates.items():
                if pred and objs:
                    definition_parts.append(
                        f"{pred}: {', '.join(objs[:5])}"
                        + (f" and {len(objs) - 5} more" if len(objs) > 5 else "")
                    )

            return "\n".join(definition_parts)

        elif query_type == "relationship":
            relation_parts = []
            for r in results:
                rel = r.get("relationLabel", {}).get(
                    "value", r.get("relation", {}).get("value", "")
                )
                dir = r.get("direction", {}).get("value", "->")
                relation_parts.append(f"{rel} ({dir})")

            return f"Relationships: {'; '.join(relation_parts)}"

        else:
            # Generic format for other query types
            return json.dumps(results[:5], indent=2) + (
                f"\n... and {len(results) - 5} more results" if len(results) > 5 else ""
            )

    def _generate_ido_prompt(
        self,
        query: str,
        entity: str,
        matched_term: str,
        matched_uri: str,
        query_type: str,
        confidence: float,
        results_context: str,
    ) -> str:
        """
        Generate a prompt for the LLM to interpret RDL query results using IDO principles.

        Args:
            query: Original user query
            entity: Entity extracted from the query
            matched_term: Term matched in RDL
            matched_uri: URI of the matched term
            query_type: Type of query performed
            confidence: Confidence score of the match
            results_context: Formatted query results

        Returns:
            Prompt string for the LLM
        """
        prompt = f"""
        You are an expert in industrial equipment and ontology, particularly the ISO 15926 / POSC
        Caesar ontology following IDO (Information and Data Object) principles.
        
        Answer the following question based on the provided query results and context. 
        Use the IDO principles to structure your answer, focusing on the appropriate IDO aspect
        (Object, Quality, Location, Potential, Temporal, or Activity) based on the query and context.
        
        QUESTION:
        {query}
        
        CONTEXT:
        - Entity: {entity}
        - Matched RDL Term: {matched_term} (confidence: {confidence:.1f}%)
        - RDL URI: {matched_uri}
        - Query Type: {query_type}
        
        QUERY RESULTS:
        {results_context}
        
        Your answer should:
        1. Be concise but informative
        2. Reflect the IDO structure when relevant
        3. Clearly explain what was found in the ontology
        4. Identify any limitations in the data if relevant
        
        ANSWER:
        """

        return prompt

    def get_ido_classification(self, term: str) -> str:
        """
        Classify a term according to IDO principles.

        Args:
            term: The term to classify

        Returns:
            IDO classification (Object, Quality, Location, Potential, Temporal, Activity)
        """
        # This is a simplified implementation that could be expanded
        # with more sophisticated classification logic

        # Example implementation using RDL query to determine type
        try:
            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            
            SELECT ?type ?typeLabel
            WHERE {{
                ?s rdfs:label "{term}"@en .
                ?s rdf:type ?type .
                OPTIONAL {{ ?type rdfs:label ?typeLabel . FILTER(LANG(?typeLabel) = 'en') }}
            }}
            """

            self.rdl_pipeline.sparql.setQuery(query)
            results = self.rdl_pipeline.sparql.query().convert()

            bindings = results["results"]["bindings"]
            if bindings:
                type_uri = bindings[0].get("type", {}).get("value", "")
                type_label = bindings[0].get("typeLabel", {}).get("value", "")

                # Map to IDO classification based on type
                if "Class" in type_uri or "Class" in type_label:
                    return "Object"
                elif "Property" in type_uri or "Property" in type_label:
                    return "Quality"
                elif "Location" in type_uri or "Location" in type_label:
                    return "Location"
                elif (
                    "Function" in type_uri
                    or "Function" in type_label
                    or "Activity" in type_uri
                ):
                    return "Activity"
                elif "Time" in type_uri or "Temporal" in type_label:
                    return "Temporal"
                else:
                    return "Object"  # Default to Object

            return "Object"  # Default
        except Exception as e:
            logger.error(f"Error determining IDO classification: {e}")
            return "Object"  # Default to Object

    def _get_resource_types(
        self, uris: List[str], max_resources: int = 50
    ) -> Dict[str, Dict]:
        """
        Get type information for a list of resource URIs.

        Args:
            uris: List of resource URIs to query
            max_resources: Maximum number of resources to query

        Returns:
            Dictionary mapping resource URIs to their type information
        """
        if not uris:
            return {}

        # Limit to max_resources to avoid excessive querying
        limited_uris = uris[:max_resources]

        # Initialize results dictionary
        resource_types = {}

        # Set batch size to avoid URI too long errors
        batch_size = 25

        # Process URIs in batches
        for i in range(0, len(limited_uris), batch_size):
            batch_uris = limited_uris[i : i + batch_size]

            # Build a SPARQL query to get type info for this batch
            values_clause = " ".join([f"<{uri}>" for uri in batch_uris])
            type_query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            
            SELECT ?resource ?type ?typeLabel 
            WHERE {{
              VALUES ?resource {{ {values_clause} }}
              ?resource rdf:type ?type .
              OPTIONAL {{ ?type rdfs:label ?typeLabel . }}
            }}
            """

            try:
                # Execute query for this batch
                self.rdl_pipeline.sparql.setQuery(type_query)
                results = self.rdl_pipeline.sparql.query().convert()

                # Process bindings into resource → types mapping
                bindings = results.get("results", {}).get("bindings", [])

                for binding in bindings:
                    resource_uri = binding.get("resource", {}).get("value", "")
                    if not resource_uri:
                        continue

                    type_uri = binding.get("type", {}).get("value", "")
                    type_label = binding.get("typeLabel", {}).get("value", "")

                    if resource_uri not in resource_types:
                        resource_types[resource_uri] = {
                            "uri": resource_uri,
                            "types": [],
                            "type_labels": [],
                        }

                    if (
                        type_uri
                        and type_uri not in resource_types[resource_uri]["types"]
                    ):
                        resource_types[resource_uri]["types"].append(type_uri)

                    if (
                        type_label
                        and type_label
                        not in resource_types[resource_uri]["type_labels"]
                    ):
                        resource_types[resource_uri]["type_labels"].append(type_label)

                logger.info(
                    f"Retrieved type information for batch of {len(batch_uris)} resources ({i+1} to {min(i+batch_size, len(limited_uris))} of {len(limited_uris)})"
                )

            except Exception as e:
                logger.error(
                    f"Error retrieving resource types for batch {i//batch_size + 1}: {e}"
                )
                # Continue with next batch instead of failing completely
                continue

        # For URIs without results, add them with empty types
        for uri in limited_uris:
            if uri not in resource_types:
                resource_types[uri] = {"uri": uri, "types": [], "type_labels": []}

        logger.info(
            f"Retrieved type information for {len(resource_types)} resources in total"
        )
        return resource_types

    def _group_resources_by_type(
        self, resource_types: Dict[str, Dict]
    ) -> Dict[str, List[str]]:
        """
        Group resources by their types.

        Args:
            resource_types: Dictionary mapping resource URIs to their type information

        Returns:
            Dictionary mapping type URIs to lists of resource URIs
        """
        type_groups = {}

        # Group resources by type
        for resource_uri, info in resource_types.items():
            for type_uri in info.get("types", []):
                if type_uri not in type_groups:
                    type_groups[type_uri] = []
                type_groups[type_uri].append(resource_uri)

        # Sort type groups by size (number of resources)
        sorted_type_groups = {
            type_uri: resources
            for type_uri, resources in sorted(
                type_groups.items(), key=lambda item: len(item[1]), reverse=True
            )
        }

        # Add type labels for improved readability
        type_labels = {}
        for _, info in resource_types.items():
            for i, type_uri in enumerate(info.get("types", [])):
                if i < len(info.get("type_labels", [])):
                    type_labels[type_uri] = info["type_labels"][i]

        # Map from type URIs to more readable labels where available
        labeled_type_groups = {}
        for type_uri, resources in sorted_type_groups.items():
            type_label = type_labels.get(type_uri, type_uri.split("/")[-1])
            labeled_type_groups[f"{type_label} ({type_uri})"] = resources

        return labeled_type_groups

    def _analyze_relevant_types(
        self, query: str, type_groups: Dict[str, List[str]]
    ) -> List[str]:
        """
        Use Qwen to analyze which types are most relevant for answering the query.

        Args:
            query: The original query string
            type_groups: Dictionary mapping type labels to lists of resource URIs

        Returns:
            List of relevant type keys from the type_groups dictionary
        """
        # Prepare type information for Qwen
        type_info = []
        for type_label, resources in type_groups.items():
            count = len(resources)
            # Only include first few resources as examples
            examples = [r.split("/")[-1] for r in resources[:3]]
            type_info.append(
                f"- {type_label}: {count} resources (e.g., {', '.join(examples)})"
            )

        type_info_str = "\n".join(type_info)

        # Create prompt for Qwen to analyze relevant types
        prompt = f"""
        You are an ontology expert analyzing query results for the question: "{query}"
        
        I have resources grouped by their types. Each group contains resources that belong to that type.
        Here are the types and the number of resources in each type:
        
        {type_info_str}
        
        Based on the query "{query}", which of these types would be MOST RELEVANT for answering the question?
        
        Return ONLY the most relevant type labels from the list above, in order of relevance. 
        Include ONLY the complete type labels exactly as shown above. 
        If none are relevant, say "None of these types are directly relevant".
        Do not include any explanation or additional text.
        """

        try:
            response = self.llm.generate(prompt).strip()

            # Extract type labels from the response
            relevant_types = []
            for line in response.split("\n"):
                line = line.strip()
                # Skip empty lines and lines that don't look like type labels
                if not line or line.startswith("None of these"):
                    if line.startswith("None of these"):
                        logger.info("Qwen indicates no relevant types found")
                    continue

                # Clean up the line to match the format in type_groups
                cleaned_line = line
                if cleaned_line.startswith("- "):
                    cleaned_line = cleaned_line[2:]
                if ":" in cleaned_line:
                    cleaned_line = cleaned_line.split(":", 1)[0].strip()

                # Check if this matches any keys in type_groups
                for type_key in type_groups.keys():
                    if cleaned_line == type_key or cleaned_line in type_key:
                        if type_key not in relevant_types:
                            relevant_types.append(type_key)
                            break

            logger.info(
                f"Identified {len(relevant_types)} relevant types: {relevant_types}"
            )
            return relevant_types

        except Exception as e:
            logger.error(f"Error analyzing relevant types: {e}")
            return []

    def _select_resources_by_type(
        self,
        relevant_types: List[str],
        type_groups: Dict[str, List[str]],
        resource_types: Dict[str, Dict],
    ) -> Dict[str, List[Dict]]:
        """
        Select resources based on relevant types for further analysis.

        Args:
            relevant_types: List of relevant type keys
            type_groups: Dictionary mapping type labels to lists of resource URIs
            resource_types: Dictionary mapping resource URIs to their type information

        Returns:
            Dictionary mapping type labels to lists of selected resources with their details
        """
        selected_resources = {}

        # If no relevant types identified, use all types but limit the resources
        if not relevant_types:
            relevant_types = list(type_groups.keys())[
                :3
            ]  # Use top 3 types by frequency
            logger.info(
                f"No relevant types specified, using top 3 types: {relevant_types}"
            )

        # For each relevant type, select resources and get their details
        for type_key in relevant_types:
            if type_key not in type_groups:
                continue

            # Get URIs for this type
            type_uris = type_groups[type_key]

            # Select a reasonable number of resources (up to 10)
            # selected_uris = type_uris[:10]
            selected_uris = type_uris

            # Get detailed information for selected resources
            selected_resources[type_key] = []
            for uri in selected_uris:
                # First check if we already have type info
                type_info = resource_types.get(uri, {})

                # Get additional resource details (labels, etc.)
                resource_details = self._get_resource_details(uri)

                # Combine info
                combined_info = {
                    "uri": uri,
                    "id": uri.split("/")[-1],
                    "types": type_info.get("types", []),
                    "type_labels": type_info.get("type_labels", []),
                }

                # Add additional details
                if resource_details:
                    combined_info.update(resource_details)

                selected_resources[type_key].append(combined_info)

        return selected_resources

    def _get_resource_details(self, uri: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific resource.

        Args:
            uri: Resource URI

        Returns:
            Dictionary with resource details (label, definition, etc.)
        """
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?label ?comment ?definition
        WHERE {{
          <{uri}> ?p ?o .
          OPTIONAL {{ <{uri}> rdfs:label ?label . }}
          OPTIONAL {{ <{uri}> rdfs:comment ?comment . }}
          OPTIONAL {{ <{uri}> <http://data.posccaesar.org/rdl/hasDefinition> ?definition . }}
        }}
        LIMIT 1
        """

        try:
            # Execute query
            self.rdl_pipeline.sparql.setQuery(query)
            results = self.rdl_pipeline.sparql.query().convert()

            bindings = results.get("results", {}).get("bindings", [])
            if not bindings:
                return {}

            binding = bindings[0]  # Take the first result

            details = {}
            if "label" in binding:
                details["label"] = binding["label"].get("value", "")
            if "comment" in binding:
                details["comment"] = binding["comment"].get("value", "")
            if "definition" in binding:
                details["definition"] = binding["definition"].get("value", "")

            return details

        except Exception as e:
            logger.error(f"Error getting resource details: {e}")
            return {}

    def _get_top_matches(
        self, query: str, k: int = 5, semantic_k: int = 100
    ) -> Dict[str, Any]:
        """
        Get term matches using a two-pass approach:
        1. Get top semantic_k matches using E5 embeddings
        2. Find the best string match using difflib from these semantic matches

        Args:
            query: Query string
            k: Number of top semantic matches to return (default: 5)
            semantic_k: Number of semantic matches to consider for string matching (default: 100)

        Returns:
            Dictionary with semantic_matches (top k) and best_string_match
        """
        # Always extract key term from query
        query_term = query.lower()
        extracted = self._extract_key_term(query)
        key_term = extracted.lower() if extracted else query_term

        logger.info(f"Extracted key term for matching: '{key_term}'")

        # Access the term matcher to use find_most_similar
        term_matcher = self.rdl_pipeline.term_matcher

        # Get all terms
        all_terms = term_matcher.all_terms

        # First check: Look for exact match in all terms (case insensitive)
        exact_match = None
        exact_match_uri = ""

        # Try to find exact match first in all terms
        exact_matches = []
        for term in all_terms:
            if term.lower() == key_term:
                exact_matches.append(term)
                # Get URI for the term
                for u, label in term_matcher.uri_to_label.items():
                    if label == term:
                        exact_match_uri = u
                        break

        # Select the exact match
        if len(exact_matches) > 0:
            # If multiple exact matches, take the shortest one as it's likely more general
            if len(exact_matches) > 1:
                logger.info(f"Multiple exact matches found: {exact_matches}")
                exact_matches.sort(key=len)

            exact_match = exact_matches[0]
            logger.info(f"Found exact string match: '{exact_match}'")

            # Create best string match object for exact match
            best_string_match = {
                "term": exact_match,
                "confidence": 100.0,  # 100% for exact match
                "uri": exact_match_uri,
                "match_type": "string",
                "exact_match": True,
            }

            # We still want to get semantic matches for completeness
            semantic_matches_with_scores = (
                term_matcher.embeddings_handler.find_most_similar(
                    query, all_terms, top_k=semantic_k
                )
            )

            # Process semantic matches
            semantic_matches = []
            for term, similarity in semantic_matches_with_scores[:k]:
                # Convert similarity to confidence percentage
                confidence = float(similarity) * 100

                # Find URI for the term
                uri = ""
                for u, label in term_matcher.uri_to_label.items():
                    if label == term:
                        uri = u
                        break

                match = {
                    "term": term,
                    "confidence": confidence,
                    "uri": uri,
                    "match_type": "semantic",
                }
                semantic_matches.append(match)

            # Log the top semantic matches
            logger.info(f"Top {k} semantic matches for query '{query}':")
            for i, match in enumerate(semantic_matches):
                logger.info(
                    f"  {i+1}. {match['term']} "
                    f"({match['confidence']:.1f}%) - {match['uri']}"
                )

            # Return both semantic matches and best string match
            return {
                "semantic_matches": semantic_matches,
                "best_string_match": best_string_match,
            }

        # If no exact match, proceed with semantic matching and string similarity
        # Use find_most_similar to get top semantic matches
        semantic_matches_with_scores = (
            term_matcher.embeddings_handler.find_most_similar(
                query, all_terms, top_k=semantic_k
            )
        )

        # Process semantic matches
        semantic_matches = []
        for term, similarity in semantic_matches_with_scores[:k]:
            # Convert similarity to confidence percentage
            confidence = float(similarity) * 100

            # Find URI for the term
            uri = ""
            for u, label in term_matcher.uri_to_label.items():
                if label == term:
                    uri = u
                    break

            match = {
                "term": term,
                "confidence": confidence,
                "uri": uri,
                "match_type": "semantic",
            }
            semantic_matches.append(match)

        # Log the top semantic matches
        logger.info(f"Top {k} semantic matches for query '{query}':")
        for i, match in enumerate(semantic_matches):
            logger.info(
                f"  {i+1}. {match['term']} "
                f"({match['confidence']:.1f}%) - {match['uri']}"
            )

        # Second pass: Find best string match using difflib
        from difflib import SequenceMatcher

        # Function to calculate string similarity ratio
        def string_similarity(a, b):
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        # Get all terms from semantic matches
        candidate_terms = [term for term, _ in semantic_matches_with_scores]

        # Find best string match using difflib
        similarity_scores = [
            (term, string_similarity(key_term, term)) for term in candidate_terms
        ]
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        best_term, similarity = similarity_scores[0]
        logger.info(f"Best string match: '{best_term}' (similarity: {similarity:.2f})")

        # Get URI for best string match
        best_uri = ""
        for u, label in term_matcher.uri_to_label.items():
            if label == best_term:
                best_uri = u
                break

        # Create best string match object
        best_string_match = {
            "term": best_term,
            "confidence": float(similarity) * 100,
            "uri": best_uri,
            "match_type": "string",
            "exact_match": False,
        }

        # Return both semantic matches and best string match
        return {
            "semantic_matches": semantic_matches,
            "best_string_match": best_string_match,
        }

    def _get_full_resource_details(self, uri: str) -> Dict[str, Any]:
        """
        Get comprehensive details about a resource including all its properties.

        Args:
            uri: Resource URI

        Returns:
            Dictionary with all resource properties and values
        """
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?predicate ?predicateLabel ?object ?objectLabel
        WHERE {{
          <{uri}> ?predicate ?object .
          OPTIONAL {{ ?predicate rdfs:label ?predicateLabel . }}
          OPTIONAL {{ 
            FILTER(isURI(?object))
            ?object rdfs:label ?objectLabel . 
          }}
        }}
        LIMIT 100
        """

        try:
            # Execute query
            self.rdl_pipeline.sparql.setQuery(query)
            results = self.rdl_pipeline.sparql.query().convert()

            bindings = results.get("results", {}).get("bindings", [])
            if not bindings:
                return {"uri": uri, "properties": {}}

            # Get resource ID from URI
            resource_id = uri.split("/")[-1]

            # Process properties
            properties = {}
            for binding in bindings:
                predicate = binding.get("predicate", {}).get("value", "")
                predicate_label = binding.get("predicateLabel", {}).get("value", "")
                object_value = binding.get("object", {}).get("value", "")
                object_label = binding.get("objectLabel", {}).get("value", "")

                # Skip rdf:type as we already have this information
                if predicate.endswith("#type"):
                    continue

                # Use predicate label if available, otherwise use last part of URI
                prop_name = (
                    predicate_label
                    if predicate_label
                    else predicate.split("/")[-1].split("#")[-1]
                )

                # Use object label if available for URI objects
                if binding.get("object", {}).get("type") == "uri" and object_label:
                    prop_value = f"{object_label} ({object_value})"
                else:
                    prop_value = object_value

                # Group by property name
                if prop_name not in properties:
                    properties[prop_name] = []

                if prop_value not in properties[prop_name]:
                    properties[prop_name].append(prop_value)

            # Create full details
            full_details = {"uri": uri, "id": resource_id, "properties": properties}

            # Add label and definition separately if available
            if "label" in properties:
                full_details["label"] = properties["label"][0]

            if "hasDefinition" in properties:
                full_details["definition"] = properties["hasDefinition"][0]
            elif "comment" in properties:
                full_details["definition"] = properties["comment"][0]

            return full_details

        except Exception as e:
            logger.error(f"Error getting full resource details: {e}")
            return {"uri": uri, "properties": {}}

    def _enrich_selected_resources(
        self, selected_resources: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """
        Enrich selected resources with full details.

        Args:
            selected_resources: Dictionary mapping type labels to lists of selected resources

        Returns:
            Enriched resources with full details
        """
        enriched_resources = {}
        total_enriched = 0

        for type_key, resources in selected_resources.items():
            enriched_resources[type_key] = []

            # Limit to first 5 resources per type to avoid excessive querying
            for resource in resources[:50]:
                uri = resource.get("uri", "")
                if uri:
                    # Get full details
                    full_details = self._get_full_resource_details(uri)

                    # Combine with existing information
                    combined_info = {**resource, **full_details}

                    enriched_resources[type_key].append(combined_info)
                    total_enriched += 1

            # Add remaining resources without enrichment
            if len(resources) > 5:
                for resource in resources[5:]:
                    enriched_resources[type_key].append(resource)

        logger.info(f"Enriched {total_enriched} resources with full details")
        return enriched_resources

    def process_bare_qwen_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query using only bare Qwen LLM without any RAG system.
        This is used for comparison with the full RAG pipeline.

        Args:
            query: Query string

        Returns:
            Response dictionary with bare Qwen answer
        """
        logger.info(f"Processing bare Qwen query: {query}")

        # Create a simple prompt for Qwen about industrial equipment and ontology
        prompt = f"""
        You are an expert in industrial equipment and ontology, particularly the ISO 15926 / POSC Caesar ontology.
        
        Answer the following question based on your knowledge:
        
        Question: {query}
        
        Provide a clear, informative answer about the industrial equipment or ontological concept being asked about.
        """

        try:
            # Get direct response from Qwen
            bare_answer = self.llm.generate(prompt)

            return {
                "question": query,
                "llm_answer": bare_answer,
                "context": {
                    "query_type": "bare_qwen",
                    "method": "direct_llm_only",
                    "no_rag": True,
                },
            }
        except Exception as e:
            logger.error(f"Error generating bare Qwen answer: {e}")
            return {
                "question": query,
                "llm_answer": f"Error generating answer: {str(e)}",
                "context": {
                    "query_type": "bare_qwen",
                    "method": "direct_llm_only",
                    "no_rag": True,
                    "error": str(e),
                },
            }
