#!/usr/bin/env python
import logging
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
EMBEDDINGS_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "rdl_e5_embeddings.pkl",
)


class E5EmbeddingsHandler:
    """
    Handler for generating and managing embeddings using the
    multilingual-e5-large-instruct model.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large-instruct",
        cache_file: str = EMBEDDINGS_CACHE_FILE,
        device: Optional[str] = None,
    ):
        """
        Initialize the embeddings handler.

        Args:
            model_name: Name of the E5 model to use
            cache_file: Path to embeddings cache file
            device: Device to use for model inference (cuda, cpu, etc.)
        """
        self.model_name = model_name
        self.cache_file = cache_file
        # Default to CPU if no device is specified
        self.device = device if device is not None else "cpu"
        self.embeddings_cache = {}

        # Load model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Loaded E5 model: {model_name} on device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading E5 model: {e}")
            raise

        # Load cache if available
        self._load_cache()

    def _load_cache(self) -> None:
        """Load embeddings cache from file if available."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self.embeddings_cache = pickle.load(f)
                logger.info(
                    f"Loaded {len(self.embeddings_cache)} cached embeddings from {self.cache_file}"
                )
            except Exception as e:
                logger.warning(f"Error loading embeddings cache: {e}")
                self.embeddings_cache = {}
        else:
            logger.info("No embeddings cache found, starting with empty cache")
            self.embeddings_cache = {}

    def _save_cache(self) -> None:
        """Save embeddings cache to file."""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.embeddings_cache, f)
            logger.info(
                f"Saved {len(self.embeddings_cache)} embeddings to cache at {self.cache_file}"
            )
        except Exception as e:
            logger.error(f"Error saving embeddings cache: {e}")

    def get_embedding(self, text: str, refresh: bool = False) -> np.ndarray:
        """
        Get embedding for a text string.

        Args:
            text: Text to embed
            refresh: Whether to refresh the cache for this text

        Returns:
            Embedding vector as numpy array
        """
        # Skip empty text
        if not text or text.strip() == "":
            logger.warning("Empty text provided for embedding")
            return np.zeros(1024)  # Return zero vector of appropriate size

        # Check cache first unless refresh is requested
        if not refresh and text in self.embeddings_cache:
            return self.embeddings_cache[text]

        # Preprocess text for the model (for E5, we need to add a prefix)
        # For queries, we should use "query: " prefix
        processed_text = f"query: {text}" if "?" in text else f"passage: {text}"

        try:
            # Generate embedding
            embedding = self.model.encode(processed_text, normalize_embeddings=True)

            # Cache the result
            self.embeddings_cache[text] = embedding

            # Save cache periodically
            if len(self.embeddings_cache) % 100 == 0:
                self._save_cache()

            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            return np.zeros(1024)  # Return zero vector

    def batch_get_embeddings(
        self, texts: List[str], refresh: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            refresh: Whether to refresh the cache for these texts

        Returns:
            Dictionary mapping text to embedding vectors
        """
        result = {}
        cache_hits = 0
        to_encode = []
        text_map = {}  # Maps processed text to original text

        # Check cache and prepare texts for batch encoding
        for text in texts:
            if not text or text.strip() == "":
                result[text] = np.zeros(1024)
                continue

            if not refresh and text in self.embeddings_cache:
                result[text] = self.embeddings_cache[text]
                cache_hits += 1
                continue

            # Preprocess text for the model
            processed_text = f"query: {text}" if "?" in text else f"passage: {text}"
            to_encode.append(processed_text)
            text_map[processed_text] = text

        if to_encode:
            try:
                # Batch encode texts
                embeddings = self.model.encode(
                    to_encode, normalize_embeddings=True, batch_size=32
                )

                # Update cache and result
                for i, processed_text in enumerate(to_encode):
                    original_text = text_map[processed_text]
                    self.embeddings_cache[original_text] = embeddings[i]
                    result[original_text] = embeddings[i]

                # Save cache if significant new embeddings
                if len(to_encode) > 10:
                    self._save_cache()

                logger.info(
                    f"Generated {len(to_encode)} new embeddings "
                    f"(cache hits: {cache_hits})"
                )
            except Exception as e:
                logger.error(f"Error batch encoding texts: {e}")
                # Provide zero vectors for failed encodings
                for processed_text in to_encode:
                    original_text = text_map[processed_text]
                    result[original_text] = np.zeros(1024)

        return result

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0-1)
        """
        # Get embeddings
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        # Compute cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def find_most_similar(
        self, query: str, candidates: List[str], top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar candidates to the query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return

        Returns:
            List of (candidate, similarity_score) tuples, sorted by score
        """
        # Get embeddings
        query_emb = self.get_embedding(query)
        candidate_embs = self.batch_get_embeddings(candidates)

        # Compute similarities
        similarities = []
        for candidate in candidates:
            emb = candidate_embs.get(candidate, np.zeros(1024))
            dot_product = np.dot(query_emb, emb)
            norm1 = np.linalg.norm(query_emb)
            norm2 = np.linalg.norm(emb)

            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm1 * norm2)

            similarities.append((candidate, similarity))

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        return similarities[:top_k]
