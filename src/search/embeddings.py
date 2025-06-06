"""
Embedding generation and semantic search functionality.
"""

import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Manages embedding generation, caching, and semantic search.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding manager with Sentence-BERT model.
        
        Args:
            model_name: Name of the Sentence-BERT model
                      - all-MiniLM-L6-v2: 384 dims, multilingual, fast
                      - all-mpnet-base-v2: 768 dims, meilleure qualité
        """
        self.model_name = model_name
        self.embedding_dim = 384 if "MiniLM" in model_name else 768
        self._model = None
        
        logger.info(f"Loading Sentence-BERT model: {model_name}")

        try:
            # Charger le modèle Sentence-BERT
            if not self._model:
                self._model = SentenceTransformer(model_name, cache_folder='./cache/sentence_transformers')

            logger.info(f"Model loaded: {self.embedding_dim}D embeddings")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        generate only one embedding for query.
        """
        if not query:
            logger.error("EMPTY QUERY: Aborting")
            return
        
        try:
            query = self._preprocess_text([query])[0]
            query_embedding = self._model.encode(query)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            return query_embedding
        except Exception as e:
            logger.error(f"Error during generating query embeddings : {e}")
            raise
    
    def generate_batch_embeddings(self, text_list: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        generate embedding using batch processing.

        Args:
            text_list (List[str]) : list of text to embed
        
        Returns:
            List[np.ndarray] : List of embeddings vectors
        """
        if not text_list:
            return [np.zeros(self.embedding_dim)]
        
        try:
            text = self._preprocess_text(text_list)

            embeddings = self._model.encode(text, batch_size=batch_size)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings
        except Exception as e :
            logger.error(f"Error during embeddings : {e}")
            raise
    
    def generate_batch_documents_embeddings(self, 
                                            titles: List[str],
                                            contents:List[str],
                                            weight_title: float = 0.4) -> List[np.ndarray]:
        """
        Generate embedding for batches of documents
        """
        try:
            title_embeddings = self.generate_batch_embeddings(text_list=titles)
            content_embeddings = self.generate_batch_embeddings(text_list=contents)
            combined = (weight_title * title_embeddings) + ((1-weight_title)*content_embeddings)

            combined = combined / np.linalg.norm(combined, axis=1, keepdims=True)

            return combined
        except Exception as e:
            logger.error(f"Error during combined embeddings : {e}")
            raise

    def _preprocess_text(self, text_list: List[str]) -> List[str]:
        """Preprocess text for better embedding quality."""
        if not text_list:
            return ""
        
        # Nettoyer et normaliser
        text_list = np.strings.strip(text_list)
        
        return text_list
    
    def compute_similarity(self, emb1: np.ndarray) -> List[float]:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: Normalized embedding array of vectors
            
        Returns:
            Vector of Cosine similarity score (0-1)
        """
        similarities = []
        try:
            # Similarité cosinus (embeddings déjà normalisés)
            for emb in emb1:
                similarity = np.vecdot(emb, emb1)
                similarities.append(similarity)
            return similarities
        except:
            return [0.0]

# Test rapide du module
if __name__ == "__main__":
    # Test basic
    emb_manager = EmbeddingManager()
    
    # Test embedding simple
    text = "FastAPI authentication with JWT tokens"
    embedding = emb_manager.generate_batch_embeddings([text])
    print(f"Generated embedding shape: {embedding.shape}")
    
    # Test similarité
    text1 = "Python web framework"
    text2 = "Flask web development"
    text3 = "Database optimization"
    
    emb = emb_manager.generate_batch_embeddings([text1, text2, text3])

    sim = emb_manager.compute_similarity(emb)
    
    print(f"Similarity 'Python web' vs 'Flask web': {sim[0][1]:.3f}")
    print(f"Similarity 'Python web' vs 'Database': {sim[0][2]:.3f}")
    print("✅ Embedding manager test completed!")