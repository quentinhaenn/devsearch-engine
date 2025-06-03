"""
Embedding generation and semantic search functionality.
"""

import logging
import numpy as np
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
        
        logger.info(f"Loading Sentence-BERT model: {model_name}")

        try:
            # Charger le modèle Sentence-BERT
            self.model = SentenceTransformer(model_name)
            logger.info(f"Model loaded: {self.embedding_dim}D embeddings")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of embedding vector
        """
        if not text or not text.strip():
            return np.zeros(self.embedding_dim)
        
        try:
            # Normalize text
            text = self._preprocess_text(text)
            
            # Generate
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype(np.float32) # Custom float32 for less memory usage
            
        except Exception as e:
            logger.error(f"❌ Failed to generate embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def generate_document_embedding(self, title: str, content: str, weight_title: float = 0.3) -> np.ndarray:
        """
        Generate embedding for a document combining title and content.
        
        Args:
            title: Document title
            content: Document content
            weight_title: Weight for title vs content (0.3 = 30% title, 70% content)
            
        Returns:
            Combined embedding vector
        """
        try:
            # Embedding séparés
            title_emb = self.generate_embedding(title)
            content_emb = self.generate_embedding(content[:500])  # Limite pour performance
            
            # Combinaison pondérée
            combined = (weight_title * title_emb) + ((1 - weight_title) * content_emb)
            
            # Re-normaliser
            combined = combined / np.linalg.norm(combined)
            
            return combined
            
        except Exception as e:
            logger.error(f"❌ Failed to generate document embedding: {e}")
            return np.zeros(self.embedding_dim)
    

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better embedding quality."""
        if not text:
            return ""
        
        # Nettoyer et normaliser
        text = text.strip()
        
        # Limiter la taille
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        return text
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1, emb2: Normalized embedding vectors
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Similarité cosinus (embeddings déjà normalisés)
            similarity = np.dot(emb1, emb2)
            return float(similarity)
        except:
            return 0.0

# Test rapide du module
if __name__ == "__main__":
    # Test basic
    emb_manager = EmbeddingManager()
    
    # Test embedding simple
    text = "FastAPI authentication with JWT tokens"
    embedding = emb_manager.generate_embedding(text)
    print(f"Generated embedding shape: {embedding.shape}")
    
    # Test similarité
    text1 = "Python web framework"
    text2 = "Flask web development"
    text3 = "Database optimization"
    
    emb1 = emb_manager.generate_embedding(text1)
    emb2 = emb_manager.generate_embedding(text2)
    emb3 = emb_manager.generate_embedding(text3)
    
    sim_12 = emb_manager.compute_similarity(emb1, emb2)
    sim_13 = emb_manager.compute_similarity(emb1, emb3)
    
    print(f"Similarity 'Python web' vs 'Flask web': {sim_12:.3f}")
    print(f"Similarity 'Python web' vs 'Database': {sim_13:.3f}")
    print("✅ Embedding manager test completed!")