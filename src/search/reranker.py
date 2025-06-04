"""
Cross Encoder
"""

import logging
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RerankingResult:
    """result of reranking"""
    original_results: List[Dict[str, Any]]
    reranked_results: List[Dict[str, Any]]
    reranking_time: float
    improvements: List[int]

class CrossEncoderReranker:
    """
    Cross encoder based reranking for search results.
    Uses ms-macro model.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Init func.

        Args:
            model_name (str): HuggingFace model name for cross encoder
        """
        self.model_name = model_name
        self.model = None
        self._model_loaded = False

        self.max_input = 512
        self.batch_size = 8

        logger.info("Cross Encoder initialized")

    def _load_model(self) -> None:
        if self._model_loaded:
            return
        
        try:
            from sentence_transformers import CrossEncoder

            start = time.time()
            self.model = CrossEncoder(self.model_name)
            load_time = time.time() - start

            self._model_loaded = True
            logger.info(f"Model loaded in {load_time} seconds")
        
        except ImportError:
            logger.error("sentences-transformer not installed.\nRun 'pip install sentences-transformer' before retrying.")
        except Exception as e:
            logger.error(f"Error caught during loading Cross Encoder model : {e}")
            raise

    def rerank(self,
               query: str,
               results: List[Dict[str, Any]],
               top_k: int) -> RerankingResult:
        """
        Rerank original results using cross-encoder scores.

        Args:
            query: Original search
            results: original results
            top_k: number of top results to return
        
        Returns:
            RerankingResult with reranked results and metadata
        """

        if not results:
            return RerankingResult([], [], 0.0, [])

        start = time.time()

        self._load_model()

        try:
            original_results = results.copy()
            query_doc_pairs = [(query, self._prepare_document_text(result)) for result in results]

            cross_scores = self._compute_cross_scores(query_doc_pairs)
            reranked = self._combine_score(results, cross_scores)
            reranked.sort(key= lambda x: x["final_score"], reverse=True)
            top = reranked[:top_k]

            reranked_time = time.time() - start
            improvements = self._calculate_improvements(original_results, top)

            logger.info(f"Reranking completed in {reranked_time:.2f}")

            return RerankingResult(
                original_results=original_results,
                reranked_results=top,
                reranking_time=reranked_time,
                improvements=improvements
            )
        except Exception as e:
            logger.error(f"Error caught during reranking : {e}")
            return RerankingResult(
                original_results=results,
                reranked_results=results[:top_k],
                reranking_time=time.time()-start,
                improvements=[]
            )
        
    def _prepare_document_text(self, result: Dict[str, Any]) -> str:
        """
        Prepare document text for reranking input
        """
        title = result.get("title", "")
        content = result.get("content", "")

        doc_text = f"{title}. {content}"

        if len(doc_text) > self.max_input:
            doc_text = doc_text[:self.max_input]
        
        return doc_text
    
    def _compute_cross_scores(self, query_doc_pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Compute cross score for each pair of query + doc

        Returns:
            List of relevance cross-score
        """
        if not self.model:
            raise RuntimeError('Cross-encoder model not loaded')
        
        try:
            all_scores = []

            for i in range(0, len(query_doc_pairs), self.batch_size):
                batch = query_doc_pairs[i:i+self.batch_size]

                batch_scores = self.model.predict(batch)
                if hasattr(batch_scores, "tolist"):
                    batch_scores = batch_scores.tolist()
                elif not isinstance(batch_scores, list):
                    batch_scores = [float(batch_scores)]
                all_scores.extend(batch_scores)
            
            return all_scores
        except Exception as e:
            logger.error(f"Error during batch scoring : {e}")
            return [0.5]*len(query_doc_pairs)
        
    def _combine_score(self, results: List[Dict[str, Any]], cross_scores: List[float]) -> List[Dict[str, Any]]:
        """
        Combine all scores to rerank.

        Args:
            results: Original search results
            cross_scores: Cross encoder scores
        
        Returns:
            Results with combined final_score
        """
        combined_results = []

        for i, result in enumerate(results):
            if i < len(cross_scores):
                original_score = result.get('score', 0.0)
                popularity = result.get("popularity_score", 0.0)
                freshness = result.get("freshness_score", 0.0)

                cross_score = cross_scores[i]

                normalized_cross = self._normalize_score(cross_score, -10, 10)
                normalized_original = self._normalize_score(original_score, 0, 1)
                normalized_popularity = self._normalize_score(popularity, 0, 1)
                normalized_freshness = self._normalize_score(freshness, 0, 1)

                final_score = (
                    0.6 * normalized_cross +
                    0.25 * normalized_original +
                    0.10 * normalized_popularity + 
                    0.05 * normalized_freshness
                )
                # Enrich result
                enriched_result = result.copy()
                enriched_result.update({
                    'cross_encoder_score': float(cross_score),
                    'final_score': float(final_score),
                    'score_breakdown': {
                        'cross_encoder': normalized_cross,
                        'original_es': normalized_original,
                        'popularity': normalized_popularity,
                        'freshness': normalized_freshness,
                        'weights': {
                            'cross_encoder': 0.6,
                            'original': 0.25, 
                            'popularity': 0.10,
                            'freshness': 0.05
                        }
                    }
                })
                combined_results.append(enriched_result)
            else:
                result_copy = result.copy()
                result_copy["final_score"] = result.get("score", 0.0)
                combined_results.append(result_copy)
        return combined_results

    def _normalize_score(self, score: float, min_val: float, max_val:float) -> float:
        if max_val == min_val:
            return 0.5
        return max(0.0, min(1.0, (score - min_val)/ (max_val - min_val)))
    
    def _calculate_improvements(self, original_results: List[Dict[str, Any]], reranked_results: List[Dict[str, Any]]) -> List[int]:
        improvements = []
        original_pos = {}
        for i, result in enumerate(original_results):
            res_id = result.get("id", result.get('title', f"doc {i}"))
            original_pos[res_id] = i
        
        for new_pos, result in enumerate(reranked_results):
            res_id = result.get("id", result.get('title', f"doc {new_pos}"))
            origin = original_pos.get(res_id, len(original_results))

            improvements.append(origin - new_pos)
        return improvements
    
    def explain_reranking(self, reranking_result: RerankingResult, query: str) -> str:
        """
        Generate human-readable explanation of reranking.
        
        Returns:
            Formatted explanation string
        """
        if not reranking_result.reranked_results:
            return "No results to explain."
        
        explanation = f"🧠 Cross-Encoder Reranking Analysis\n"
        explanation += f"🔍 Query: '{query}'\n"
        explanation += f"⏱️  Processing time: {reranking_result.reranking_time:.2f}ms\n"
        explanation += f"📊 Documents processed: {len(reranking_result.original_results)}\n"
        explanation += f"🎯 Top results returned: {len(reranking_result.reranked_results)}\n\n"
        
        explanation += "🏆 Top Reranked Results:\n"
        
        for i, result in enumerate(reranking_result.reranked_results[:5], 1):
            title = result.get('title', 'Unknown')[:50]
            original_score = result.get("score_breakdown").get("original_es", 0.0)
            final_score = result.get('final_score', 0)
            cross_score = result.get('cross_encoder_score', 0)
            
            # Position improvement
            improvement = reranking_result.improvements[i-1] if i-1 < len(reranking_result.improvements) else 0
            improvement_text = f"(+{improvement})" if improvement > 0 else f"({improvement})" if improvement < 0 else "(=)"
            
            explanation += f"\n{i}. {title}... {improvement_text}\n"
            explanation += f"      Original score: {original_score:.3f}\n"
            explanation += f"   🎯 Final Score: {final_score:.3f}\n"
            explanation += f"   🧠 AI Relevance: {cross_score:.3f}\n"
            
            if 'score_breakdown' in result:
                breakdown = result['score_breakdown']
                explanation += f"   📊 Components: "
                explanation += f"AI={breakdown['cross_encoder']:.2f}({breakdown['weights']['cross_encoder']:.0%}), "
                explanation += f"ES={breakdown['original_es']:.2f}({breakdown['weights']['original']:.0%}), "
                explanation += f"Pop={breakdown['popularity']:.2f}({breakdown['weights']['popularity']:.0%})\n"
        
        # Summary statistics
        positive_improvements = sum(1 for imp in reranking_result.improvements if imp > 0)
        explanation += f"\n📈 Reranking Impact:\n"
        explanation += f"   • {positive_improvements}/{len(reranking_result.improvements)} results improved position\n"
        
        avg_improvement = np.mean(reranking_result.improvements) if reranking_result.improvements else 0
        explanation += f"   • Average position change: {avg_improvement:+.1f}\n"
        
        return explanation

# Test function
def test_reranker():
    """Test the cross-encoder reranker with sample data."""
    
    # Sample search results
    test_results = [
        {
            'id': '1',
            'title': 'Docker Deployment Best Practices',
            'content': 'This guide covers best practices for deploying applications using Docker containers in production environments.',
            'score': 0.5,
            'popularity_score': 0.45,
            'freshness_score': 0.8
        },
        {
            'id': '2',
            'title': 'FastAPI Authentication Tutorial',
            'content': 'Learn how to implement secure authentication in FastAPI applications using JWT tokens and OAuth2.',
            'score': 0.65,
            'popularity_score': 0.9,
            'freshness_score': 0.6
        },
        {
            'id': '3',
            'title': 'Python Web Framework Comparison',
            'content': 'Comprehensive comparison of Python web frameworks including Django, Flask, and FastAPI.',
            'score': 0.6,
            'popularity_score': 0.7,
            'freshness_score': 0.6
        }
    ]
    
    # Test reranking
    reranker = CrossEncoderReranker()
    
    query = "api authentication security"
    
    result = reranker.rerank(query, test_results, top_k=3)
    
    print(reranker.explain_reranking(result, query))

if __name__ == "__main__":
    test_reranker()