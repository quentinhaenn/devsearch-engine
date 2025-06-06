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
    search_time: float
    reranking_time: Optional[float]
    improvements: Optional[List[int]]

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
               search_time: float,
               top_k: int,
               no_rerank: bool) -> RerankingResult:
        """
        Rerank original results using cross-encoder scores.

        Args:
            query: Original search
            results: original results
            search_time: time taken for original search
            top_k: number of top results to return
            no_rerank: if True, skip reranking and return original results
        
        Returns:
            RerankingResult with reranked results and metadata if not no_rerank else original results.
        Raises:
            RuntimeError: If model is not loaded or if there is an error during reranking.
        """

        if not results:
            return RerankingResult([], [], 0.0, 0.0, [])
        if no_rerank:
            logger.info("Skipping reranking as no_rerank is set to True")
            return RerankingResult(
                original_results=results,
                reranked_results=results[:top_k],
                search_time=search_time,
                reranking_time=0.0,
                improvements=[]
            )

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
                search_time=search_time,
                reranking_time=reranked_time,
                improvements=improvements
            )
        except Exception as e:
            logger.error(f"Error caught during reranking : {e}")
            return RerankingResult(
                original_results=results,
                reranked_results=results[:top_k],
                search_time=search_time,
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
            all_scores = self.model.predict(query_doc_pairs, batch_size=self.batch_size, show_progress_bar=True)
            
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
        o_scores = [res.get("score", 0.0) for res in results]
        p_scores = [res.get("popularity_score", 0.0) for res in results]
        f_scores = [res.get("freshness_score", 0.0) for res in results]

        try:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            normalized_o = scaler.fit_transform(np.reshape(o_scores, (-1, 1))).flatten()
            normalized_p = scaler.fit_transform(np.reshape(p_scores,(-1,1))).flatten()
            normalized_f = scaler.fit_transform(np.reshape(f_scores, (-1,1))).flatten()
            normalized_cross = scaler.fit_transform(np.reshape(cross_scores, (-1,1))).flatten()

        except ImportError:
            normalized_o = self._normalize_score(o_scores)
            normalized_p = self._normalize_score(p_scores)
            normalized_f = self._normalize_score(f_scores)
            normalized_cross = self._normalize_score(cross_scores)

        
        final_scores = (
            0.6 * normalized_cross +
            0.25 * normalized_o +
            0.10 * normalized_p +
            0.05 * normalized_f
        )
        for i, result in enumerate(results):
            enriched_result = result.copy()
            enriched_result.update({
                'cross_encoder_score': float(normalized_cross[i]),
                'final_score': float(final_scores[i]),
                'score_breakdown': {
                    'cross_encoder': normalized_cross[i],
                    'original_es': normalized_o[i],
                    'popularity': normalized_p[i],
                    'freshness': normalized_f[i],
                    'weights': {
                        'cross_encoder': 0.6,
                        'original': 0.25, 
                        'popularity': 0.10,
                        'freshness': 0.05
                    }
                }
            })
            combined_results.append(enriched_result)
        return combined_results

    def _normalize_score(self, scores: List[float]) -> List[float]:
        normalized = (scores - np.min(scores))/(np.max(scores)-np.min(scores))
        return normalized.tolist()
    
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
        return np.array(improvements)
    
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
            improvement = reranking_result.improvements[i]
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
        mask = reranking_result.improvements > 0
        positive_improvements = reranking_result.improvements[mask]
        explanation += f"\n📈 Reranking Impact:\n"
        explanation += f"   • {len(positive_improvements)}/{len(reranking_result.improvements)} results improved position\n"
        
        avg_pos_improvement = np.mean(positive_improvements)
        explanation += f"   • Average positive change: {avg_pos_improvement:+.1f}\n"
        
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