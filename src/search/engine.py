"""
Module for search engine functionality.
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from src.data import client
from .embeddings import EmbeddingManager
from .reranker import CrossEncoderReranker, RerankingResult
from elasticsearch import dsl

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class SearchEngine:

    def __init__(self):
        """
        Initialize the search engine.
        """
        self.client = client.ElasticSearchClient()
        self.indexes = set()
        self.default_index = "devsearch_index"
        self.default_timeout = 30  # seconds
        self.embedding_manager = EmbeddingManager()
        self.reranker = CrossEncoderReranker()

    def _build_query(self, query: str, index_name: str = None, **kwargs) -> Dict[str, Any]:
        """
        Build a search query for ES. Kwargs may include additional parameters like filters, sorting, etc.
        
        Args:
            query (str): The search query string.
            index_name (str): The name of the index to search.
            **kwargs: Additional parameters for the search query.
        
        Returns:
            Dict[str, Any]: A dictionary representing the search query.
        """
        
        top_k = kwargs.get("top_k", 10)
        es_filters = []
        if kwargs.get("filter_file_type"):
            es_filters.append({"terms": {"file_type.keyword" : kwargs.get("filter_file_type")}})
        if kwargs.get("filter_language"):
            es_filters.append({"terms": {"language.keyword" : kwargs.get("filter_language")}})
        
        date_filters = {}
        if kwargs.get("date_from"):
            date_filters["gte"] = kwargs.get("date_from")
        if kwargs.get("date_to"):
            date_filters["lte"] = kwargs.get("date_to")
        
        candidates = top_k*5

        s = dsl.Search()
        knn_s = dsl.Search()
        s = s.from_dict(
            {
            "size": top_k or 10,  # Default to top 10 results if not specified
            "highlight": {
                "fields" :{
                    "title": {
                        "pre_tags": ["<em>"],
                        "post_tags": ["</em>"],
                        "number_of_fragments": 1
                    },
                    "content": {
                        "pre_tags": ["<em>"],
                        "post_tags": ["</em>"],
                        "number_of_fragments": 2,
                        "fragment_size": 150
                    }
                }
            },
            "_source" : [
                "id", "title", "content", "file_type", "language", "tags", "created_at", "github_stars", "view_count", "popularity_score", "freshness_score", "tags", "source"
            ]
        }
        )
        knn_s.source([
                "id", "title", "content", "file_type", "language", "tags", "created_at", "github_stars", "view_count", "popularity_score", "freshness_score", "tags", "source"
            ])
        s = s.query("multi_match", 
                query=query,
                fields=["title^2", "content", "tags"],
                type="best_fields",
                minimum_should_match="75%",
                fuzziness = "AUTO"
                )
        knn_s = knn_s.knn(
            field="embedding",
            k = candidates,
            num_candidates= candidates*2,
            query_vector=self.embedding_manager.generate_query_embedding(query=query).tolist()
        )
        text_query = {
                "multi_match":{
                    "query": query,
                    "fields": ["title^2", "content", "tags"],
                    "type": "best_fields",
                    "minimum_should_match": "75%",
                    "fuzziness": "AUTO"
                    }
                }
        
        knn_query = {
            "field": "embedding",
            "query_vector" : self.embedding_manager.generate_query_embedding(query=query).tolist(),
            "k": candidates,
            "num_candidates":candidates*2
        }

        if es_filters:
            text_query = {
                "bool": {
                    "must": text_query,
                    "filter": es_filters
                }
            }
            knn_query["filter"] = es_filters
        

        text = s.to_dict()
        knn_query = knn_s.to_dict()
        
        return [text, knn_query]
        

    def simple_search(self, query: str, index_name: str = None, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform a simple search on the given index.

        Args:
            query (str): The search query.
            index_name (str): The name of the index to search.

        Returns:
            List[Dict[str, Any]]: A list of search results.
        """
        # Placeholder for search logic
        index_name = index_name or self.default_index
        if index_name not in self.indexes:
            if self.client.index_exists(index_name):
                self.indexes.add(index_name)
            else:
                raise ValueError(f"Index '{index_name}' does not exist.")
        
        if not query or not query.strip():
            logger.warning("Empty search query provided.")
            return self._empty_result(query)
        query = query.strip()
        start_time = datetime.now()

        try:
            formatted_results = []
            queries = self._build_query(query, top_k=top_k)
            for query in queries:
                res = self.client.client.search(
                    index=index_name,
                    body=query,
                    request_timeout=self.default_timeout
                )
                search_time = datetime.now() - start_time
                format_res = self._parse_es_response(res, query, search_time)
                formatted_results.append(format_res)
                logger.info(f"Search Completed: {format_res['total_hits']} hits found in {search_time} seconds.")
            
            return self._manual_rrf(formatted_results, rank_constant=40)
        
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
            # return self._empty_result(query)

    def _parse_es_response(self, response: Dict[str, Any], query: str, search_time: datetime) -> Dict[str, Any]:
        """
        Parse ES response and format results for display.
        """
        hits = response.get('hits', {})
        total_hits = hits.get('total', {}).get('value', 0)
        result_list = []

        for hit in hits.get('hits', []):
            source = hit.get('_source', {})
            es_score = hit.get('_score', 0.0)
            highlight = hit.get('highlight', {})
            result = {
                "id": source.get("id"),
                "title": self._extract_highlighted_text(highlight.get('title'), source.get('title')),
                "content":  self._create_content_snippet(highlight.get('content'), source.get('content')),
                "score": round(es_score, 2),
                "metadata": self._extract_metadata(source)
            }
            result_list.append(result)

        return {
            "query": query,
            "search_time": search_time,
            "total_hits": total_hits,
            "results": result_list
        }
    
    def _extract_highlighted_text(self, highlights: List[str], default_text: str) -> str:
        """
        Extract highlighted text or return default text if no highlights are found.
        """
        if highlights:
            return highlights[0]
        return default_text

    def _create_content_snippet(self, highlights: List[str], default_content: str) -> str:
        """
        Create a content snippet from highlights or return default content if no highlights are found.
        """
        if highlights:
            return " [...] ".join(highlights)
        
        return default_content[:150] + "..." if len(default_content) > 150 else default_content
    
    def _extract_metadata(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from the source document.
        """
        return {
            "file_type": source.get("file_type"),
            "language": source.get("language"),
            "tags": source.get("tags", []),
            "created_at": source.get("created_at"),
            "github_stars": source.get("github_stars", 0),
            "view_count": source.get("view_count", 0),
            "popularity_score": round(source.get("popularity_score", 0.0), 3),
            "freshness_score": round(source.get("freshness_score", 0.0), 3),
            "source": source.get("source")
        }
    
    def _empty_result(self, query: str) -> Dict[str, Any]:
        """
        Return an empty result structure when no results are found or an error occurs.
        """
        return {
            "query": query,
            "search_time": 0,
            "total_hits": 0,
            "results": [],
            "error": "No results found or an error occurred."
        }
    
    def _error_result(self, query: str, error_message: str) -> Dict[str, Any]:
        """
        Return an error result structure with the provided error message.
        
        Args:
            query (str): The search query that caused the error.
            error_message (str): The error message to include in the result.
        
        Returns:
            Dict[str, Any]: A dictionary containing the error information.
        """
        return {
            "query": query,
            "search_time": 0,
            "total_hits": 0,
            "results": [],
            "error": error_message
        }

    def _manual_rrf(self, results: List[Dict[str, Any]], rank_constant: int=40) -> Dict[str, Any]:
        if not results:
            return {}
        returning_dict = results[-1].copy()
        fused_scores = {}

        for result in results:
            result_list = result["results"]
            for rank, doc in enumerate(result_list):
                doc_id = doc.get("id")
                rrf_score_contribution = 1.0 / (rank_constant + (rank +1))

                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {
                        "doc": doc,
                        "rrf_score":0.0
                    } 
                
                fused_scores[doc_id]['rrf_score'] += rrf_score_contribution
        
        combined_results = []
        for doc_id, data in fused_scores.items():
            data["doc"]["score"] = data["rrf_score"]
            data["doc"]["original_score_type"] = "rrf_manual"
            combined_results.append(data["doc"])

        combined_results.sort(key=lambda x: x["score"], reverse=True)
        returning_dict["results"] = combined_results
        return returning_dict

    ### Method for CLI ###
    def search(self, query: str, index_name: str = None, search_type: Literal["simple"] = "simple", **kwargs) -> RerankingResult:
        if search_type == "simple":
            start = datetime.now()
            results = self.simple_search(query, index_name=index_name, **kwargs)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        search_time = datetime.now() - start
        return self.reranker.rerank(query=query, results=results.get("results", []), top_k=kwargs.get("top_k", 10), search_time=search_time, no_rerank=kwargs.get("no_rerank", False))

    def explain_results(self, results: RerankingResult, query: str):
        print(self.reranker.explain_reranking(reranking_result=results, query=query))

    #### Quick and Advanced Tests ####
    def quick_test(self) -> None:
        """
        Quick test to verify the search engine functionality.
        """
        try:
            logger.info("Running quick test...")
            results = self.simple_search("FastAPI")
            if results['total_hits'] > 0:
                logger.info(f"Quick test passed with {results['total_hits']} hits.")
            else:
                logger.warning("Quick test returned no results.")
            
            top_result = results["results"][0] if results["results"] else None
            print(f"\nTop Result: \nTitle: {top_result['title']}\nContent: {top_result['content']}\nScore: {top_result['score']}\nMetadata: {top_result['metadata']}")
            print(f'\nOthers : {results["results"][1:]}')
        except Exception as e:
            logger.error(f"Quick test failed: {e}")
            raise
        
        finally:
            logger.info("Quick test completed.")

    def advanced_test(self):
        logger.info("Running advanced test...")
        test_cases = [
        # Test 1: Filtre par type
        {
            "name": "Documentation only",
            "query": "Docker",
            "filters": {"file_type": "documentation"},
            "expected": "Should find only official Docker docs"
        },
        
        # Test 2: Documents récents
        {
            "name": "Recent content",
            "query": "Python",
            "boost": {"recent": 2.0},
            "expected": "Should prioritize recent Python content"
        },
        
        # Test 3: Tri par popularité
        {
            "name": "Popular frameworks",
            "query": "web framework",
            "filters": {},
            "sort_by": "popularity",
            "expected": "Should sort by GitHub stars"
        },
        
        # Test 4: Combinaison filtres
        {
            "name": "Popular recent docs",
            "query": "authentication",
            "filters": {
                "file_type": ["documentation", "tutorial"]
            },
            "boost": {
                "recent": 1.5,
                "popularity": 2.0,
                "official_docs": 3.0
            },
            "expected": "Popular auth docs with official boost"
        },
        
        # Test 5: Filtre langue
        {
            "name": "French content",
            "query": "authentification",
            "filters": {"language": "fr"},
            "expected": "Should find French authentication content"
        }
    ]
    
        for i, test in enumerate(test_cases, 1):
            print(f"Test {i}: {test['name']}")
            print(f"Query: '{test['query']}'")
            print(f"Filters: {test.get('filters', {})}")
            
            results = self.simple_search(
                query=test['query'],
                filters=test.get('filters', {}),
                sort_by=test.get('sort_by', 'relevance'),
                boost =test.get('boost', {}),
                size=3
            )
            
            print(f"Results: {results['total_hits']} hits")
            if results['results']:
                top_result = results['results'][0]
                print(f"Top result: {top_result['title']}")
                print(f"Metadata: Type={top_result['metadata']['file_type']}, "
                    f"Stars={top_result['metadata']['github_stars']}, "
                    f"Lang={top_result['metadata']['language']}")
            
            print(f"Expected: {test['expected']}")
            print("-" * 60)

    def semantic_test_suite(self) -> None:
        """Test semantic search capabilities."""
        test_cases = [
            {
                "query": "web framework",
                "expected": "Should find FastAPI, Flask, Django content"
            },
            {
                "query": "deploy container",
                "expected": "Should find Docker, Kubernetes deployment guides"  
            },
            {
                "query": "secure API",
                "expected": "Should find authentication, JWT, security docs"
            },
            {
                "query": "authentification utilisateur",  # FR
                "expected": "Should find EN authentication content too"
            }
        ]
        
        for test in test_cases:
            print(f"\n🧠 Semantic test: '{test['query']}'")
            

            semantic_results = self.simple_search(test['query'], top_k=3)
            print(f"   Semantic hits: {semantic_results['total_hits']}")
            
            if semantic_results['results']:
                print(f"   Top semantic result: {semantic_results['results'][0]['title']}")
            
            print(f"   Expected: {test['expected']}")

if __name__ == "__main__":
    print("Testing SearchEngine...")
    search_engine = SearchEngine()
    search_engine.quick_test()
    search_engine.advanced_test()
    search_engine.semantic_test_suite()
    print("SearchEngine tests completed.")