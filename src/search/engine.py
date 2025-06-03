"""
Module for search engine functionality.
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from src.data import client
from .embeddings import EmbeddingManager

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
        
        es_query = self._build_simple_query(query, **kwargs)
        complex_keywords = ["boolean", "filters", "sort_by", "agg", "rerank", "boost"]
        if any(keyword in kwargs for keyword in complex_keywords):
            logger.info("Building complex query.")
            es_query = self._build_complex_query(es_query, **kwargs)
        return es_query
    
    def _build_simple_query(self, query: str, **kwargs) -> Dict[str, Any]:
        top_k = kwargs.get("top_k", None)
        return {
                "query": {
                    "multi_match":{
                        "query": query,
                        "fields": ["title^2", "content", "tags"],
                        "type": "best_fields",
                        "minimum_should_match": "75%",
                        "fuzziness": "AUTO"
                    }
                },
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

    def _build_complex_query(self, es_query: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Extend the base query with additional parameters like filters, sorting, etc.
        
        Args:
            es_query (Dict[str, Any]): The base search query.
            **kwargs: Additional parameters to extend the query.
        
        Returns:
            Dict[str, Any]: The extended search query.
        """
        base_query = es_query.get("query", {})

        bool_query = {
            "bool": {
                "must": base_query,
                "filter": [],
            }
        }

        # Add filters if provided
        filters = kwargs.get("filters", {})
        if filters:
            self._add_metadata_filters(bool_query["bool"]["filter"], filters)
            self._add_date_filters(bool_query["bool"]["filter"], filters)

        boost = kwargs.get("boost", None)
        if boost:
            self._add_boosting(bool_query["bool"], boost)
        
        es_query["query"] = bool_query

        sorting = kwargs.get("sort", None)
        if sorting:
            self._add_sorting(es_query, sorting)
        
        return es_query
    
    def _add_metadata_filters(self, filter_list: List[Dict[str, Any]], filters: Dict[str, Any]) -> None:
        """
        Add metadata filters to the filter list.
        
        Args:
            filter_list (List[Dict[str, Any]]): The list to add filters to.
            filters (Dict[str, Any]): The filters to apply.
        """
        for key, value in filters.items():
            if key in ["file_type", "language", "tags", "source"]:
                if isinstance(value, str):
                    value = [value]

                filter_list.append({
                    "terms": {key: value}
                })
        
    
    def _add_date_filters(self, filter_list: List[Dict[str, Any]], filters: Dict[str, Any]) -> None:
        """
        Add date filters to the filter list.
        
        Args:
            filter_list (List[Dict[str, Any]]): The list to add filters to.
            filters (Dict[str, Any]): The filters to apply.
        """
        if "date_range" in filters:
            date_range = filters["date_range"]
            if isinstance(date_range, dict) and "gte" in date_range and "lte" in date_range:
                filter_list.append({
                    "range": {
                        "created_at": {
                            "gte": date_range["gte"],
                            "lte": date_range["lte"]
                        }
                    }
                })
        
    def _add_boosting(self, bool_query: Dict[str, Any], boost: Dict[str, float]) -> None:
        """
        Add boosting to the query.
        
        Args:
            bool_query (Dict[str, Any]): The boolean query to modify.
            boost (Dict[str, float]): The boosting parameters.
                    boost structure should be like: {key_field: boost_value, ...}
        """
        if boost is None or len(boost) == 0:
            logger.warning("No boosting parameters provided, skipping boosting.")
            return
        bool_query.setdefault("should", [])
        boost_queries = bool_query["should"]
        for field, boost_value in boost.items():
            if field == "official_docs":
                boost_queries.append({
                    "term": {
                        "source": {
                            "value": "official_docs",
                            "boost": boost_value
                        }
                    }
                })
            elif field == "recent":
                boost_queries.append({
                    "range": {
                        "freshness_score": {
                            "gte": 0.5,
                            "boost": boost_value
                        }
                    }
                })
            
            elif field == "popularity":
                boost_queries.append({
                    "range": {
                        "popularity_score": {
                            "gte": 0.5,
                            "boost": boost_value
                        }
                    }
                })
        
    def _add_sorting(self, es_query: Dict[str, Any], sort_by: str, order: Optional[Literal["desc", "asc"]] = None) -> None:
        """
        Add sorting to the search query.
        
        Args:
            es_query (Dict[str, Any]): The search query to modify.
            sort_by (str): The sorting field.
            order (str): The order of sorting, either 'asc' or 'desc'.
        """
        if not sort_by:
            logger.warning("No sorting parameters provided, skipping sorting.")
            return
        
        if not order:
            order = "desc"
        key = None  # Default sorting key
        match sort_by:
            case "popularity":
                key = "popularity_score"
            case "freshness":
                key = "freshness_score"
            case "recent":
                key = "created_at"
            case "stars":
                key = "github_stars"
        if key:
            es_query["sort"] = [
                {key : {"order": order}},
                {"_score": {"order": "desc"}}
            ]
        logger.info(f"Sorting added: {key} in {order} order.")
            

    def simple_search(self, query: str, index_name: str = None, top_k: int = None) -> List[Dict[str, Any]]:
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
            es_query = self._build_query(query, top_k=top_k)

            res = self.client.client.search(
                index=index_name,
                body=es_query,
                request_timeout=self.default_timeout
            )
            search_time = datetime.now() - start_time
            formatted_results = self._parse_es_response(res, query, search_time)
            logger.info(f"Search Completed: {formatted_results['total_hits']} hits found in {search_time} seconds.")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return self._empty_result(query)

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

    def advanced_search(self, query: str, index_name: str = None, filters: Dict[str, Any] = None, sort_by: str = "relevance", order: Optional[Literal["desc", "asc"]] = None, size: int = 10, boost: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Perform an advanced search on the given index with optional filters.

        Args:
            query (str): The search query.
            index_name (str): The name of the index to search.
            filters (Dict[str, Any], optional): Filters to apply to the search.

        Returns:
            List[Dict[str, Any]]: A list of search results.
        """
        # Placeholder for advanced search logic
        query = query.strip() if query else ""
        if not query:
            logger.warning("Empty search query provided for advanced search.")
            return self._empty_result(query)
        index_name = index_name or self.default_index

        if index_name not in self.indexes:
            if self.client.index_exists(index_name):
                self.indexes.add(index_name)
            else:
                raise ValueError(f"Index '{index_name}' does not exist.")
        start_time = datetime.now()
        try:
            query = self._build_query(query, index_name=index_name, filters=filters, sort=sort_by, order=order, top_k=size, boost=boost)
            res = self.client.client.search(
                index=index_name,
                body=query,
                request_timeout=self.default_timeout
            )
            search_time = datetime.now() - start_time
            formatted_results = self._parse_es_response(res, query, search_time)
            logger.info(f"Advanced Search Completed: {formatted_results['total_hits']} hits found in {search_time} seconds.")
            return formatted_results
        except Exception as e:
            logger.error(f"Error during advanced search: {e}")
            return self._error_result(query, str(e))
    

    ### Semantic Search Methods ###
    def semantic_search(self, 
                   query: str, 
                   index_name: str = None, 
                   top_k: int = 10,
                   hybrid_weight: float = 0.5) -> Dict[str, Any]:
        """
        Perform semantic search using embeddings + BM25 hybrid approach.
        
        Args:
            query: Search query
            index_name: ES index name
            top_k: Number of results
            hybrid_weight: Weight for vector vs BM25 (0.5 = 50/50)
            
        Returns:
            Search results with semantic ranking
        """
        index_name = index_name or self.default_index
        
        if not query or not query.strip():
            return self._empty_result(query)
        
        query = query.strip()
        start_time = datetime.now()
        
        logger.info(f"🧠 Semantic search: '{query}'")
        
        try:
            query_embedding = self.embedding_manager.generate_embedding(query)
            

            bm25_query = self._build_simple_query(query, top_k=top_k).get("query", {})

            es_query = {
                "query": {
                    "function_score": {
                        "query": bm25_query,
                        "functions": [
                            {
                                "script_score": {
                                    "script": {
                                        "source": f"""
                                        double vectorScore = cosineSimilarity(params.query_vector, 'embedding') + 1.0;
                                        double bm25Score = _score;
                                        return {hybrid_weight} * vectorScore + {1-hybrid_weight} * bm25Score;
                                        """,
                                        "params": {"query_vector": query_embedding.tolist()}
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": top_k,
                "_source": [
                    "id", "title", "content", "file_type", "language",
                    "created_at", "github_stars", "popularity_score", 
                    "freshness_score", "tags", "source"
                ]
            }
            

            response = self.client.client.search(
                index=index_name,
                body=es_query,
                request_timeout=self.default_timeout
            )
            
            search_time = datetime.now() - start_time
            formatted_results = self._parse_es_response(response, query, search_time)
            formatted_results['search_type'] = 'semantic_hybrid'
            formatted_results['hybrid_weight'] = hybrid_weight
            
            logger.info(f"🧠 Semantic search completed: {formatted_results['total_hits']} hits")
            return formatted_results
        
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return self._error_result(query, str(e))


    ### Method for CLI ###
    def search(self, query: str, index_name: str = None, search_type: Literal["simple", "advanced", "semantic"] = "simple", **kwargs) -> Dict[str, Any]:
        if search_type == "simple":
            return self.simple_search(query, index_name=index_name, **kwargs)
        elif search_type == "advanced":
            return self.advanced_search(query, index_name=index_name, **kwargs)
        elif search_type == "semantic":
            return self.semantic_search(query, index_name=index_name, **kwargs)
        else:
            raise ValueError(f"Unknown search type: {search_type}")

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
            
            results = self.advanced_search(
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
            

            semantic_results = self.semantic_search(test['query'], top_k=3)
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