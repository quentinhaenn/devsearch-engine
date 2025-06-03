"""
Configuration for ElasticSearch and other data-related settings.
"""
import logging
from typing import Dict, Any
import os
#Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElasticSearchConfig:
    """
    Config class for ES connection and settings.
    """
    ES_ENV = os.getenv("ES_HOST", "localhost")
    HOST: str = ES_ENV.split(":")[0] if ":" in ES_ENV else ES_ENV
    PORT: int = int(ES_ENV.split(":")[1]) if ":" in ES_ENV else 9200  # Default port for ES is 9200
    URL: str = f"http://{HOST}:{PORT}"

    # Timeout settings
    TIMEOUT: int = 30 #global timeout for all requests
    CONNECT_TIMEOUT: int = 10 # connection attempt timeout
    MAX_RETRIES: int = 3 # retries after failed connection attempts. Secure against network issues.

    INDEX_NAME: str = "devsearch_index"
    INDEX_SETTINGS: Dict[str, Any] = {
        "analysis": {
            "filter": {
                "synonym_filter" :{
                    "type":"synonym",
                    "synonyms_path": "synonyms.txt",
                    "updateable": True  # Allows updating synonyms without reindexing
                }
            },
            "analyzer": {
                "standard_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding", "synonym_filter", "stop"]
                }
            },
        },
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "1s",
    }

    # Mapping for the index
    INDEX_MAPPING: Dict[str, Any] = {
        "properties": {
            "id": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "standard",
                "search_analyzer": "standard_analyzer",
                "fields": {
                    "keywords" : {"type": "keyword"}
                }
            },
            "content": {
                "type": "text",
                "analyzer": "standard",
                "search_analyzer": "standard_analyzer"
            },

            # vector embeddings for semantic search
            "embedding": {
                "type": "dense_vector",
                "dims": 384,  # Sentence-BERT base model dimensions. 
                "index": True,
                "similarity": "cosine" # Classic cosine similarity for NLP via embeddings. TODO: test if other similarity metrics work better.
            },
            #Metadata fields
            "file_type": {"type": "keyword"},
            "path": {"type": "keyword"},
            "language": {"type": "keyword"},
            "tags": {"type": "keyword"},
            "source": {"type": "keyword"},

            # Datetime fields
            "created_at": {"type": "date"},
            "modified_at": {"type": "date"},

            # Additional metadata for scoring
            "github_stars": {"type": "integer"},
            "view_count": {"type": "integer"},
            "popularity_score": {"type": "float"},
            "freshness_score": {"type": "float"},
        }
    }