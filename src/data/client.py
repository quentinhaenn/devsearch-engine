"""
ES client for managing connections and operations with Elasticsearch instance.
"""

import logging
from typing import Dict, Any, Optional
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, RequestError
from .config import ElasticSearchConfig

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ESSingleton(type):
    """
    Metaclass for singleton pattern to ensure only one instance of ElasticSearchClient exists.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class ElasticSearchClient(metaclass=ESSingleton):
    """
    Client class for managing ElasticSearch connections and operations. Uses Singleton pattern to ensure only one instance exists.
    """
    _instance: Optional["ElasticSearchClient"] = None

    def __init__(self):
        """
        initialize ElasticSearchClient with connection settings, index name, and mapping.

        If the instance already exists, return it instead of creating a new one.
        """
        self.config = ElasticSearchConfig()
        self.client = Elasticsearch(
            [self.config.URL],
            request_timeout=self.config.TIMEOUT,
            sniff_timeout=self.config.CONNECT_TIMEOUT,
            max_retries=self.config.MAX_RETRIES,
            retry_on_timeout=True,
        )

        self._test_connection()
        logger.info(f"Connected to ElasticSearch at {self.config.URL}")
        ElasticSearchClient._instance = self

    def _test_connection(self):
        """
        Test if connection is established by pinging the server.
        """
        try:
            logger.info(f"{self.client.info()}")
            if not self.client.ping():
                raise ConnectionError("Failed to connect to ElasticSearch server.")
            
            info = self.client.info()
            cluster_health = self.client.cluster.health()

            logger.info("Connection established successfully.")
            logger.info(f"Version: {info['version']['number']}")
            logger.info(f"Cluster Name: {info['cluster_name']}")
            logger.info(f"Status: {cluster_health['status']}")
        except ConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise
        except Exception as e :
            logger.error(f" Unexpected error while testing connection to ElasticSearch: {e}")
            raise
    
    def ping(self) -> bool:
        """
        ping the server
        """

        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Error pinging ElasticSearch: {e}")
            return False
    
    def get_cluster_health(self) -> Dict[str, Any]:
        """
        Get the health status of the ElasticSearch cluster.
        """
        try:
            health = self.client.cluster.health()
            logger.info(f"Cluster health: {health}")
            return health
        except RequestError as e:
            logger.error(f"Request error while getting cluster health: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error while getting cluster health: {e}")
            return {}

    def index_exists(self, index_name: Optional[str]) -> bool:
        """
        Check if an index exists.
        """
        index_name = index_name or self.config.INDEX_NAME
        try:
            exists = self.client.indices.exists(index=index_name)
            logger.info(f"index exists: {exists}")
            return exists
        except RequestError as e:
            logger.error(f"Request error while checking index existence: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error while checking index existence: {e}")
            return False
    

    def create_index(self, index_name: Optional[str] = None, mapping: Dict[str, Any] = None, force_recreate: bool = False) -> bool:
        """
        Create an index with the specified mapping and name.
        """
        index_name = index_name or self.config.INDEX_NAME
        try:
            exists = self.index_exists(index_name)
            if exists and not force_recreate:
                logger.info(f"Index '{index_name}' already exists. Use force_recreate=True to recreate it.")
                return False
            
            if exists and force_recreate:
                logger.info(f"deleting index before recreation '{index_name}'")
                self.client.indices.delete(index=index_name)

            index_body = {
                "settings": self.config.INDEX_SETTINGS,
                "mappings": mapping or self.config.INDEX_MAPPING
            }

            self.client.indices.create(index=index_name, body=index_body)
            logger.info(f"Index '{index_name}' recreated successfully.")
            return True
        except RequestError as e:
            logger.error(f"Error creating index '{index_name}': {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error while creating index '{index_name}': {e}")
            return False


    def delete_index(self, index_name: Optional[str] = None) -> bool:
        """
        Delete an index.
        """
        index_name = index_name or self.config.INDEX_NAME
        try:
            if not self.index_exists(index_name):
                logger.info(f"Index '{index_name}' does not exist. No action taken.")
                return False
            
            self.client.indices.delete(index=index_name)
            logger.info(f"Index '{index_name}' deleted successfully.")
            return True
        except RequestError as e:
            logger.error(f"Error deleting index '{index_name}': {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error while deleting index '{index_name}': {e}")
            return False

if __name__ == "__main__":
    print("Testting ElasticSearchClient...")
    try:
        es_client = ElasticSearchClient()
        print("ElasticSearchClient initialized successfully.")
        
        # Test connection
        if es_client.ping():
            print("ElasticSearch is reachable.")
        else:
            print("ElasticSearch is not reachable.")

        # Get cluster health
        health = es_client.get_cluster_health()
        print(f"Cluster Health: {health}")

        # creation/deletion of index for testing
        index_name = "test_index"
        es_client.create_index(index_name=index_name)

        print(f"Index '{index_name}' exists: {es_client.index_exists(index_name)}")

        es_client.delete_index(index_name=index_name)
        print(f"Index '{index_name}' exists after deletion: {es_client.index_exists(index_name)}")

        print("All tests passed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        logger.error(f"An error occurred during testing: {e}")
        raise