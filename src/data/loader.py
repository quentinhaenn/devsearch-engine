"""
Data loader for the project.

Expected behavior:
- Load data from JSON -- until new data source is implemented.
- Process the data to extract relevant fields.
- Validate data integrity.
- Fill scores for metadata fields like freshness and popularity.
- Store the processed data in the Elasticsearch index.
"""

import json
import logging
from math import exp
from typing import List, Dict, Any
from .client import ElasticSearchClient
from .config import ElasticSearchConfig

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Base class for loading and processing data.
    """

    def __init__(self):
        self.config = ElasticSearchConfig()
        self.client = ElasticSearchClient()
        self.index_name = self.config.INDEX_NAME
        self.client.ping()
        logger.info(f"Connected to ElasticSearch at {self.config.URL}")
    
    def load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries representing the data.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                logger.info(f"Loaded {len(data)} records from {file_path}")
                return data
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return []
        
    def validate_document(self, document: Dict[str, Any]) -> bool:
        """
        Validate the structure and required fields of a document.
        
        Args:
            document (Dict[str, Any]): Document to validate.
        
        Returns:
            bool: True if valid, False otherwise.
        """
        required_fields = ['id', 'title', 'content', 'created_at', "language", "tags", "source"]
        for field in required_fields:
            if field not in document:
                logger.error(f"Missing required field: {field} in document {document.get('id')}")
                return False
        # Validate some specific fields
        if not document.get("modified_at"):
            document["modified_at"] = document.get("created_at", "")
        return True
    
    def calculate_scores(self, document: Dict[str, Any]) -> bool:
        """
        Calculate scores for metadata fields like freshness and popularity.

        Depends on `calculate_freshness` and `calculate_popularity` methods.
        Args:
            document (Dict[str, Any]): Document to calculate scores for.
        
        Returns:
            bool: True if scores are calculated successfully, False otherwise.
        """
        return self.calculate_freshness(document) and self.calculate_popularity(document)
    
    def bulk_index(self, documents: List[Dict[str, Any]], batch_size: int = 1000) -> Dict[str, Any]:
        """
        Bulk index documents into Elasticsearch.
        Returns a dictionary with stats for indexing.
        Args:
            documents (List[Dict[str, Any]]): List of documents to index.
            batch_size (int): Number of documents to index in each batch.
        Returns:
            Dict[str, Any]: Dictionary with indexing stats.
        """
        stats = {
            "total": len(documents),
            "indexed": 0,
            "failed": 0,
            "errors": []
        }
        logger.info(f"Starting bulk indexing of {len(documents)} documents in batches of {batch_size}")

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_number = i // batch_size + 1
            logger.info(f"Indexing batch {batch_number} with {len(batch)} documents")
            bulk_body = []
            for doc in batch:
                action = {
                    "index": {
                        "_index": self.index_name,
                        "_id": doc.get('id', None)  # Use document ID if available
                    }
                }

                bulk_body.append(action)
                bulk_body.append(doc)

            try:
                response = self.client.client.bulk(body=bulk_body)
                for item in response['items']:
                    if "index" in item and item["index"].get("status") in [200, 201]:
                        stats["indexed"] += 1
                    else:
                        stats["failed"] += 1
                        error_info = {
                            "document_id": item["index"].get("_id"),
                            "status": item["index"].get("status"),
                            "error": item["index"].get("error", {}).get("reason", "Unknown error")
                        }
                        stats["errors"].append(error_info)
                        logger.error(f"Failed to index document {item['index'].get('_id')}: {error_info}")
                logger.info(f"Batch {batch_number} indexed successfully. Indexed: {stats['indexed']}, Failed: {stats['failed']}")
            except Exception as e:
                logger.error(f"Error indexing batch {batch_number}: {e}")
                stats["failed"] += len(batch)
                stats["errors"].append({
                    "batch_number": batch_number,
                    "error": str(e),
                    "document_affected": len(batch)
                })                

        logger.info(f"Bulk indexing completed. Total: {stats['total']}, Indexed: {stats['indexed']}, Failed: {stats['failed']}")

        return stats
    
    def load_and_index(self, file_path: str, 
                       force_recreate: bool = False,
                       batch_size: int= 100) -> Dict[str, Any]:
        """
        Pipeline to load data from a JSON file, validate, process and index it into Elasticsearch.
        Args:
            file_path (str): Path to the JSON file.
            force_recreate (bool): Whether to force recreate the index.
            batch_size (int): Number of documents to index in each batch.
        Returns:
            Dict[str, Any]: Dictionary with indexing stats.
        """
        pipeline_stats = {
            "start_time": None,
            "end_time": None,
            "documents_loaded": 0,
            "documents_processed": 0,
            "documents_indexed": 0,
            "processing_errors": [],
            "indexing_stats": {}
        }
        import time
        pipeline_stats["start_time"] = time.time()
        logger.info(f"Starting data loading and indexing from {file_path}")

        try:
            logger.info("STEP 1: Preparing ES index")
            if not self.client.create_index(self.index_name, force_recreate=force_recreate):
                logger.error(f"Failed to create or connect to index {self.index_name}. Exiting.")
                return {"error": "Failed to create or connect to index."}
            logger.info(f"Index {self.index_name} is ready for indexing.")

            logger.info("STEP 2: Loading data from JSON file")
            raw_documents = self.load_json(file_path=file_path)
            pipeline_stats["documents_loaded"] = len(raw_documents)
            if not raw_documents:
                logger.error("No documents loaded from the JSON file.")
                return {"error": "No documents loaded from the JSON file."}
            
            logger.info("STEP 3: Validating and processing documents")
            processed_documents = []
            for doc in raw_documents:
                try:
                    processed_document= self.prepare_for_indexing(doc)
                    if processed_document:
                        processed_documents.append(processed_document)
                        pipeline_stats["documents_processed"] += 1
                except Exception as e:
                    logger.error(f"Error processing document {doc.get('id')}: {e}")
                    pipeline_stats["processing_errors"].append({
                        "document_id": doc.get('id'),
                        "error": str(e)
                    })
            
            logger.info("STEP 4: Indexing processed documents")
            if processed_documents:
                indexing_stats = self.bulk_index(processed_documents, batch_size=batch_size)
                pipeline_stats["documents_indexed"] = indexing_stats.get("indexed", 0)
                pipeline_stats["indexing_stats"] = indexing_stats
            else:
                logger.warning("No documents to index after processing.")

            logger.info("STEP 5: Finalizing pipeline")
            self.client.client.indices.refresh(index=self.index_name)
            actual_count = int(self.client.client.cat.count(index=self.index_name, format="json")[0]['count'])
            logger.info(f"Index {self.index_name} now contains {actual_count} documents.")

            if actual_count != pipeline_stats["documents_indexed"]:
                logger.warning(f"Discrepancy in indexed documents: expected {pipeline_stats['documents_indexed']}, actual {actual_count}.")
        
        except Exception as e:
            logger.error(f"Unexpected error during data loading and indexing: {e}")
            pipeline_stats["processing_errors"].append({
                "error": str(e)
            })
        
        finally:
            pipeline_stats["end_time"] = time.time()
            elapsed_time = pipeline_stats["end_time"] - pipeline_stats["start_time"]
            logger.info(f"Data loading and indexing completed in {elapsed_time:.2f} seconds.")
            logger.info(f"Total documents loaded: {pipeline_stats['documents_loaded']}, Processed: {pipeline_stats['documents_processed']}, Indexed: {pipeline_stats['documents_indexed']}")
            logger.info(f"Processing errors: {len(pipeline_stats['processing_errors'])}, Indexing stats: {pipeline_stats['indexing_stats']}")
            if pipeline_stats.get("indexing_stats"):
                failed = pipeline_stats["indexing_stats"].get("failed", 0)
                if failed > 0:
                    logger.warning(f"Indexing completed with {failed} failures. Check errors for details.")
        
        return pipeline_stats

    def calculate_freshness(self, document: Dict[str, Any]) -> bool:
        """
        Calculate freshness score based on the created and modified dates.

        Actual implementation uses exponential decay to calculate freshness. Further improvements should be considered after making the all search engine work.
        
        Args:
            document (Dict[str, Any]): Document to calculate freshness for.
        
        Returns:
            bool: True if freshness score is calculated successfully, False otherwise.
        """
        # Placeholder for actual implementation
        created_at = document.get('created_at')
        modified_at = document.get('modified_at')
        if not created_at or not modified_at:
            logger.error(f"Missing freshness fields in document {document.get('id')}")
            return False
        # Parse dates
        import datetime as dt
        modified_date = dt.datetime.fromisoformat(modified_at.replace('Z', '+00:00'))
        today = dt.datetime.now(dt.timezone.utc)
        # Calculate freshness score based on modified date
        age_in_days = (today - modified_date).days
        if age_in_days < 0:
            logger.error(f"Modified date {modified_at} is in the future for document {document.get('id')}")
            return False
        # Exponential decay function for freshness
        freshness_score = exp(-age_in_days/365)  # Decay over a year
        document['freshness_score'] = freshness_score
        logger.info(f"Calculated freshness score {freshness_score} for document {document.get('id')}")
        return True


    def calculate_popularity(self, document: Dict[str, Any]) -> bool:
        """
        Calculate popularity score based on view count and GitHub stars.

        Actual implementation uses a weighted sum of view count, GitHub stars and other factors like if the document is an official repository or documentation.
        
        Args:
            document (Dict[str, Any]): Document to calculate popularity for.

        Returns:
            bool: True if popularity score is calculated successfully, False otherwise.
        """
        stars = document.get('github_stars', 0)
        view_count = document.get('view_count', 0)
        if stars is None or view_count is None:
            logger.error(f"Missing popularity fields in document {document.get('id')}")
            return False
        # Normalize and calculate scores
        normalized_stars = self._normalize_stars(stars)
        normalized_views = self.normalize_view_count(view_count)
        popularity_score = (0.7 * normalized_stars) + (0.3 * normalized_views)
        document['popularity_score'] = popularity_score
        logger.info(f"Calculated popularity score {popularity_score} for document {document.get('id')}")
        return True

    def prepare_for_indexing(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the document for indexing in Elasticsearch.
        
        Args:
            document (Dict[str, Any]): Document to prepare.
        
        Returns:
            Dict[str, Any]: Prepared document.
        """
        if not self.validate_document(document):
            logger.error(f"Document {document.get('id')} is invalid.")
            return {}
        
        # Calculate scores
        self.calculate_scores(document)
        
        return document
    
    def _normalize_stars(self, github_stars: int) -> float:
        """
        Normalize Github stars count to a score between 0 and 1. Maximum currently considered is 10000 stars.
        Args:
            github_stars (int): Number of GitHub stars.
        Returns:
            float: Normalized popularity score.
        """
        max_stars = 10000
        return min(github_stars / max_stars, 1.0)
    
    def normalize_view_count(self, view_count: int) -> float:
        """
        Normalize view count to a score between 0 and 1. Maximum currently considered is 100000 views.
        Args:
            view_count (int): Number of views.
        Returns:
            float: Normalized view count score.
        """
        max_views = 100000
        return min(view_count / max_views, 1.0)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean the text by removing unnecessary characters and formatting like newlines, extra spaces, accented characters, etc.
        Args:
            text (str): Text to clean.
        Returns:
            str: Cleaned text.
        """
        if not text:
            return ""
        # Remove accents and normalize whitespace
        import unicodedata
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        return text.strip().replace('\n', ' ').replace('\r', '').replace('\t', ' ').replace('  ', ' ')
    
    def _parse_date(self, date: str) -> str:
        """
        Parse a date string from ISO8601 format to a standard format for Elasticsearch.
        Args:
            date (str): Date string in ISO8601 format.
        Returns:
            str: Parsed date string in standard format.
        """

        from datetime import datetime
        try:
            parsed_date = datetime.fromisoformat(date.replace('Z', '+00:00'))
            return parsed_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError as e:
            logger.error(f"Error parsing date {date}: {e}")
            return ""
    

    def _quick_test(self, sample_size=3) -> None:
        """
        Quick test to check if the DataLoader is working correctly.
        """
        logger.info("Running quick test for DataLoader...")
        sample_file = "sample_data/sample_dataset.json"

        try:
            docs = self.load_json(sample_file)
            if len(docs) > sample_size:
                docs = docs[:sample_size]
        
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix=".json") as temp_file:
                json.dump(docs, temp_file, indent=2)
                temp_file_path = temp_file.name

            stats = self.load_and_index(temp_file_path, force_recreate=True, batch_size=10)

            import os
            os.unlink(temp_file_path)  # Clean up temporary file

            if stats['documents_indexed'] == len(docs):
                logger.info("Quick test passed: All documents indexed successfully.")
            else:
                logger.error(f"Quick test failed: Expected {len(docs)} documents indexed, but got {stats['documents_indexed']}.")
                logger.error(f"Indexing stats: {stats['indexing_stats']}")
        except Exception as e:
            logger.error(f"Error during quick test: {e}")
            raise e
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting DataLoader quick test...")
    data_loader = DataLoader()
    data_loader._quick_test(sample_size=3)
    logger.info("DataLoader quick test completed.")