"""
Component 5: Semantic Search + Response Extraction Engine

GOAL:
Given:
- ticket_id
- query_text (subject + email body)
- category_name

Steps:
1. Load data_db.csv
2. Match ticket_id row
3. If 'data_from_IM_pls' has value:
     → status_keys = ['loan_status', 'repayment_status', 'last_stage_checklist']
   Else:
     → status_keys = ['lr_status', 'disbursement_completion_date']

4. Vectorize query_text (use same model as ingestion)
5. Use category_name to select Milvus collection (sheet name sanitized)
6. Do semantic search in Milvus → get top 1 matched row
7. From that row, find first matching key from status_keys
8. Final Output:
   {
       "ticket_id": input_ticket_id,
       "response": response_value_from_milvus_row
   }
"""

import pandas as pd
import numpy as np
import re
import logging
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility
import config.settings as settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _sanitize(name):
    """
    Lowercase, replace non-alphanumeric with underscores, remove leading/trailing underscores.
    """
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


class SemanticResponseEngine:
    """
    Semantic Response Engine for fetching accurate answers from Milvus collections
    based on ticket metadata, category, and query text.
    """
    
    def __init__(self):
        """
        Initialize the Semantic Response Engine.
        - Loads the embedding model
        - Establishes Milvus connection
        - Loads data_db.csv once on initialization
        """
        try:
            # Load embedding model (same as ingestion)
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            self.embedding_dim = settings.EMBEDDING_DIM
            
            # Connect to Milvus
            connections.connect(alias="default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
            logger.info("✅ Milvus connection established")
            
            # Load data_db.csv once on initialization
            self.data_df = pd.read_csv(settings.CSV_DB_FILE)
            logger.info(f"✅ Loaded data_db.csv with {len(self.data_df)} rows")
            
        except Exception as e:
            logger.error(f"❌ Error initializing SemanticResponseEngine: {e}")
            raise
    
    def _get_ticket_row(self, ticket_id: str) -> pd.Series:
        """
        Find the row in data_db.csv that matches the given ticket_id.
        
        Args:
            ticket_id (str): The ticket identifier
            
        Returns:
            pd.Series: The matching row or None if not found
        """
        try:
            # Convert ticket_id to string for consistent matching
            ticket_id = str(ticket_id)
            
            # Try different possible column names for ticket_id
            possible_columns = ['Ticket ID', 'ticket_id']
            
            for col in possible_columns:
                if col in self.data_df.columns:
                    # Find matching row (case-insensitive)
                    row = self.data_df[self.data_df[col].astype(str) == ticket_id]
                    if not row.empty:
                        return row.iloc[0]
            
            logger.warning(f"⚠️ No matching row found for ticket_id: {ticket_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error finding ticket row: {e}")
            return None
    
    def _determine_status_keys(self, ticket_row: pd.Series) -> list:
        """
        Determine which status keys to use based on the ticket row data.
        
        Args:
            ticket_row (pd.Series): The ticket row from data_db.csv
            
        Returns:
            list: List of status keys to search for
        """
        try:
            if ticket_row is None:
                # Default status keys if no row found
                return ['lr_status', 'disbursement_completion_date']
            
            # Check if 'data_from_IM_pls' has a value
            im_value = str(ticket_row.get('data_from_IM_pls', '')).strip()
            
            if im_value and im_value.lower() != 'nan' and im_value:
                # If data_from_IM_pls has value, use these status keys
                status_keys = ['loan_status', 'repayment_status', 'last_stage_checklist']
                logger.info(f"📋 Using IM status keys for ticket_id: {ticket_row.get('Ticket ID', 'Unknown')}")
            else:
                # Otherwise, use these status keys
                status_keys = ['lr_status', 'disbursement_completion_date']
                logger.info(f"📋 Using default status keys for ticket_id: {ticket_row.get('Ticket ID', 'Unknown')}")
            
            return status_keys
            
        except Exception as e:
            logger.error(f"❌ Error determining status keys: {e}")
            return ['lr_status', 'disbursement_completion_date']
    
    def _vectorize_query(self, query_text: str) -> list:
        """
        Vectorize the query text using the same embedding model as ingestion.
        
        Args:
            query_text (str): The text to vectorize
            
        Returns:
            list: Normalized embedding vector
        """
        try:
            # Encode the query text
            embedding = self.model.encode([query_text])
            
            # Normalize the vector (same as ingestion)
            normalized = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            
            return normalized.tolist()[0]
            
        except Exception as e:
            logger.error(f"❌ Error vectorizing query: {e}")
            raise
    
    def _search_milvus_collection(self, collection_name: str, query_vector: list) -> dict:
        """
        Search the Milvus collection for the most similar vector.
        
        Args:
            collection_name (str): Name of the Milvus collection
            query_vector (list): The query vector to search for
            
        Returns:
            dict: The most similar row from Milvus or None if not found
        """
        try:
            # Check if collection exists
            if not utility.has_collection(collection_name):
                logger.warning(f"⚠️ Collection '{collection_name}' does not exist")
                return None
            
            # Load the collection
            collection = Collection(name=collection_name)
            collection.load()
            
            # Search for the most similar vector
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=1,
                output_fields=["*"]
            )
            
            if results and len(results[0]) > 0:
                # Return the first (most similar) result
                result = results[0][0]
                logger.info(f"✅ Found similar result in collection '{collection_name}' with score: {result.score}")
                return result.entity.to_dict()
            else:
                logger.warning(f"⚠️ No similar results found in collection '{collection_name}'")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error searching Milvus collection '{collection_name}': {e}")
            return None
    
    def _extract_response_value(self, milvus_row: dict, status_keys: list) -> str:
        """
        Extract the response value from the Milvus row based on status keys.
        
        Args:
            milvus_row (dict): The row from Milvus
            status_keys (list): List of status keys to search for
            
        Returns:
            str: The response value or default message
        """
        try:
            if milvus_row is None:
                return "No relevant response found."
            
            # Search for the first matching status key
            for key in status_keys:
                if key in milvus_row and milvus_row[key]:
                    value = str(milvus_row[key]).strip()
                    if value and value.lower() != 'nan':
                        logger.info(f"✅ Found response value for key '{key}': {value}")
                        return value
            
            logger.warning(f"⚠️ No matching status keys found in Milvus row")
            return "No relevant response found."
            
        except Exception as e:
            logger.error(f"❌ Error extracting response value: {e}")
            return "No relevant response found."
    
    def get_response(self, ticket_id: str, category_name: str, query_text: str) -> dict:
        """
        Get the semantic response for a given ticket.
        
        Args:
            ticket_id (str): The ticket identifier
            category_name (str): The category name (matches Milvus collection)
            query_text (str): The query text (subject + email body)
            
        Returns:
            dict: Response with ticket_id and response value
        """
        try:
            logger.info(f"🔍 Processing ticket_id: {ticket_id}, category: {category_name}")
            
            # Step 1: Load data_db.csv and find row with ticket_id
            ticket_row = self._get_ticket_row(ticket_id)
            
            # Step 2: Determine status keys based on ticket data
            status_keys = self._determine_status_keys(ticket_row)
            
            # Step 3: Vectorize query_text using same embedding model
            query_vector = self._vectorize_query(query_text)
            
            # Step 4: Search Milvus collection
            collection_name = _sanitize(category_name)
            milvus_row = self._search_milvus_collection(collection_name, query_vector)
            
            # Step 5: Extract response value from the matched row
            response_value = self._extract_response_value(milvus_row, status_keys)
            
            # Final output
            result = {
                "ticket_id": ticket_id,
                "response": response_value
            }
            
            logger.info(f"✅ Generated response for ticket_id: {ticket_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error getting response for ticket_id {ticket_id}: {e}")
            return {
                "ticket_id": ticket_id,
                "response": "No relevant response found."
            }
    
    def __del__(self):
        """
        Cleanup method to close connections.
        """
        try:
            connections.disconnect("default")
            logger.info("✅ Disconnected from Milvus")
        except Exception as e:
            logger.error(f"❌ Error disconnecting from Milvus: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the engine
    engine = SemanticResponseEngine()
    
    # Example usage
    ticket_id = "3633261"
    category_name = "Predisbursal_Loan_Query_IM+_instances"
    query_text = "Payment done but not reflected I have to paid my due in manually but not reflected in loan"
    
    result = engine.get_response(ticket_id, category_name, query_text)
    print(f"Result: {result}") 