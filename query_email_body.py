#!/usr/bin/env python3
"""
Email Body Query Script

This script provides a robust way to query email bodies from a Milvus database
with clear error handling and flexible input options.
"""

from typing import Dict, List, Optional, Union
from pymilvus import connections, Collection
import config.settings as settings

class EmailQueryError(Exception):
    """Custom exception for email query-related errors."""
    pass

class EmailQueryManager:
    """
    Manages email queries to Milvus database with comprehensive error handling.
    """
    def __init__(self, host: str = '', port: str = ''):
        """
        Initialize the query manager with Milvus connection details.
        
        Args:
            host (str): Milvus database host
            port (str): Milvus database port
        """
        self.host = host or settings.MILVUS_HOST
        self.port = port or settings.MILVUS_PORT
        self.connection_status = False

    def connect(self) -> bool:
        """
        Establish connection to Milvus database.
        
        Returns:
            bool: Connection status
        """
        try:
            connections.connect("default", host=self.host, port=self.port)
            self.connection_status = True
            print("✅ Connected to Milvus successfully")
            return True
        except Exception as e:
            print(f"❌ Milvus Connection Error: {e}")
            return False

    def _find_field(self, collection: Collection, field_type: str) -> Optional[str]:
        """
        Find a field in the collection schema based on type.
        
        Args:
            collection (Collection): Milvus collection
            field_type (str): Type of field to find (e.g., 'subject', 'status', 'body')
        
        Returns:
            Optional[str]: Field name if found, None otherwise
        """
        fields = [f.name for f in collection.schema.fields]
        
        # Mapping of field types to possible field name patterns
        field_patterns = {
            'subject': ['subject', 'email_subject'],
            'status': ['status', 'email_status'],
            'body': ['email_body', 'body', 'content', 'message']
        }
        
        for pattern in field_patterns.get(field_type, [field_type]):
            matching_field = next((f for f in fields if pattern in f.lower()), None)
            if matching_field:
                return matching_field
        
        return None

    def query_emails(self, query_params: Dict[str, str]) -> List[Dict]:
        """
        Query emails based on provided parameters.
        
        Args:
            query_params (Dict[str, str]): Dictionary of query parameters
        
        Returns:
            List[Dict]: List of matching email records
        """
        if not self.connection_status:
            raise EmailQueryError("Not connected to Milvus. Call connect() first.")
        
        try:
            # Required parameters
            category = query_params.get('category', '')
            subject = query_params.get('subject', '')
            status = query_params.get('status', '')
            
            if not all([category, subject, status]):
                raise EmailQueryError("Missing required parameters: category, subject, status")
            
            # Load collection
            collection = Collection(category)
            collection.load()
            
            # Find fields dynamically
            subject_field = self._find_field(collection, 'subject')
            status_field = self._find_field(collection, 'status')
            body_field = self._find_field(collection, 'body')
            
            if not all([subject_field, status_field, body_field]):
                raise EmailQueryError(f"Could not find required fields. Available: {[f.name for f in collection.schema.fields]}")
            
            # Build query expression
            expr = f'{subject_field} == "{subject}" and {status_field} == "{status}"'
            
            # Execute query
            results = collection.query(
                expr=expr,
                output_fields=[field for field in [body_field, subject_field, status_field] if field],
                limit=10
            )
            
            return results
        
        except Exception as e:
            print(f"❌ Query Error: {e}")
            return []

    def display_results(self, results: List[Dict], max_body_length: int = 500):
        """
        Display query results in a formatted manner.
        
        Args:
            results (List[Dict]): List of email records
            max_body_length (int): Maximum length of email body to display
        """
        if not results:
            print("⚠️ No matching records found")
            return
        
        print(f"\n✅ Found {len(results)} matching record(s):")
        print("=" * 80)
        
        for i, record in enumerate(results, 1):
            print(f"\n📧 Email {i}:")
            print("-" * 40)
            
            for key, value in record.items():
                if 'body' in key.lower():
                    # Truncate long email bodies
                    display_value = str(value)[:max_body_length] + '...' if len(str(value)) > max_body_length else str(value)
                    print(f"📄 {key}: {display_value}")
                else:
                    print(f"   {key}: {value}")

def main():
    """
    Main function demonstrating usage of EmailQueryManager.
    """
    # Example query parameters
    query_params = {
        'category': 'predisbursal_loan_query_loan_ca',
        'subject': 'loan cancel request',
        'status': 'IMSanctioned'
    }
    
    # Initialize and connect to Milvus
    query_manager = EmailQueryManager()
    if not query_manager.connect():
        return
    
    try:
        # Execute query
        results = query_manager.query_emails(query_params)
        
        # Display results
        query_manager.display_results(results)
    
    except EmailQueryError as e:
        print(f"❌ Query Failed: {e}")

if __name__ == "__main__":
    main() 