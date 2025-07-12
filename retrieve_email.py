from pymilvus import connections, Collection
import argparse
import pandas as pd
from typing import List, Dict, Any

def connect_to_milvus(host: str = "localhost", port: str = "19530") -> None:
    """Connect to Milvus server."""
    try:
        connections.connect(host=host, port=port)
        print(f"✅ Connected to Milvus at {host}:{port}")
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        raise

def list_collections() -> List[str]:
    """List all available collections in Milvus."""
    from pymilvus import utility
    collections = utility.list_collections()
    print("\nAvailable collections:")
    for i, col in enumerate(collections, 1):
        print(f"{i}. {col}")
    return collections

def get_collection_fields(collection_name: str) -> tuple:
    """Get available fields in the collection that might contain subject and content."""
    try:
        collection = Collection(collection_name)
        collection.load()
        
        # Common field name variations
        possible_subject_fields = ['subject', 'Subject', 'SUBJECT', 'title', 'Title', 'TITLE', 'query', 'Query', 'QUERY']
        possible_body_fields = ['email_body', 'body', 'content', 'text', 'message', 'email', 'Email', 'EMAIL']
        
        # Find actual field names in the collection
        subject_field = None
        body_field = None
        
        for field in collection.schema.fields:
            field_lower = field.name.lower()
            if not subject_field and any(f in field_lower for f in ['subject', 'title', 'query']):
                subject_field = field.name
            if not body_field and any(f in field_lower for f in ['body', 'content', 'text', 'message', 'email']):
                body_field = field.name
            
            # If we found both, we can stop searching
            if subject_field and body_field:
                break
        
        return subject_field, body_field, [f.name for f in collection.schema.fields if f.name not in ['id', 'embedding']]
    except Exception as e:
        print(f"❌ Error getting fields for collection {collection_name}: {e}")
        return None, None, []

def search_in_collection(collection_name: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for content in a collection using the best available fields."""
    try:
        # Get the collection fields
        subject_field, body_field, all_fields = get_collection_fields(collection_name)
        if not all_fields:
            return []
        
        print(f"\n🔍 Collection fields: {', '.join(all_fields)}")
        
        # Load the collection
        collection = Collection(collection_name)
        collection.load()
        
        # Create a search vector (dummy vector since we're doing metadata filtering)
        search_vectors = [[0.0] * 384]  # Assuming 384-dim vectors
        
        # Search parameters
        search_params = {
            "metric_type": "L2",
            "offset": 0,
            "ignore_growing": False,
            "params": {"nprobe": 10}
        }
        
        # Build filter expression using available fields
        if subject_field:
            expr = f"{subject_field} like '%{query}%'"
            print(f"   Using field for search: '{subject_field}'")
        else:
            # If no subject field, try to search in any text field
            text_fields = [f for f in all_fields if f.lower() not in ['id', 'embedding']]
            if not text_fields:
                print("❌ No searchable text fields found in the collection")
                return []
                
            # Create OR condition for all text fields
            conditions = [f"{field} like '%{query}%'" for field in text_fields]
            expr = " or ".join(conditions)
            print(f"   Searching in fields: {', '.join(text_fields)}")
        
        # Execute search with filter
        results = collection.search(
            data=search_vectors,
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=all_fields
        )
        
        # Process results
        emails = []
        for hits in results:
            for hit in hits:
                # Get all available fields
                email = {"score": hit.score, "id": hit.id}
                
                # Add all available fields from the result
                for field in all_fields:
                    email[field] = hit.entity.get(field, "N/A")
                
                # Try to identify subject and body if not explicitly found
                if not subject_field and 'subject' not in email:
                    for field in all_fields:
                        if any(f in field.lower() for f in ['subject', 'title', 'query']):
                            email['subject'] = hit.entity.get(field, "No subject")
                            break
                
                if not body_field and 'body' not in email:
                    for field in all_fields:
                        if any(f in field.lower() for f in ['body', 'content', 'text', 'message', 'email']):
                            email['body'] = hit.entity.get(field, "No content")
                            break
                
                emails.append(email)
        
        return emails
        
    except Exception as e:
        print(f"❌ Error searching collection {collection_name}: {e}")
        return []

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Search content in Milvus collections')
    parser.add_argument('--collection', '-c', help='Name of the collection to search in')
    parser.add_argument('--query', '-q', help='Search query (partial match in text fields)')
    parser.add_argument('--list', '-l', action='store_true', help='List all available collections')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of results to return (default: 5)')
    
    args = parser.parse_args()
    
    # Connect to Milvus
    connect_to_milvus()
    
    # List collections if requested
    if args.list:
        list_collections()
        return
    
    # Validate arguments
    if not args.collection or not args.query:
        print("❌ Please provide both collection name and search query")
        print("\nExample usage:")
        print("  python retrieve_email.py -c collection_name -q 'loan status'")
        print("  python retrieve_email.py -l  # to list all collections")
        return
    
    # Search for content
    print(f"\n🔍 Searching in collection: {args.collection}")
    print(f"   Search query: '{args.query}'")
    print(f"   Max results: {args.limit}")
    
    emails = search_in_collection(args.collection, args.query, args.limit)
    
    # Display results
    if not emails:
        print("\n❌ No matching content found.")
        return
    
    print(f"\n✅ Found {len(emails)} matching items:")
    print("-" * 80)
    
    for i, item in enumerate(emails, 1):
        print(f"\n📄 Result {i} (Score: {item['score']:.4f})")
        print("-" * 60)
        
        # Display subject/title if available
        subject = next((item[k] for k in item if 'subject' in k.lower() or 'title' in k.lower() or 'query' in k.lower() 
                       and item[k] not in ['N/A', None]), "No subject/title")
        print(f"Subject/Title: {subject}")
        
        # Display body/content if available
        body = next((item[k] for k in item if ('body' in k.lower() or 'content' in k.lower() or 'text' in k.lower() or 'message' in k.lower())
                     and item[k] not in ['N/A', None]), "")
        print(f"Content: {str(body)[:200]}..." if body else "No content available")
        
        # Print other metadata fields
        print("\n📋 Metadata:")
        for key, value in item.items():
            key_lower = key.lower()
            if key not in ['id', 'score'] and value not in ['N/A', None] and not any(f in key_lower for f in ['subject', 'title', 'body', 'content', 'text', 'message', 'email']):
                print(f"  {key}: {value}")
        
        print("-" * 60)

if __name__ == "__main__":
    main()
