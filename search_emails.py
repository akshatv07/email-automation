import json
import logging
import argparse
from datetime import datetime
from typing import Optional, Tuple, Any, Dict, List, Union
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def format_search_results(results: List[Dict], collection_name: str, status_field: str) -> Dict[str, Any]:
    """Format search results into a structured dictionary."""
    formatted = {
        "metadata": {
            "collection": collection_name,
            "status_field": status_field,
            "timestamp": datetime.utcnow().isoformat(),
            "results_count": len(results)
        },
        "results": []
    }
    
    for i, hit in enumerate(results):
        result = {
            "rank": i + 1,
            "id": str(hit.get("id", None)),
            "distance": float(hit.get("distance", 0.0)) if "distance" in hit else None,
            "status": None,
            "fields": {}
        }
        
        # Extract entity data
        entity = hit.get("entity", None)
        if entity and hasattr(entity, '_row_data'):
            for field_name, value in entity._row_data.items():
                result["fields"][field_name] = str(value)
                if field_name.lower() == status_field.lower():
                    result["status"] = str(value)
        elif isinstance(hit, dict):
            for field_name, value in hit.items():
                if field_name not in ["id", "distance", "entity"]:
                    result["fields"][field_name] = str(value)
                    if field_name.lower() == status_field.lower():
                        result["status"] = str(value)
        
        formatted["results"].append(result)
    
    return formatted

def connect_to_milvus(host: str = "localhost", port: str = "19530") -> bool:
    """Connect to Milvus server."""
    try:
        connections.connect(alias="default", host=host, port=port)
        print(f"✅ Connected to Milvus at {host}:{port}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        return False

def get_status_field(collection, status_field_name: str = "") -> Tuple[str, bool]:
    """Find the correct status field name in the collection."""
    # First try exact match
    for field in collection.schema.fields:
        if field.name == status_field_name:
            return field.name, True
    
    # Then try case-insensitive match
    for field in collection.schema.fields:
        if field.name.lower() == status_field_name.lower():
            print(f"Found status field (case-insensitive): {field.name}")
            return field.name, True
    
    # Try to find similar fields
    similar_fields = [f.name for f in collection.schema.fields 
                     if status_field_name.lower() in f.name.lower() or f.name.lower() in status_field_name.lower()]
    
    if similar_fields:
        print(f"⚠️  Status field '{status_field_name}' not found. Similar fields: {', '.join(similar_fields)}")
        return similar_fields[0], True
    
    print(f"❌ Status field '{status_field_name}' not found in collection")
    print("Available fields:", [f.name for f in collection.schema.fields])
    return "", False

def search_emails(
    email_body: str,
    subject: str,
    category: str,
    status_field: str = "",
    top_k: int = 1,
    return_json: bool = False
) -> Union[Dict[str, Any], str, None]:
    """
    Search for the most relevant email and return its status value or full results.
    
    Args:
        email_body: The email body text to search for
        subject: The email subject to search for
        category: The collection/category to search in
        status_field: Name of the status field to return
        top_k: Number of results to consider (default: 1)
        return_json: If True, returns results as JSON string
        
    Returns:
        If return_json is False: The status value from the most relevant matching document
        If return_json is True: JSON string with full search results
        None if no results found or error occurred
    """
    # First check if collection exists
    if not utility.has_collection(category):
        print(f"❌ Collection '{category}' not found")
        return None
    
    # Get collection and its schema
    collection = Collection(category)
    collection.load()
    
    # Get schema information
    schema = collection.schema
    print(f"\n🔍 Collection schema: {schema}")
    
    # Find vector fields
    vector_fields = [f for f in schema.fields if f.dtype == 101]  # 101 is vector type
    if not vector_fields:
        print("❌ No vector field found in collection")
        return None
        
    vector_field = vector_fields[0]
    expected_dim = vector_field.params.get('dim')
    print(f"Using vector field '{vector_field.name}' with dimension {expected_dim}")
    
    # If we couldn't get dimension, use a default
    if not expected_dim:
        expected_dim = 384  # Common default for all-MiniLM-L6-v2
        print(f"⚠️  Using default dimension: {expected_dim}")
    
    # Generate embeddings with the correct dimension
    try:
        subject_embedding = model.encode([subject])[0].tolist()
        body_embedding = model.encode([email_body])[0].tolist()
        
        # Check if we need to adjust dimensions
        if expected_dim and len(subject_embedding) != expected_dim:
            print(f"⚠️  Warning: Embedding dimension mismatch. Expected {expected_dim}, got {len(subject_embedding)}")
            print("Trying to adjust dimensions...")
            # Simple truncation or padding if needed
            if len(subject_embedding) > expected_dim:
                subject_embedding = subject_embedding[:expected_dim]
                body_embedding = body_embedding[:expected_dim]
            else:
                padding = [0.0] * (expected_dim - len(subject_embedding))
                subject_embedding.extend(padding)
                body_embedding.extend(padding)
    except Exception as e:
        print(f"❌ Error generating embeddings: {str(e)}")
        return None
    
    # Get the correct status field name
    status_field_name = status_field
    status_field, has_status = ("", False)
    if status_field_name:
        status_field, has_status = get_status_field(collection, status_field_name)
        if not has_status:
            print(f"⚠️  Status field '{status_field_name}' not found in collection. Proceeding without status field filter.")
            status_field = ""
    else:
        print("ℹ️  No status field specified. Proceeding without status field filter.")
    
    # Prepare output fields - always include subject and email_body
    output_fields = ['subject', 'email_body']
    if status_field:
        output_fields.insert(0, status_field)
    print(f"Will retrieve fields: {output_fields}")
    
    # Search in the collection - search in both subject and body
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    
    try:
        # Use the vector field name from the collection
        vector_field_name = vector_field.name
        print(f"Using vector field '{vector_field_name}' for search")
        
        # Use subject embedding by default
        search_embedding = subject_embedding
        print(f"Searching with embedding dimension: {len(search_embedding)}")
        
        # Search with combined embedding
        try:
            print(f"\n🔍 Executing search with parameters:")
            print(f"- Vector field: {vector_field_name}")
            print(f"- Output fields: {output_fields}")
            print(f"- Top K: {top_k}")
            
            search_results = collection.search(
                data=[search_embedding],
                anns_field=vector_field_name,
                param=search_params,
                limit=top_k,
                output_fields=output_fields
            )
            
            if not isinstance(search_results, list):
                try:
                    search_results = list(search_results)
                except Exception:
                    search_results = []
            # Ensure each result set (hits) is a list
            hits = []
            if search_results and len(search_results) > 0:
                sr0 = search_results[0]
                if hasattr(sr0, '__iter__') and not isinstance(sr0, list):
                    try:
                        hits = list(sr0)
                    except Exception:
                        hits = []
                elif isinstance(sr0, list):
                    hits = sr0
                else:
                    hits = []
            print(f"\n🔍 Search completed. Found {len(hits)} results.")
            
            # Debug: Print raw search results structure
            print("\n�� Raw search results structure:")
            for i, result_set in enumerate(search_results):
                if not isinstance(result_set, list):
                    try:
                        result_set = list(result_set)
                    except Exception:
                        result_set = []
                print(f"  - Result set {i}: {len(result_set)} hits")
                for j, hit in enumerate(result_set):
                    hit_id = hit.get("id", None) if isinstance(hit, dict) else getattr(hit, "id", None)
                    hit_distance = hit.get("distance", None) if isinstance(hit, dict) else getattr(hit, "distance", None)
                    print(f"    - Hit {j}: ID={hit_id}, Distance={hit_distance}")
                    entity = hit.get("entity", None) if isinstance(hit, dict) else getattr(hit, "entity", None)
                    if entity and hasattr(entity, '_row_data'):
                        print(f"      Entity fields: {list(entity._row_data.keys())}")
                    elif isinstance(hit, dict):
                        print(f"      Fields: {list(hit.keys())}")
                    else:
                        print("      No entity data found")
        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            print(f"Vector field name: {vector_field_name}")
            print(f"Embedding dimension: {len(search_embedding) if search_embedding else 'N/A'}")
            raise
        
        # Format the results
        if hits:
            formatted_results = format_search_results(hits, category, status_field if status_field else "")
        else:
            formatted_results = format_search_results([], category, status_field if status_field else "")

        # Check for empty subject and email_body in the top result
        if formatted_results["results"]:
            top_result = formatted_results["results"][0]
            subject = top_result["fields"].get("subject", "")
            email_body = top_result["fields"].get("email_body", "")
            if not subject.strip() and not email_body.strip():
                return "Subject and email body are empty for this result."

        if return_json:
            return json.dumps(formatted_results, indent=2)
        
        # For backward compatibility, return just the status from the top result
        if formatted_results["results"] and formatted_results["results"][0]["status"]:
            return formatted_results["results"][0]["status"]
        
        return None
        
    except Exception as e:
        import traceback
        logger.error("\n❌ Error during vector search:")
        traceback.print_exc()
        
        # Try to get at least some data from the collection
        try:
            logger.info("\n🔍 Attempting to list first few documents...")
            results = collection.query(
                "",  # Empty expression to get all
                output_fields=[status_field, 'subject'],
                limit=3
            )
            
            if results:
                logger.info("\n✅ Found documents in collection:")
                for i, doc in enumerate(results):
                    logger.info(f"\n--- Document {i+1} ---")
                    for key, value in doc.items():
                        # Truncate long values for better readability
                        value_str = str(value)[:100] + "..." if value and len(str(value)) > 100 else str(value)
                        logger.info(f"- {key}: {value_str}")
                        
                    # If we found the status field, return its value
                    if status_field in doc:
                        return doc[status_field]
                
                logger.warning(f"\n⚠️  Status field '{status_field}' not found in the sample documents")
            else:
                logger.error("\n❌ No documents found in the collection")
                
        except Exception as query_error:
            logger.error(f"\n❌ Error querying collection: {str(query_error)}")
            
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Search for similar emails in Milvus.')
    parser.add_argument('--body', type=str, help='Email body to search for')
    parser.add_argument('--subject', type=str, default='', help='Email subject (optional)')
    parser.add_argument('--category', type=str, help='Collection/category to search in')
    parser.add_argument('--status-field', type=str, default='status', help='Name of the status field to return')
    parser.add_argument('--top-k', type=int, default=1, help='Number of results to return (default: 1)')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--output', type=str, help='Output file path (for JSON output)')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Use hardcoded test case 2 (loan under review)
    test_case_2 = {
        'email_body': "Sir mera loan 24 ghante se jada ho gaya under review bata raha h 199 rupay bhi jma kr diya hu",
        'subject': "Loan 24 hours under review",
        'category': "predisbursal_loan_query_credit",
        'status_field': "im_processing",
        'top_k': 10
    }
    
    # Use the test case values if no arguments provided
    email_body = args.body or test_case_2['email_body']
    subject = args.subject or test_case_2['subject']
    category = args.category or test_case_2['category']
    status_field = args.status_field or test_case_2['status_field']
    top_k = args.top_k or test_case_2['top_k']
    
    # Connect to Milvus
    if not connect_to_milvus():
        error_msg = "❌ Could not connect to Milvus. Exiting..."
        if args.json:
            print(json.dumps({"error": error_msg}, indent=2))
        else:
            print(error_msg)
        return
    
    try:
        # Search for similar emails
        result = search_emails(
            email_body=email_body,
            subject=subject,
            category=category,
            status_field=status_field,
            top_k=top_k,
            return_json=args.json
        )
        
        if result is not None:
            if args.json:
                output = result if isinstance(result, str) else json.dumps(result, indent=2)
                if args.output:
                    with open(args.output, 'w') as f:
                        f.write(output)
                    print(f"✅ Results saved to {args.output}")
                else:
                    print(output)
            else:
                if isinstance(result, str):
                    print(f"✅ Status: {result}")
                else:
                    print(f"✅ Status: {result['results'][0]['status'] if result['results'] else 'Not found'}")
        else:
            error_msg = "❌ No matching emails found or status field not found."
            if args.json:
                print(json.dumps({"error": error_msg}, indent=2))
            else:
                print(error_msg)
                
    except Exception as e:
        error_msg = f"❌ An error occurred: {str(e)}"
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(error_msg)
    finally:
        # Disconnect from Milvus
        connections.disconnect("default")

if __name__ == "__main__":
    main()
