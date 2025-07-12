from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
import argparse
from typing import Dict, List, Optional

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def connect_to_milvus(host: str = "localhost", port: str = "19530") -> bool:
    """Connect to Milvus server."""
    try:
        connections.connect(alias="default", host=host, port=port)
        print(f"✅ Connected to Milvus at {host}:{port}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        return False

def search_emails(
    email_body: str,
    subject: str,
    category: str,
    status: str = "",
    top_k: int = 3
) -> List[Dict]:
    """
    Search for emails in Milvus based on email body, subject, category, and status.
    
    Args:
        email_body: The email body text to search for
        subject: The email subject to search for
        category: The collection/category to search in
        status: Status to filter by (optional)
        top_k: Number of results to return
        
    Returns:
        List of matching email documents with scores
    """
    # Generate embeddings for both subject and body
    subject_embedding = model.encode([subject])[0].tolist()
    body_embedding = model.encode([email_body])[0].tolist()
    
    # Check if collection exists
    if not utility.has_collection(category):
        print(f"❌ Collection '{category}' not found")
        return []
    
    # Load the collection
    collection = Collection(category)
    collection.load()
    
    # Build search parameters
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    }
    
    # Build filter expression if status is provided
    filter_expr = None
    if status:
        # Check if status field exists in the collection
        status_field = None
        for field in collection.schema.fields:
            if field.name.lower() == status.lower():
                status_field = field.name
                break
        
        if status_field:
            filter_expr = f"{status_field} == '{status}'"
    
    # Output fields to retrieve
    output_fields = ["email_body", "subject"]
    if status:
        output_fields.append(status)
    
    # Search in the collection - search in both subject and body
    subject_results = collection.search(
        data=[subject_embedding],
        anns_field="subject_embedding",  # Assuming this is the field name for subject embeddings
        param=search_params,
        limit=top_k,
        expr=filter_expr,
        output_fields=output_fields
    )
    
    body_results = collection.search(
        data=[body_embedding],
        anns_field="email_body_embedding",  # Assuming this is the field name for body embeddings
        param=search_params,
        limit=top_k,
        expr=filter_expr,
        output_fields=output_fields
    )
    
    # Combine and deduplicate results
    combined_results = {}
    
    # Process subject results
    for hits in subject_results:
        for hit in hits:
            if hit.id not in combined_results:
                combined_results[hit.id] = {
                    "id": hit.id,
                    "subject_score": 1 - hit.distance,
                    "body_score": 0,
                    "entity": hit.entity._row_data
                }
    
    # Process body results and combine scores
    for hits in body_results:
        for hit in hits:
            if hit.id in combined_results:
                combined_results[hit.id]["body_score"] = 1 - hit.distance
            else:
                combined_results[hit.id] = {
                    "id": hit.id,
                    "subject_score": 0,
                    "body_score": 1 - hit.distance,
                    "entity": hit.entity._row_data
                }
    
    # Calculate combined score (weighted average - 40% subject, 60% body)
    for result in combined_results.values():
        result["combined_score"] = (result["subject_score"] * 0.4) + (result["body_score"] * 0.6)
    
    # Sort by combined score
    sorted_results = sorted(combined_results.values(), key=lambda x: x["combined_score"], reverse=True)
    
    return sorted_results[:top_k]

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Search for emails in Milvus using vector similarity")
    parser.add_argument("--email_body", required=True, help="Email body text to search for")
    parser.add_argument("--subject", required=True, help="Email subject to search for")
    parser.add_argument("--category", required=True, help="Category/Collection to search in")
    parser.add_argument("--status", default="", help="Status field name to filter by (optional)")
    parser.add_argument("--status_value", default="", help="Status value to filter by (optional)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to return")
    
    args = parser.parse_args()
    
    # Connect to Milvus
    if not connect_to_milvus():
        return
    
    # Search for emails
    results = search_emails(
        email_body=args.email_body,
        subject=args.subject,
        category=args.category,
        status=args.status_value if args.status else "",
        top_k=args.top_k
    )
    
    # Display results
    if not results:
        print("No matching emails found.")
        return
    
    print(f"\n🔍 Found {len(results)} matching emails:")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n📧 Result {i} (ID: {result['id']})")
        print(f"  Combined Score: {result['combined_score']:.4f}")
        print(f"  Subject Match: {result['subject_score']:.4f}")
        print(f"  Body Match:    {result['body_score']:.4f}")
        
        entity = result['entity']
        if 'subject' in entity:
            print(f"\n  Subject: {entity['subject']}")
        
        if 'email_body' in entity:
            body = entity['email_body']
            if len(body) > 150:
                body = body[:150] + "..."
            print(f"\n  Body: {body}")
        
        # Print status field if it exists
        if args.status and args.status in entity:
            print(f"\n  {args.status.capitalize()}: {entity[args.status]}")

if __name__ == "__main__":
    main()
