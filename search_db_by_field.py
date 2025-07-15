import argparse
from pymilvus import Collection, connections, utility
from sentence_transformers import SentenceTransformer
import json
from numpy import dot
from numpy.linalg import norm


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def main():
    parser = argparse.ArgumentParser(description="Vector search Milvus collection by subject+email body and return closest match's subject, email_body, and a metadata field.")
    parser.add_argument('--collection', type=str, required=True, help='Collection name to search in')
    parser.add_argument('--subject', type=str, required=False, help='Subject to search for (optional)')
    parser.add_argument('--body', type=str, required=False, help='Email body to search for (optional)')
    parser.add_argument('--metadata', type=str, required=True, help='Metadata field name to return')
    parser.add_argument('--top-k', type=int, default=1, help='Number of top results to return (default: 1)')
    parser.add_argument('--json', action='store_true', help='Output results as JSON in email_responder.py-compatible format')
    parser.add_argument('--output', type=str, help='Output file path for JSON output')
    args = parser.parse_args()

    # Connect to Milvus
    connections.connect(alias="default", host="localhost", port="19530")

    # Check if collection exists
    if not utility.has_collection(args.collection):
        error_msg = f"Collection '{args.collection}' not found."
        if args.json and args.output:
            output_dict = {'error': error_msg}
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json.dumps(output_dict, ensure_ascii=False, indent=2))
            print(f"JSON error written to {args.output}")
        else:
            print(error_msg)
        return

    # Load collection
    collection = Collection(args.collection)
    collection.load()

    # Check if metadata field exists
    field_names = [f.name for f in collection.schema.fields]
    if args.metadata not in field_names:
        error_msg = f"Field '{args.metadata}' not found in collection '{args.collection}'. Available fields: {field_names}"
        if args.json and args.output:
            output_dict = {'error': error_msg}
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json.dumps(output_dict, ensure_ascii=False, indent=2))
            print(f"JSON error written to {args.output}")
        else:
            print(error_msg)
        return

    # Find vector field
    vector_fields = [f for f in collection.schema.fields if f.dtype == 101]
    if not vector_fields:
        print("No vector field found in collection.")
        return
    vector_field = vector_fields[0]
    expected_dim = vector_field.params.get('dim', 384)

    # Generate embedding for search
    model = SentenceTransformer('all-MiniLM-L6-v2')
    subject = args.subject.strip() if args.subject else ''
    body = args.body.strip() if args.body else ''
    if subject and body:
        query_text = subject + ' ' + body
    elif subject:
        query_text = subject
    elif body:
        query_text = body
    else:
        print("At least one of subject or body must be provided.")
        return
    embedding = model.encode([query_text])[0].tolist()
    if len(embedding) != expected_dim:
        if len(embedding) > expected_dim:
            embedding = embedding[:expected_dim]
        else:
            embedding += [0.0] * (expected_dim - len(embedding))

    # Prepare output fields
    output_fields = ["subject", "email_body", args.metadata]

    # Search
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(
        data=[embedding],
        anns_field=vector_field.name,
        param=search_params,
        limit=args.top_k,
        output_fields=output_fields
    )
    # Convert results to list for compatibility
    if not isinstance(results, list):
        if hasattr(results, '__iter__') or hasattr(results, '__getitem__'):
            try:
                results = list(results)
            except Exception:
                results = []
        else:
            results = []
    # Print results
    if not isinstance(results, list) or len(results) == 0:
        no_match_msg = f"No matching records found in collection '{args.collection}' for field '{args.metadata}'."
        if args.json and args.output:
            output_dict = {'error': no_match_msg}
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json.dumps(output_dict, ensure_ascii=False, indent=2))
            print(f"JSON error written to {args.output}")
        else:
            print(no_match_msg)
        return
    hits = results[0]
    if hasattr(hits, '__iter__') and not isinstance(hits, list):
        try:
            hits = list(hits)
        except Exception:
            hits = []
    if not isinstance(hits, list) or len(hits) == 0:
        no_match_msg = f"No matching records found in collection '{args.collection}' for field '{args.metadata}'."
        if args.json and args.output:
            output_dict = {'error': no_match_msg}
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json.dumps(output_dict, ensure_ascii=False, indent=2))
            print(f"JSON error written to {args.output}")
        else:
            print(no_match_msg)
        return
    for i, hit in enumerate(hits, 1):
        entity = hit.get('entity', {})
        metadata_value = entity.get(args.metadata, '')
        # Prepare the texts
        result_subject = entity.get('subject', '')
        result_body = entity.get('email_body', '')
        # For similarity, use the same logic as above
        result_text = (result_subject + ' ' + result_body).strip()
        query_emb = model.encode([query_text])[0]
        result_emb = model.encode([result_text])[0]
        similarity = dot(query_emb, result_emb) / (norm(query_emb) * norm(result_emb))
        matchness = similarity * 100
        print(f"{args.metadata}: {metadata_value}")
        print(f"Similarity Score: {matchness:.2f}% ")
        print()

    if args.json:
        # Prepare metadata
        metadata = {
            'collection': args.collection,
            'query_subject': args.subject,
            'query_body': args.body,
            'top_k': args.top_k,
            'results_count': len(hits)
        }
        # Prepare results in expected format
        results = []
        for i, hit in enumerate(hits, 1):
            entity = hit.get('entity', {})
            result = {
                'rank': i,
                'id': hit.get('id', None),
                'distance': getattr(hit, 'distance', None) if not isinstance(hit, dict) else hit.get('distance', None),
                'fields': entity
            }
            results.append(result)
        output_dict = {
            'metadata': metadata,
            'results': results
        }
        output_json = json.dumps(output_dict, ensure_ascii=False, indent=2)
        if args.output:
            # Always overwrite the output file with new results
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_json)
            print(f"JSON results written to {args.output}")
        else:
            print(output_json)
        return

if __name__ == "__main__":
    main() 