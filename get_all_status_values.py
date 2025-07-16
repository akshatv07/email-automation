from pymilvus import connections, Collection, utility


def connect_to_milvus(host: str = "localhost", port: str = "19530"):
    try:
        connections.connect(alias="default", host=host, port=port)
        print(f"✅ Connected to Milvus at {host}:{port}")
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        raise

def print_collection_fields():
    connect_to_milvus()
    collections = utility.list_collections()
    if not collections:
        print("No collections found in Milvus.")
        return
    for col_name in collections:
        try:
            collection = Collection(col_name)
            collection.load()
            # Get all field names except 'subject' and 'email_body'
            all_fields = [f.name for f in collection.schema.fields if f.name not in ['subject', 'email_body']]
            fields_str = ', '.join(all_fields) if all_fields else '(none)'
            print(f"Collection: {col_name}")
            print(f"  Fields: {fields_str}\n")
        except Exception as e:
            print(f"❌ Error processing collection '{col_name}': {e}")

def main():
    print_collection_fields()

if __name__ == "__main__":
    main() 