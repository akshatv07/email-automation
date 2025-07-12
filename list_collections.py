from pymilvus import connections, utility

def list_milvus_collections(host: str = "localhost", port: str = "19530"):
    """
    List all collections in the Milvus database.
    
    Args:
        host: Milvus server host
        port: Milvus server port
    """
    try:
        # Connect to Milvus
        connections.connect(host=host, port=port)
        print(f"✅ Connected to Milvus at {host}:{port}")
        
        # Get list of all collections
        collections = utility.list_collections()
        
        if not collections:
            print("No collections found in the database.")
            return
            
        print("\n📂 Collections in Milvus:")
        for i, collection_name in enumerate(sorted(collections), 1):
            print(f"{i}. {collection_name}")
            
        print(f"\nTotal collections: {len(collections)}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        # Close the connection
        connections.disconnect(host)
        print("\n🔌 Disconnected from Milvus")

if __name__ == "__main__":
    list_milvus_collections()
