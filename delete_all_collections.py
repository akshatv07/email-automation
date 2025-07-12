from pymilvus import connections, utility
import sys

def connect_to_milvus(host: str = "localhost", port: str = "19530") -> bool:
    """Connect to Milvus server."""
    try:
        connections.connect(host=host, port=port)
        print(f"✅ Connected to Milvus at {host}:{port}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        return False

def list_collections() -> list:
    """List all collections in Milvus."""
    try:
        collections = utility.list_collections()
        if not collections:
            print("No collections found in Milvus.")
            return []
        
        print("\n📋 Collections in Milvus:")
        print("=" * 50)
        
        for i, col_name in enumerate(collections, 1):
            try:
                num_entities = Collection(col_name).num_entities
                print(f"{i}. {col_name} (entities: {num_entities:,})")
            except:
                print(f"{i}. {col_name} (unable to get entity count)")
        
        return collections
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        return []

def delete_collections(collections: list, skip_confirmation: bool = False) -> None:
    """Delete all collections with confirmation."""
    if not collections:
        print("No collections to delete.")
        return
    
    print("\n⚠️  WARNING: This will delete the following collections:")
    for col in collections:
        print(f"   - {col}")
    
    if not skip_confirmation:
        confirm = input("\nAre you sure you want to delete ALL collections? This cannot be undone. (y/N): ")
        if confirm.lower() != 'y':
            print("\nOperation cancelled. No collections were deleted.")
            return
    
    print("\nDeleting collections...")
    success_count = 0
    
    for col_name in collections:
        try:
            # Drop the collection
            utility.drop_collection(col_name)
            print(f"✅ Deleted collection: {col_name}")
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to delete collection {col_name}: {e}")
    
    print(f"\n🎉 Successfully deleted {success_count} out of {len(collections)} collections.")

def main():
    # Check for --force flag
    skip_confirmation = '--force' in sys.argv
    
    # Connect to Milvus
    if not connect_to_milvus():
        sys.exit(1)
    
    # List all collections
    collections = list_collections()
    
    if not collections:
        print("No collections to delete.")
        sys.exit(0)
    
    # Delete collections with confirmation
    delete_collections(collections, skip_confirmation)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MILVUS COLLECTION DELETION TOOL".center(60))
    print("=" * 60)
    print("\nThis tool will delete ALL collections from your Milvus database.")
    print("This action cannot be undone.\n")
    
    main()
    
    print("\n" + "=" * 60)
    print("OPERATION COMPLETE".center(60))
    print("=" * 60)
