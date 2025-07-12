from pymilvus import connections, utility, Collection
import pandas as pd

def connect_to_milvus(host: str = "localhost", port: str = "19530") -> None:
    """Connect to Milvus server."""
    try:
        connections.connect(host=host, port=port)
        print(f"✅ Connected to Milvus at {host}:{port}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        return False

def list_all_collections():
    """List all collections and their basic info."""
    try:
        collections = utility.list_collections()
        if not collections:
            print("No collections found in Milvus.")
            return []
        
        print(f"\n📊 Found {len(collections)} collections:")
        print("=" * 80)
        
        # Print just the collection names first
        print("\nCollection Names:")
        print("-" * 40)
        for i, col_name in enumerate(collections, 1):
            print(f"{i}. {col_name}")
        
        # Ask user if they want to inspect a specific collection
        while True:
            choice = input("\nEnter collection number to inspect (or 'all' for all, 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                return []
                
            if choice == 'all':
                collections_to_inspect = collections
                break
                
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(collections):
                    collections_to_inspect = [collections[idx]]
                    break
                else:
                    print("❌ Invalid collection number. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number, 'all', or 'q'.")
        
        collection_info = []
        
        for col_name in collections_to_inspect:
            try:
                print("\n" + "=" * 80)
                print(f"🔍 Inspecting collection: {col_name}")
                print("=" * 80)
                
                col = Collection(col_name)
                col.load()
                
                # Get number of entities
                num_entities = col.num_entities
                
                # Get schema info
                schema = col.schema
                fields = []
                searchable_fields = []
                
                print(f"\n📋 Schema for collection: {col_name}")
                print(f"Number of entries: {num_entities:,}")
                
                # First, collect all field info
                for field in schema.fields:
                    field_info = {
                        'name': field.name,
                        'type': str(field.dtype),
                        'is_primary': field.is_primary,
                        'auto_id': getattr(field, 'auto_id', False),
                        'description': getattr(field, 'description', '')
                    }
                    
                    # Add field params if they exist
                    if hasattr(field, 'params'):
                        field_info['params'] = field.params
                        
                        # Check if this is a searchable field (not embedding or ID)
                        if field.name not in ['id', 'embedding']:
                            searchable_fields.append(field.name)
                    
                    fields.append(field_info)
                
                # Print searchable fields first
                print("\n🔍 Searchable Fields:")
                print("-" * 40)
                if searchable_fields:
                    for i, field_name in enumerate(searchable_fields, 1):
                        print(f"{i}. {field_name}")
                else:
                    print("No searchable fields found (only ID and embedding fields exist).")
                
                # Then print all fields with details
                print("\n📝 All Fields:")
                print("-" * 40)
                for field in fields:
                    field_str = f"- {field['name']}: {field['type']}"
                    if field['is_primary']:
                        field_str += " (Primary Key)"
                    if field['auto_id']:
                        field_str += " [Auto ID]"
                    
                    print(field_str)
                    
                    # Print field description if available
                    if field['description']:
                        print(f"  Description: {field['description']}")
                    
                    # Print field parameters if they exist
                    if 'params' in field and field['params']:
                        print("  Parameters:")
                        for k, v in field['params'].items():
                            print(f"    - {k}: {v}")
                
                # Get a sample document if available
                if num_entities > 0 and searchable_fields:
                    try:
                        # Try to get sample data from searchable fields
                        sample = col.query(
                            expr=f"id in [1, {min(10, num_entities)}]",
                            output_fields=searchable_fields[:5]  # First 5 searchable fields
                        )
                        
                        if sample:
                            print("\n📄 Sample Data from Searchable Fields:")
                            print("-" * 40)
                            for i, doc in enumerate(sample[:2]):  # Show up to 2 samples
                                print(f"\nDocument {i+1}:")
                                for k, v in doc.items():
                                    val = str(v)
                                    if len(val) > 100:
                                        val = val[:100] + '... [truncated]'
                                    print(f"  {k}: {val}")
                    except Exception as e:
                        print(f"\n⚠️  Could not retrieve sample data: {str(e)[:200]}")
                
                # Add to collection info
                collection_info.append({
                    'name': col_name,
                    'num_entities': num_entities,
                    'fields': fields,
                    'searchable_fields': searchable_fields
                })
                
            except Exception as e:
                print(f"\n❌ Error inspecting collection {col_name}: {str(e)[:200]}")
            
            print("\n" + "=" * 80 + "\n")
        
        return collection_info
        
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        return []

def get_collection_schema(collection_name: str) -> None:
    """Get detailed schema information for a specific collection."""
    try:
        col = Collection(collection_name)
        col.load()
        
        print(f"\n📋 Schema for collection: {collection_name}")
        print("=" * 80)
        
        print(f"\n📊 Collection Stats:")
        print(f"- Number of entities: {col.num_entities:,}")
        
        print("\n🔍 Fields:")
        print("-" * 80)
        
        schema = col.schema
        for field in schema.fields:
            print(f"\nField Name: {field.name}")
            print(f"- Data Type: {field.dtype}")
            print(f"- Is Primary Key: {field.is_primary}")
            print(f"- Auto ID: {getattr(field, 'auto_id', 'N/A')}")
            
            # For vector fields, show dimension
            if hasattr(field, 'params') and 'dim' in field.params:
                print(f"- Dimension: {field.params['dim']}")
            
            # For variable-length fields, show max length if available
            if hasattr(field, 'params') and 'max_length' in field.params:
                print(f"- Max Length: {field.params['max_length']}")
        
        # Show index information
        print("\n📈 Indexes:")
        print("-" * 40)
        indexes = col.indexes
        for idx in indexes:
            print(f"\nIndex on: {', '.join(idx.field_name)}")
            print(f"- Index Type: {idx.params['index_type']}")
            print(f"- Metric Type: {idx.params['metric_type']}")
            print(f"- Params: {idx.params['params']}")
        
        # Show partition information
        print("\n🗂️ Partitions:")
        print("-" * 40)
        partitions = col.partitions
        for p in partitions:
            print(f"- {p.name}: {p.num_entities:,} entities")
            
    except Exception as e:
        print(f"❌ Error getting schema for {collection_name}: {e}")

def main():
    # Connect to Milvus
    if not connect_to_milvus():
        return
    
    # List all collections
    collections = utility.list_collections()
    
    if not collections:
        print("No collections found in Milvus.")
        return
    
    while True:
        # Display menu
        print("\n" + "=" * 60)
        print("MILVUS COLLECTION INSPECTOR".center(60))
        print("=" * 60)
        
        # List all collections with numbers
        print("\nAvailable Collections:")
        print("-" * 40)
        for i, col_name in enumerate(collections, 1):
            try:
                col = Collection(col_name)
                col.load()
                num_entities = col.num_entities
                print(f"{i:2d}. {col_name:<40} ({num_entities:>5,} entities)")
            except Exception as e:
                print(f"{i:2d}. {col_name:<40} (error loading)")
        
        # Get user choice
        print("\nOptions:")
        print("  [number]  - Inspect collection by number")
        print("  l         - Reload collections list")
        print("  q         - Quit")
        
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == 'q':
            print("\n👋 Exiting...")
            break
            
        if choice == 'l':
            collections = utility.list_collections()
            if not collections:
                print("No collections found in Milvus.")
                return
            continue
            
        try:
            # Try to parse as a number
            idx = int(choice) - 1
            if 0 <= idx < len(collections):
                col_name = collections[idx]
                print(f"\n{'='*60}")
                print(f"INSPECTING COLLECTION: {col_name}".center(60))
                print(f"{'='*60}")
                get_collection_schema(col_name)
                
                # Ask if user wants to see sample data
                try:
                    col = Collection(col_name)
                    col.load()
                    if col.num_entities > 0:
                        view_sample = input("\nView sample data? (y/N): ").strip().lower()
                        if view_sample == 'y':
                            sample_size = min(3, col.num_entities)
                            print(f"\n📄 Sample Data (first {sample_size} records):")
                            print("-" * 60)
                            
                            # Get field names (excluding embedding)
                            fields = [f.name for f in col.schema.fields if f.name != 'embedding']
                            
                            # Get sample data
                            results = col.query(
                                expr=f"id >= 1",
                                offset=0,
                                limit=sample_size,
                                output_fields=fields
                            )
                            
                            for i, doc in enumerate(results, 1):
                                print(f"\n📝 Document {i}:")
                                for k, v in doc.items():
                                    val = str(v)
                                    if len(val) > 100:
                                        val = val[:100] + '... [truncated]'
                                    print(f"  {k}: {val}")
                except Exception as e:
                    print(f"\n⚠️  Could not retrieve sample data: {str(e)[:200]}")
                
                input("\nPress Enter to continue...")
            else:
                print("❌ Invalid collection number.")
        except ValueError:
            print("❌ Please enter a valid number, 'l' to reload, or 'q' to quit.")

if __name__ == "__main__":
    main()
