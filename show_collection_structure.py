#!/usr/bin/env python3
"""
Milvus Collection Structure Viewer
Shows detailed information about a collection's schema and sample data
"""

from pymilvus import connections, Collection
import config.settings as settings

def connect_to_milvus():
    """Connect to Milvus database"""
    try:
        connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        print("✅ Connected to Milvus successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        return False

def list_all_collections():
    """List all available collections in Milvus"""
    try:
        from pymilvus import utility
        collections = utility.list_collections()
        return collections
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        return []

def show_collection_structure(collection_name):
    """
    Display detailed structure of a Milvus collection
    
    Args:
        collection_name (str): Name of the collection to analyze
    """
    try:
        print(f"📦 Analyzing collection: {collection_name}")
        print("=" * 60)
        
        # Load collection
        collection = Collection(collection_name)
        collection.load()
        
        # Get schema information
        schema = collection.schema
        fields = schema.fields
        
        print(f"📋 Collection Schema:")
        print(f"   Collection Name: {collection_name}")
        print(f"   Description: {schema.description}")
        print(f"   Number of Fields: {len(fields)}")
        print("-" * 60)
        
        # Display field details
        print("🔍 Field Details:")
        for i, field in enumerate(fields, 1):
            print(f"\n  {i}. Field: {field.name}")
            print(f"     Type: {field.dtype}")
            print(f"     Primary Key: {field.is_primary}")
            print(f"     Auto ID: {field.auto_id}")
            print(f"     Description: {field.description}")
            
            # Show additional info for vector fields
            if hasattr(field, 'params') and field.params:
                print(f"     Parameters: {field.params}")
        
        # Get collection statistics
        print("\n📊 Collection Statistics:")
        try:
            stats = collection.get_statistics()
            print(f"   Row Count: {stats['row_count']}")
        except Exception as e:
            print(f"   Row Count: Unable to retrieve ({e})")
        
        # Show sample data
        print("\n📄 Sample Data (first 3 records):")
        print("-" * 60)
        
        try:
            # Get all field names for output
            field_names = [f.name for f in fields]
            
            # Query sample data
            sample_data = collection.query(
                expr="",  # Empty expression to get all records
                output_fields=field_names,
                limit=3
            )
            
            if sample_data:
                for i, record in enumerate(sample_data, 1):
                    print(f"\n📄 Record {i}:")
                    print("-" * 30)
                    for key, value in record.items():
                        if key == 'embedding':
                            # Show vector info without displaying the full vector
                            if isinstance(value, list):
                                print(f"  {key}: [Vector with {len(value)} dimensions]")
                            else:
                                print(f"  {key}: [Vector data]")
                        else:
                            # Show text data (truncate if too long)
                            display_value = str(value)
                            if len(display_value) > 100:
                                display_value = display_value[:100] + "..."
                            print(f"  {key}: {display_value}")
            else:
                print("   No data found in collection")
                
        except Exception as e:
            print(f"   Error retrieving sample data: {e}")
        
        # Show field analysis
        print("\n🔍 Field Analysis:")
        print("-" * 60)
        
        text_fields = []
        vector_fields = []
        other_fields = []
        
        for field in fields:
            if field.dtype in ['VarChar', 'String']:
                text_fields.append(field.name)
            elif field.dtype in ['FloatVector', 'BinaryVector']:
                vector_fields.append(field.name)
            else:
                other_fields.append(field.name)
        
        print(f"   Text Fields ({len(text_fields)}): {text_fields}")
        print(f"   Vector Fields ({len(vector_fields)}): {vector_fields}")
        print(f"   Other Fields ({len(other_fields)}): {other_fields}")
        
        # Suggest common field mappings
        print("\n💡 Field Mapping Suggestions:")
        print("-" * 60)
        
        subject_field = next((f for f in text_fields if 'subject' in f.lower()), None)
        status_field = next((f for f in text_fields if 'status' in f.lower()), None)
        email_body_field = next((f for f in text_fields if any(pattern in f.lower() for pattern in ['body', 'content', 'message', 'email']), None)
        
        if subject_field:
            print(f"   Subject field: {subject_field}")
        if status_field:
            print(f"   Status field: {status_field}")
        if email_body_field:
            print(f"   Email body field: {email_body_field}")
        
        if not any([subject_field, status_field, email_body_field]):
            print("   No common field patterns detected")
        
        print("\n✅ Collection analysis completed!")
        
    except Exception as e:
        print(f"❌ Error analyzing collection '{collection_name}': {e}")

def main():
    """Main function with interactive collection selection"""
    print("🔍 Milvus Collection Structure Viewer")
    print("=" * 50)
    
    # Connect to Milvus
    if not connect_to_milvus():
        return
    
    # List available collections
    collections = list_all_collections()
    if not collections:
        print("❌ No collections found or error listing collections")
        return
    
    print(f"\n📋 Available Collections ({len(collections)}):")
    for i, collection in enumerate(collections, 1):
        print(f"   {i}. {collection}")
    
    print("\n" + "=" * 50)
    
    # Get user input
    while True:
        user_input = input("Enter collection name (or 'list' to see collections again): ").strip()
        
        if user_input.lower() == 'list':
            print(f"\n📋 Available Collections ({len(collections)}):")
            for i, collection in enumerate(collections, 1):
                print(f"   {i}. {collection}")
            continue
        
        if not user_input:
            print("❌ Please enter a collection name")
            continue
        
        if user_input in collections:
            show_collection_structure(user_input)
            break
        else:
            print(f"❌ Collection '{user_input}' not found")
            print(f"Available collections: {collections}")

if __name__ == "__main__":
    main() 