#!/usr/bin/env python3
"""
Simple Milvus Collection Structure Viewer
Shows structure of a specific collection
"""

from pymilvus import connections, Collection
import config.settings as settings

def show_collection_info(collection_name):
    """Show detailed information about a specific collection"""
    
    try:
        # Connect to Milvus
        connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        print("✅ Connected to Milvus successfully")
        
        # Load collection
        print(f"📦 Loading collection: {collection_name}")
        collection = Collection(collection_name)
        collection.load()
        
        # Get schema
        schema = collection.schema
        fields = schema.fields
        
        print(f"\n📋 Collection: {collection_name}")
        print("=" * 60)
        print(f"Description: {schema.description}")
        print(f"Total Fields: {len(fields)}")
        print("-" * 60)
        
        # Show all fields
        print("🔍 Field Structure:")
        for i, field in enumerate(fields, 1):
            print(f"\n{i}. {field.name}")
            print(f"   Type: {field.dtype}")
            print(f"   Primary Key: {field.is_primary}")
            print(f"   Auto ID: {field.auto_id}")
            if field.description:
                print(f"   Description: {field.description}")
        
        # Show sample data
        print(f"\n📄 Sample Data (first 2 records):")
        print("-" * 60)
        
        try:
            field_names = [f.name for f in fields]
            sample_data = collection.query(
                expr="",
                output_fields=field_names,
                limit=2
            )
            
            if sample_data:
                for i, record in enumerate(sample_data, 1):
                    print(f"\n📄 Record {i}:")
                    for key, value in record.items():
                        if key == 'embedding':
                            if isinstance(value, list):
                                print(f"  {key}: [Vector with {len(value)} dimensions]")
                            else:
                                print(f"  {key}: [Vector data]")
                        else:
                            display_value = str(value)
                            if len(display_value) > 150:
                                display_value = display_value[:150] + "..."
                            print(f"  {key}: {display_value}")
            else:
                print("   No data found in collection")
                
        except Exception as e:
            print(f"   Error retrieving sample data: {e}")
        
        # Field categorization
        print(f"\n🔍 Field Categories:")
        text_fields = [f.name for f in fields if f.dtype in ['VarChar', 'String']]
        vector_fields = [f.name for f in fields if f.dtype in ['FloatVector', 'BinaryVector']]
        other_fields = [f.name for f in fields if f.dtype not in ['VarChar', 'String', 'FloatVector', 'BinaryVector']]
        
        print(f"   Text Fields: {text_fields}")
        print(f"   Vector Fields: {vector_fields}")
        print(f"   Other Fields: {other_fields}")
        
        print(f"\n✅ Collection analysis completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # You can change this to any collection name you want to analyze
    collection_name = "predisbursal_loan_query_loan_ca_data"
    show_collection_info(collection_name) 