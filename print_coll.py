from pymilvus import connections, Collection, utility
import pandas as pd
import config.settings as settings
from tabulate import tabulate
import os

def connect_to_milvus():
    """Connect to Milvus database"""
    try:
        connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        print("✅ Connected to Milvus successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        return False

def get_collection_info(collection_name):
    """Get detailed information about a collection"""
    try:
        collection = Collection(collection_name)
        collection.load()
        
        # Get schema information
        schema = collection.schema
        field_names = [field.name for field in schema.fields]
        
        # Get row count (alternative to get_statistics)
        try:
            row_count = collection.num_entities
        except:
            row_count = "Unknown"
        
        return {
            'name': collection_name,
            'fields': field_names,
            'row_count': row_count,
            'collection': collection
        }
    except Exception as e:
        print(f"❌ Error loading collection '{collection_name}': {e}")
        return None

def display_collection_data(collection_info, limit=10, save_csv=False):
    """Display collection data in a pretty table format"""
    collection_name = collection_info['name']
    collection = collection_info['collection']
    field_names = collection_info['fields']
    row_count = collection_info['row_count']
    
    print(f"\n{'='*80}")
    print(f"📦 Collection: {collection_name}")
    print(f"📋 Fields: {', '.join(field_names)}")
    print(f"📊 Row Count: {row_count}")
    print(f"{'='*80}")
    
    try:
        # Query data
        results = collection.query(
            expr="id >= 0", 
            output_fields=field_names, 
            limit=limit
        )
        
        if not results:
            print("❌ No data found in collection")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Display pretty table
        print(f"\n📋 Sample Data (showing {len(df)} records):")
        print("-" * 80)
        
        # Create a display-friendly DataFrame
        display_df = df.copy()
        
        # Truncate long text fields for display
        for col in display_df.columns:
            if col != 'embedding':
                display_df[col] = display_df[col].astype(str).apply(
                    lambda x: x[:50] + "..." if len(x) > 50 else x
                )
            else:
                display_df[col] = display_df[col].apply(
                    lambda x: f"[Vector: {len(x)} dims]"
                )
        
        # Display using tabulate for better formatting
        print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
        
        # Show full data for first few records
        print(f"\n🔍 Full Data (first 3 records):")
        print("-" * 80)
        
        for i, record in enumerate(results[:3]):
            print(f"\n--- Record {i+1} ---")
            for key, value in record.items():
                if key == 'embedding':
                    print(f"  {key}: [Vector with {len(value)} dimensions]")
                else:
                    # Show full text for metadata fields
                    display_value = str(value)
                    if len(display_value) > 200:
                        display_value = display_value[:200] + "..."
                    print(f"  {key}: {display_value}")
        
        # Save to CSV if requested
        if save_csv:
            # Get current working directory
            current_dir = os.getcwd()
            csv_filename = f"{collection_name}_data.csv"
            csv_path = os.path.join(current_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            print(f"\n💾 Saved full data to: {csv_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error querying collection '{collection_name}': {e}")
        return None

def main():
    """Main function to list and display all collections"""
    print("🔍 Milvus Collection Viewer")
    print("=" * 50)
    
    # Connect to Milvus
    if not connect_to_milvus():
        return
    
    # List all collections
    try:
        all_collections = utility.list_collections()
        print(f"\n📦 Total Collections in Milvus: {len(all_collections)}")
        
        if not all_collections:
            print("❌ No collections found in Milvus")
            return
        
        print(f"📋 Collections: {', '.join(all_collections)}")
        
        # Process each collection
        all_data = {}
        
        for collection_name in all_collections:
            collection_info = get_collection_info(collection_name)
            if collection_info:
                df = display_collection_data(collection_info, limit=10, save_csv=True)
                if df is not None:
                    all_data[collection_name] = df
        
        # Summary
        print(f"\n{'='*80}")
        print("📊 SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Successfully processed {len(all_data)} collections")
        
        for name, df in all_data.items():
            print(f"  - {name}: {len(df)} records")
        
        if all_data:
            print(f"\n💾 CSV files saved in current directory:")
            current_dir = os.getcwd()
            print(f"   📁 Directory: {current_dir}")
            for name in all_data.keys():
                csv_file = f"{name}_data.csv"
                csv_path = os.path.join(current_dir, csv_file)
                if os.path.exists(csv_path):
                    file_size = os.path.getsize(csv_path)
                    print(f"   📄 {csv_file} ({file_size} bytes)")
        else:
            print("❌ No collections were successfully processed")
        
        print("🔍 Use the verification functions in test_ingest.py for detailed analysis")
        
    except Exception as e:
        print(f"❌ Error listing collections: {e}")

if __name__ == "__main__":
    main() 