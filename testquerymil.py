from pymilvus import connections, Collection
import re
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

def sanitize_field_name(field_name):
    """Sanitize field name to match Milvus naming conventions"""
    # Remove special characters and replace spaces with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', field_name)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized

def find_field_by_pattern(fields, pattern):
    """Find field that matches the given pattern (case-insensitive)"""
    pattern_lower = pattern.lower()
    for field in fields:
        if pattern_lower in field.lower():
            return field
    return None

def get_collection_schema_info(collection):
    """Get detailed schema information for debugging"""
    try:
        fields = [f.name for f in collection.schema.fields]
        print(f"📋 Available fields in collection: {fields}")
        
        # Show field details for better debugging
        print("🔍 Field details:")
        for i, field in enumerate(collection.schema.fields):
            print(f"  {i+1}. {field.name} (type: {field.dtype})")
        
        return fields
    except Exception as e:
        print(f"❌ Error getting schema info: {e}")
        return []

def query_milvus_collection(category, subject_input, status_input):
    """
    Query Milvus collection with robust field detection and error handling
    
    Args:
        category (str): Collection name
        subject_input (str): Exact subject to match
        status_input (str): Exact status to match
    
    Returns:
        list: Query results or None if error
    """
    try:
        # Connect to Milvus
        if not connect_to_milvus():
            return None
        
        # Load collection
        print(f"📦 Loading collection: {category}")
        collection = Collection(category)
        collection.load()
        
        # Get detailed schema information
        fields = get_collection_schema_info(collection)
        if not fields:
            print("❌ Failed to retrieve collection schema")
            return None
        
        # Try to identify the Subject and Status fields dynamically
        print("\n🔍 Detecting required fields...")
        subject_field = find_field_by_pattern(fields, 'subject')
        status_field = find_field_by_pattern(fields, 'status')
        
        # Enhanced error reporting for missing fields
        if not subject_field:
            print("❌ Subject field not found!")
            print(f"   Available fields: {fields}")
            print("   Looking for fields containing 'subject' (case-insensitive)")
            return None
        
        if not status_field:
            print("❌ Status field not found!")
            print(f"   Available fields: {fields}")
            print("   Looking for fields containing 'status' (case-insensitive)")
            return None
        
        print(f"✅ Found subject field: {subject_field}")
        print(f"✅ Found status field: {status_field}")
        
        # Validate inputs
        if not subject_input or not status_input:
            print("❌ Invalid inputs: subject_input and status_input cannot be empty")
            return None
        
        # Build query expression with validation
        expr = f'{subject_field} == "{subject_input}" and {status_field} == "{status_input}"'
        print(f"🔍 Query expression: {expr}")
        
        # Validate expression is not None or empty
        if not expr or expr.strip() == "":
            print("❌ Generated expression is empty or None")
            return None
        
        # Perform query with additional error handling
        print("🔍 Executing query...")
        try:
            results = collection.query(
                expr=expr, 
                output_fields=fields, 
                limit=10
            )
            print(f"✅ Query executed successfully")
            return results
            
        except Exception as query_error:
            print(f"❌ Query execution failed: {query_error}")
            print(f"   Expression used: {expr}")
            print(f"   Subject field: {subject_field}")
            print(f"   Status field: {status_field}")
            return None
        
    except Exception as e:
        print(f"❌ Error querying collection: {e}")
        print(f"   Collection: {category}")
        print(f"   Subject input: {subject_input}")
        print(f"   Status input: {status_input}")
        return None

def display_results(results, category, subject_input, status_input):
    """Display query results in a formatted way"""
    if not results:
        print(f"\n⚠️ No matching records found for:")
        print(f"   Collection: {category}")
        print(f"   Subject: '{subject_input}'")
        print(f"   Status: '{status_input}'")
        return
    
    print(f"\n✅ Found {len(results)} matching record(s):")
    print(f"   Collection: {category}")
    print(f"   Subject: '{subject_input}'")
    print(f"   Status: '{status_input}'")
    print("=" * 80)
    
    for i, record in enumerate(results, 1):
        print(f"\n📄 Record {i}:")
        print("-" * 40)
        for key, value in record.items():
            if key == 'embedding':
                print(f"  {key}: [Vector with {len(value)} dimensions]")
            else:
                # Show full text for metadata fields
                display_value = str(value)
                if len(display_value) > 200:
                    display_value = display_value[:200] + "..."
                print(f"  {key}: {display_value}")

def main():
    """Main function with example usage"""
    print("🔍 Milvus Collection Query Tool")
    print("=" * 50)
    
    # Example inputs (you can modify these)
    category = "predisbursal_loan_query_loan_ca_data"
    subject_input = "loan cancel request"
    status_input = "IMDisbursed"
    
    print(f"📦 Collection: {category}")
    print(f"🔍 Subject: {subject_input}")
    print(f"📊 Status: {status_input}")
    print("-" * 50)
    
    # Execute query
    results = query_milvus_collection(category, subject_input, status_input)
    
    # Display results
    display_results(results, category, subject_input, status_input)
    
    # Summary
    if results:
        print(f"\n✅ Query completed successfully!")
        print(f"📊 Total records returned: {len(results)}")
    else:
        print(f"\n❌ Query completed with no results.")

if __name__ == "__main__":
    main() 