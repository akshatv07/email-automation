#!/usr/bin/env python3
"""
Test script for robust Milvus query functionality
Demonstrates the improved error handling and field detection
"""

from pymilvus import connections, Collection
import config.settings as settings

def test_connection():
    """Test basic Milvus connection"""
    try:
        connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        print("✅ Milvus connection test successful")
        return True
    except Exception as e:
        print(f"❌ Milvus connection failed: {e}")
        return False

def test_collection_exists(collection_name):
    """Test if collection exists and get its schema"""
    try:
        collection = Collection(collection_name)
        collection.load()
        
        fields = [f.name for f in collection.schema.fields]
        print(f"✅ Collection '{collection_name}' exists")
        print(f"📋 Available fields: {fields}")
        return fields
    except Exception as e:
        print(f"❌ Collection '{collection_name}' not found or error: {e}")
        return None

def test_field_detection(collection_name):
    """Test field detection logic"""
    try:
        collection = Collection(collection_name)
        collection.load()
        
        fields = [f.name for f in collection.schema.fields]
        
        # Test subject field detection
        subject_field = next((f for f in fields if 'subject' in f.lower()), None)
        print(f"🔍 Subject field detection: {subject_field}")
        
        # Test status field detection
        status_field = next((f for f in fields if 'status' in f.lower()), None)
        print(f"🔍 Status field detection: {status_field}")
        
        return subject_field, status_field
    except Exception as e:
        print(f"❌ Field detection test failed: {e}")
        return None, None

def test_query_expression(collection_name, subject_field, status_field):
    """Test query expression construction"""
    if not subject_field or not status_field:
        print("❌ Cannot test query expression - missing required fields")
        return None
    
    subject_input = "loan cancel request"
    status_input = "IMDisbursed"
    
    expr = f'{subject_field} == "{subject_input}" and {status_field} == "{status_input}"'
    print(f"🔍 Generated expression: {expr}")
    
    # Validate expression
    if not expr or expr.strip() == "":
        print("❌ Generated expression is empty or None")
        return None
    
    if "None" in expr:
        print("❌ Expression contains None values")
        return None
    
    print("✅ Expression validation passed")
    return expr

def main():
    """Run comprehensive tests"""
    print("🧪 Testing Robust Milvus Query System")
    print("=" * 50)
    
    # Test 1: Connection
    print("\n1️⃣ Testing Milvus connection...")
    if not test_connection():
        print("❌ Cannot proceed - connection failed")
        return
    
    # Test 2: Collection existence
    print("\n2️⃣ Testing collection existence...")
    collection_name = "predisbursal_loan_query_loan_ca_data"
    fields = test_collection_exists(collection_name)
    if not fields:
        print("❌ Cannot proceed - collection not found")
        return
    
    # Test 3: Field detection
    print("\n3️⃣ Testing field detection...")
    subject_field, status_field = test_field_detection(collection_name)
    if not subject_field or not status_field:
        print("❌ Cannot proceed - required fields not found")
        return
    
    # Test 4: Expression construction
    print("\n4️⃣ Testing query expression construction...")
    expr = test_query_expression(collection_name, subject_field, status_field)
    if not expr:
        print("❌ Cannot proceed - expression construction failed")
        return
    
    # Test 5: Actual query execution
    print("\n5️⃣ Testing actual query execution...")
    try:
        collection = Collection(collection_name)
        collection.load()
        
        results = collection.query(
            expr=expr,
            output_fields=fields,
            limit=5
        )
        
        if results:
            print(f"✅ Query successful! Found {len(results)} records")
            print("📄 Sample result:")
            print(f"   {results[0]}")
        else:
            print("⚠️ Query executed but no results found")
            
    except Exception as e:
        print(f"❌ Query execution failed: {e}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main() 