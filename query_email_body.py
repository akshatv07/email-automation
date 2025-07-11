#!/usr/bin/env python3
"""
Email Body Query Script
Accepts input in format: subject:value, status:value, category:value
Returns email body from Milvus database
"""

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

def parse_input(input_string):
    """
    Parse input string in format: subject:value, status:value, category:value
    
    Args:
        input_string (str): Input string in the specified format
        
    Returns:
        dict: Parsed key-value pairs
    """
    try:
        # Split by comma and clean up
        pairs = [pair.strip() for pair in input_string.split(',')]
        
        result = {}
        for pair in pairs:
            if ':' in pair:
                key, value = pair.split(':', 1)
                result[key.strip()] = value.strip()
        
        return result
    except Exception as e:
        print(f"❌ Error parsing input: {e}")
        return None

def find_email_body_field(fields):
    """Find the email body field in the collection schema"""
    # Common variations of email body field names
    email_body_patterns = [
        'email_body', 'body', 'emailbody', 'message_body', 
        'content', 'email_content', 'message', 'text'
    ]
    
    for pattern in email_body_patterns:
        field = next((f for f in fields if pattern in f.lower()), None)
        if field:
            return field
    
    return None

def query_email_body(category, subject_input, status_input):
    """
    Query Milvus collection for email body based on subject and status
    
    Args:
        category (str): Collection name
        subject_input (str): Subject to match
        status_input (str): Status to match
        
    Returns:
        list: Query results with email body
    """
    try:
        # Load collection
        print(f"📦 Loading collection: {category}")
        collection = Collection(category)
        collection.load()
        
        # Get all fields from schema
        fields = [f.name for f in collection.schema.fields]
        print(f"📋 Available fields: {fields}")
        
        # Find required fields
        subject_field = next((f for f in fields if 'subject' in f.lower()), None)
        status_field = next((f for f in fields if 'status' in f.lower()), None)
        email_body_field = find_email_body_field(fields)
        
        # Validate required fields
        if not subject_field:
            print("❌ Subject field not found in collection")
            print(f"   Available fields: {fields}")
            return None
            
        if not status_field:
            print("❌ Status field not found in collection")
            print(f"   Available fields: {fields}")
            return None
            
        if not email_body_field:
            print("❌ Email body field not found in collection")
            print(f"   Available fields: {fields}")
            return None
        
        print(f"✅ Found fields:")
        print(f"   Subject: {subject_field}")
        print(f"   Status: {status_field}")
        print(f"   Email Body: {email_body_field}")
        
        # Build query expression
        expr = f'{subject_field} == "{subject_input}" and {status_field} == "{status_input}"'
        print(f"🔍 Query expression: {expr}")
        
        # Execute query
        print("🔍 Executing query...")
        results = collection.query(
            expr=expr,
            output_fields=[email_body_field, subject_field, status_field],
            limit=10
        )
        
        return results, email_body_field
        
    except Exception as e:
        print(f"❌ Error querying collection: {e}")
        return None, None

def display_email_bodies(results, email_body_field):
    """Display email bodies in a formatted way"""
    if not results:
        print("⚠️ No matching records found")
        return
    
    print(f"\n✅ Found {len(results)} matching record(s):")
    print("=" * 80)
    
    for i, record in enumerate(results, 1):
        print(f"\n📧 Email {i}:")
        print("-" * 40)
        
        # Display email body
        if email_body_field in record:
            email_body = record[email_body_field]
            if email_body:
                # Clean up the email body for display
                cleaned_body = str(email_body).strip()
                if len(cleaned_body) > 500:
                    cleaned_body = cleaned_body[:500] + "..."
                print(f"📄 Email Body:\n{cleaned_body}")
            else:
                print("📄 Email Body: [Empty]")
        else:
            print("📄 Email Body: [Field not found]")
        
        # Display other relevant fields
        for key, value in record.items():
            if key != email_body_field:
                print(f"   {key}: {value}")

def main():
    """Main function with interactive input"""
    print("📧 Email Body Query Tool")
    print("=" * 50)
    print("Input format: subject:value, status:value, category:value")
    print("Example: subject:loan cancel request, status:IMSanctioned, category:predisbursal_loan_query_loan_ca")
    print("-" * 50)
    
    # Get user input
    user_input = input("Enter your query: ").strip()
    
    if not user_input:
        print("❌ No input provided")
        return
    
    # Parse input
    parsed = parse_input(user_input)
    if not parsed:
        print("❌ Invalid input format")
        return
    
    # Extract values
    subject = parsed.get('subject')
    status = parsed.get('status')
    category = parsed.get('category')
    
    if not all([subject, status, category]):
        print("❌ Missing required fields: subject, status, or category")
        print(f"   Parsed values: {parsed}")
        return
    
    print(f"\n🔍 Query Details:")
    print(f"   Category: {category}")
    print(f"   Subject: {subject}")
    print(f"   Status: {status}")
    print("-" * 50)
    
    # Connect to Milvus
    if not connect_to_milvus():
        return
    
    # Execute query
    results, email_body_field = query_email_body(category, subject, status)
    
    # Display results
    display_email_bodies(results, email_body_field)
    
    # Summary
    if results:
        print(f"\n✅ Query completed successfully!")
        print(f"📊 Total records returned: {len(results)}")
    else:
        print(f"\n❌ Query completed with no results.")

if __name__ == "__main__":
    main() 