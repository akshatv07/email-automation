from core.vector_ingestion_engine import VectorIngestionEngine
import pandas as pd
import os

def create_test_excel():
    """
    Create a test Excel file with different sheet structures to demonstrate dynamic schema.
    """
    # Create test data with different column structures
    sheet1_data = {
        'Subject': ['Ticket Issue 1', 'Login Problem', 'Payment Error'],
        'Email Body': ['I cannot access my account', 'Login page not working', 'Payment failed'],
        'Ticket ID': ['T001', 'T002', 'T003'],
        'Status': ['Open', 'In Progress', 'Closed'],
        'Priority': ['High', 'Medium', 'Low']
    }
    
    sheet2_data = {
        'Subject': ['Feature Request', 'Bug Report', 'General Question'],
        'Email Body': ['Can you add dark mode?', 'App crashes on startup', 'How to reset password?'],
        'Customer ID': ['C001', 'C002', 'C003'],
        'Resolution Date': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'Department': ['UI/UX', 'Engineering', 'Support']
    }
    
    sheet3_data = {
        'Subject': ['System Down', 'Data Export', 'API Issue'],
        'Email Body': ['Server is not responding', 'Need data export feature', 'API returning 500 errors'],
        'Incident ID': ['I001', 'I002', 'I003'],
        'Severity': ['Critical', 'High', 'Medium'],
        'Assigned To': ['John Doe', 'Jane Smith', 'Bob Wilson'],
        'Estimated Fix': ['2 hours', '1 day', '3 days']
    }
    
    # Create Excel file with multiple sheets
    with pd.ExcelWriter('test_dynamic_schema.xlsx', engine='openpyxl') as writer:
        pd.DataFrame(sheet1_data).to_excel(writer, sheet_name='Support_Tickets', index=False)
        pd.DataFrame(sheet2_data).to_excel(writer, sheet_name='Feature_Requests', index=False)
        pd.DataFrame(sheet3_data).to_excel(writer, sheet_name='System_Issues', index=False)
    
    print("✅ Created test Excel file: test_dynamic_schema.xlsx")
    print("📋 Sheet structures:")
    print("  - Support_Tickets: Subject, Email Body, Ticket ID, Status, Priority")
    print("  - Feature_Requests: Subject, Email Body, Customer ID, Resolution Date, Department")
    print("  - System_Issues: Subject, Email Body, Incident ID, Severity, Assigned To, Estimated Fix")

def main():
    # Create test data
    create_test_excel()
    
    # Initialize the engine
    engine = VectorIngestionEngine()
    
    # Ingest the test file
    print("\n🚀 Starting ingestion process...")
    engine.ingest_excel_file("test_dynamic_schema.xlsx")
    
    # Verify each collection
    print("\n🔍 Starting verification process...")
    
    collections_to_verify = ['support_tickets', 'feature_requests', 'system_issues']
    
    for collection_name in collections_to_verify:
        print(f"\n{'='*50}")
        print(f"🔍 Verifying collection: '{collection_name}'")
        results = engine.verify_collection(collection_name, limit=3)
        
        if results:
            print(f"✅ Verification successful! Retrieved {len(results)} records from '{collection_name}'")
        else:
            print(f"❌ Verification failed for collection '{collection_name}'")
    
    # Clean up test file
    if os.path.exists('test_dynamic_schema.xlsx'):
        os.remove('test_dynamic_schema.xlsx')
        print("\n🧹 Cleaned up test file")

if __name__ == "__main__":
    main() 