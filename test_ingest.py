from core.vector_ingestion_engine import VectorIngestionEngine
import os
import pandas as pd
import re

def _sanitize(name):
    """Sanitize collection name"""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")

def main():
    # Initialize the engine
    engine = VectorIngestionEngine()

    # Ingest all Excel files in the folder
    print("🚀 Starting ingestion process...")
    engine.ingest_excel_folder("data/kb_sheets")

    # Verify insertion by querying one collection
    print("\n🔍 Starting verification process...")

    # Find the first Excel file and get its first sheet name
    excel_files = [f for f in os.listdir("data/kb_sheets") if f.endswith(".xlsx")]
    if excel_files:
        first_file = os.path.join("data/kb_sheets", excel_files[0])
        excel = pd.ExcelFile(first_file)
        if excel.sheet_names:
            first_sheet = excel.sheet_names[0]
            collection_name = _sanitize(first_sheet)
            
            print(f"🔍 Verifying collection: '{collection_name}'")
            results = engine.verify_collection(collection_name, limit=5)
            
            if results:
                print(f"\n✅ Verification successful! Retrieved {len(results)} records from '{collection_name}'")
                print("📊 Each collection dynamically adapts to its sheet's column structure:")
                print("   - Only 'Subject' + 'Email Body' are used for embeddings")
                print("   - All other columns become metadata fields")
                print("   - Each sheet gets its own unique schema")
            else:
                print(f"\n❌ Verification failed for collection '{collection_name}'")
        else:
            print("❌ No sheets found in Excel file")
    else:
        print("❌ No Excel files found in data/kb_sheets folder")

if __name__ == "__main__":
    main() 