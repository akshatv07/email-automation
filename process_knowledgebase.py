import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from dotenv import load_dotenv
import glob

# Load environment variables
load_dotenv()

# Milvus connection parameters
MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
COLLECTION_NAME = "email_knowledgebase"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2 model

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def connect_to_milvus():
    """Establish connection to Milvus server."""
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"Connected to Milvus server: {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise

def create_collection():
    """Create Milvus collection if it doesn't exist."""
    try:
        # First, try to drop the collection if it exists
        if utility.has_collection(COLLECTION_NAME):
            print(f"Dropping existing collection: {COLLECTION_NAME}")
            utility.drop_collection(COLLECTION_NAME, timeout=5)
            print("Collection dropped successfully")
        
        # Wait a moment to ensure the collection is fully dropped
        import time
        time.sleep(2)
        
        # Define collection schema with explicit field types
        print("Creating new collection...")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="sheet_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="email_body", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ]
        
        # Enable dynamic fields for any additional columns
        schema = CollectionSchema(
            fields=fields, 
            description="Email Knowledge Base",
            enable_dynamic_field=True  # This allows additional fields to be stored as JSON
        )
        
        # Create the collection
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        
        # Create an IVF_FLAT index for the embedding field
        print("Creating index...")
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        
        # Create the index
        collection.create_index(
            field_name="embedding", 
            index_params=index_params,
            index_name="embedding_idx"
        )
        
        # Load the collection to make it searchable
        collection.load()
        
        print(f"Created collection: {COLLECTION_NAME}")
        return collection
        
    except Exception as e:
        print(f"Error creating collection: {str(e)}")
        print("Trying to continue with existing collection...")
        try:
            collection = Collection(COLLECTION_NAME)
            collection.load()
            return collection
        except Exception as e2:
            print(f"Failed to load existing collection: {str(e2)}")
            raise

def process_sheet(file_path, sheet_name):
    """Process a single sheet from an Excel file.
    
    Args:
        file_path (str): Path to the Excel file
        sheet_name (str): Name of the sheet to process (will be used as collection name)
        
    Returns:
        tuple: (list of processed data, error message if any)
    """
    try:
        print(f"  Processing sheet: {sheet_name}")
        
        try:
            # Read the full data with header in row 1 (0-based index)
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=1)
            
            # Clean column names (remove any extra whitespace)
            df.columns = [str(col).strip() for col in df.columns]
            
            # Find subject and email body columns (case-insensitive)
            subject_col = None
            body_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if 'subject' in col_lower:
                    subject_col = col
                elif 'email' in col_lower and 'body' in col_lower:
                    body_col = col
            
            if not subject_col and not body_col:
                return [], "Could not find subject or email body columns"
            
            processed_data = []
            empty_rows = 0
            
            # Process each row
            for idx, row in df.iterrows():
                # Skip entirely empty rows
                if row.isna().all():
                    continue
                    
                # Get subject and body
                subject = str(row[subject_col]) if subject_col and pd.notna(row.get(subject_col, None)) else ''
                email_body = str(row[body_col]) if body_col and pd.notna(row.get(body_col, None)) else ''
                
                # Skip if both are empty
                if not subject and not email_body:
                    empty_rows += 1
                    continue
                
                # Get all other columns as metadata
                metadata = {}
                for col in df.columns:
                    if col not in [subject_col, body_col] and pd.notna(row.get(col, None)):
                        metadata[str(col).strip()] = str(row[col]).strip()
                
                processed_data.append({
                    'file_name': os.path.basename(file_path),
                    'sheet_name': sheet_name,
                    'subject': subject.strip(),
                    'email_body': email_body.strip(),
                    **metadata
                })
            
            print(f"  Processed {len(processed_data)} rows (skipped {empty_rows} empty rows)")
            
            if not processed_data:
                return [], "No valid data found in sheet"
                
            return processed_data, None
            
        except Exception as e:
            print(f"  Error reading sheet with header in row 2: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, f"Error processing sheet: {str(e)}"
        
    except Exception as e:
        error_msg = f"Error processing sheet {sheet_name}: {str(e)}"
        print(f"  {error_msg}")
        import traceback
        traceback.print_exc()
        return None, error_msg

def process_excel_files(directory):
    """Process all Excel files in the given directory, sheet by sheet.
    
    Args:
        directory (str): Directory containing Excel files to process
        
    Returns:
        dict: Dictionary with sheet names as keys and their data as values
    """
    try:
        excel_files = [f for f in os.listdir(directory) if f.lower().endswith(('.xlsx', '.xls'))]
        
        if not excel_files:
            print("No Excel files found in the directory.")
            return {}
        
        print(f"Found {len(excel_files)} Excel files to process.")
        all_sheets_data = {}
        
        for file_name in excel_files:
            file_path = os.path.join(directory, file_name)
            print(f"\n{'='*70}")
            print(f"PROCESSING FILE: {file_name}")
            print(f"{'='*70}")
            
            try:
                # Get all sheet names
                try:
                    xl = pd.ExcelFile(file_path)
                    sheet_names = xl.sheet_names
                    print(f"\nFound {len(sheet_names)} sheets: {', '.join(sheet_names)}")
                except Exception as e:
                    print(f"  Error reading Excel file {file_name}: {str(e)}")
                    continue
                
                for sheet_name in sheet_names:
                    # Skip sheets that start with 'Sheet' as they're usually empty
                    if sheet_name.lower().startswith('sheet'):
                        print(f"\nSkipping sheet (starts with 'Sheet'): {sheet_name}")
                        continue
                    
                    print(f"\n{'='*60}")
                    print(f"SHEET: {sheet_name}")
                    print(f"{'='*60}")
                    
                    # Process the sheet
                    sheet_data, error = process_sheet(file_path, sheet_name)
                    
                    if error:
                        print(f"  ERROR: {error}")
                        all_sheets_data[f"{file_name} - {sheet_name}"] = []
                        continue
                    
                    if not sheet_data:
                        print("  No valid data found in sheet after processing")
                        all_sheets_data[f"{file_name} - {sheet_name}"] = []
                        continue
                    
                    # Print sample of the data
                    print(f"\n  SAMPLE DATA (first row):")
                    sample = sheet_data[0]
                    print(f"  - Subject: {sample.get('subject', '')[:100]}...")
                    print(f"  - Body: {sample.get('email_body', '')[:100]}...")
                    print(f"  - Additional fields: {', '.join(k for k in sample.keys() if k not in ['file_name', 'sheet_name', 'subject', 'email_body'])}")
                    
                    all_sheets_data[f"{file_name} - {sheet_name}"] = sheet_data
                    
            except Exception as e:
                print(f"\nERROR processing {file_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Print summary
        print("\n" + "="*70)
        print("PROCESSING SUMMARY")
        print("="*70)
        total_sheets = len(all_sheets_data)
        total_records = sum(len(data) for data in all_sheets_data.values())
        print(f"Processed {total_sheets} sheets with a total of {total_records} records")
        
        return all_sheets_data
        
    except Exception as e:
        print(f"\nFATAL ERROR in process_excel_files: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def generate_embeddings(texts):
    """Generate embeddings for a list of texts.
    
    Args:
        texts: List of strings, dictionaries, or a mix of both
    """
    if not texts:
        return []
    
    combined_texts = []
    
    for text in texts:
        if isinstance(text, str):
            combined_texts.append(text)
        elif isinstance(text, dict):
            # Handle case where text might be a dictionary with subject and email_body
            subject = str(text.get('subject', ''))
            body = str(text.get('email_body', text.get('body', '')))
            combined_texts.append(f"{subject} {body}".strip())
        else:
            # Try to convert to string as a fallback
            combined_texts.append(str(text))
    
    try:
        # Generate embeddings with error handling
        if not combined_texts:
            return []
            
        embeddings = model.encode(combined_texts, show_progress_bar=True)
        
        # Convert to list if it's a numpy array
        if hasattr(embeddings, 'tolist'):
            return embeddings.tolist()
        return embeddings
        
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        print(f"Sample text: {combined_texts[0][:100]}..." if combined_texts else "No texts to process")
        raise

def ensure_string(value, field_name=None):
    """Convert value to string, handling lists and other types."""
    if value is None:
        return ''
    if isinstance(value, (list, dict)):
        if field_name == 'email_body' and isinstance(value, list) and len(value) > 0:
            # If it's the email_body field and it's a non-empty list, join it
            return ' '.join(str(item) for item in value if item is not None)
        return str(value)
    return str(value)

def create_collection_for_sheet(collection_name):
    """Create a Milvus collection for a specific sheet."""
    try:
        # Drop existing collection if it exists
        if utility.has_collection(collection_name):
            print(f"  Dropping existing collection: {collection_name}")
            utility.drop_collection(collection_name)
        
        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="sheet_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="email_body", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ]
        
        # Enable dynamic fields for any additional columns
        schema = CollectionSchema(
            fields=fields,
            description=f"Email data from {collection_name}",
            enable_dynamic_field=True
        )
        
        # Create the collection
        collection = Collection(name=collection_name, schema=schema)
        
        # Create index
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        
        collection.create_index(
            field_name="embedding",
            index_params=index_params,
            index_name=f"{collection_name}_idx"
        )
        
        # Load the collection
        collection.load()
        
        print(f"  Created and loaded collection: {collection_name}")
        return collection
        
    except Exception as e:
        print(f"  Error creating collection {collection_name}: {str(e)}")
        return None

def insert_sheet_data(collection_name, sheet_data):
    """Insert data for a single sheet into its collection."""
    if not sheet_data:
        print("  No data to insert for this sheet")
        return 0
    
    # Create collection for this sheet
    collection = create_collection_for_sheet(collection_name)
    if not collection:
        return 0
    
    # Get all possible fields from the data
    all_fields = set()
    for record in sheet_data:
        all_fields.update(record.keys())
    
    # Standard fields that we'll handle specially
    standard_fields = {'file_name', 'sheet_name', 'subject', 'email_body'}
    dynamic_fields = all_fields - standard_fields
    
    print(f"  Found {len(dynamic_fields)} dynamic fields")
    
    batch_size = 10
    total_inserted = 0
    
    # Process data in batches
    for i in range(0, len(sheet_data), batch_size):
        batch = sheet_data[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(sheet_data) - 1) // batch_size + 1
        
        print(f"  Processing batch {batch_num}/{total_batches} (rows {i} to {min(i + batch_size, len(sheet_data)) - 1})...")
        
        batch_entities = {}
        
        # First, convert all values to strings in the batch
        processed_batch = []
        for record in batch:
            processed_record = {}
            for key, value in record.items():
                # Convert value to string, handling different types
                if value is None:
                    processed_record[key] = ''
                elif isinstance(value, (list, tuple, dict)):
                    # Convert lists, tuples, and dicts to JSON strings
                    import json
                    try:
                        processed_record[key] = json.dumps(value, ensure_ascii=False)
                    except:
                        processed_record[key] = str(value)
                else:
                    processed_record[key] = str(value).strip()
            processed_batch.append(processed_record)
        
        # Process standard fields
        for field in standard_fields:
            field_values = []
            for record in processed_batch:
                field_values.append(record.get(field, ''))
            batch_entities[field] = field_values
        
        # Process dynamic fields
        for field in dynamic_fields:
            field_values = []
            has_non_empty = False
            
            for record in processed_batch:
                value = record.get(field, '')
                field_values.append(value)
                if value:  # Check if value is non-empty
                    has_non_empty = True
            
            # Only include if there are non-empty values
            if has_non_empty:
                batch_entities[field] = field_values
        
        # Generate embeddings
        try:
            # Prepare text for embeddings
            texts = []
            for record in processed_batch:
                subject = record.get('subject', '')
                body = record.get('email_body', '')
                texts.append(f"{subject} {body}".strip())
            
            # Generate embeddings
            if not texts or not any(texts):
                print("    No valid text for embeddings, skipping batch")
                continue
                
            embeddings = generate_embeddings(texts)
            if not embeddings:
                print("    Failed to generate embeddings, skipping batch")
                continue
                
            batch_entities['embedding'] = embeddings
            
            # Debug: Print field types before insertion
            print("    Field types before insertion:")
            for field, values in batch_entities.items():
                if field != 'embedding':
                    sample = values[0] if values else 'None'
                    print(f"      {field}: {type(sample).__name__} (sample: {str(sample)[:50]}...)")
            
            # Insert the batch
            insert_result = collection.insert(batch_entities)
            total_inserted += len(batch)
            print(f"    Inserted {len(batch)} records (Total: {total_inserted})")
            
            # Flush to ensure data is written
            collection.flush()
            
        except Exception as e:
            print(f"    Error in batch {batch_num}: {str(e)}")
            # Print the problematic data
            print("    Problematic data sample:")
            for field, values in batch_entities.items():
                if field != 'embedding':
                    print(f"      {field}: {values[0] if values else 'None'}")
            continue
    
    print(f"  Finished inserting {total_inserted} records into {collection_name}")
    return total_inserted

def process_and_insert_data(sheets_data):
    """Process and insert data for all sheets."""
    if not sheets_data:
        print("No data to process.")
        return
    
    total_sheets = len(sheets_data)
    processed_sheets = 0
    
    print(f"\n{'='*60}")
    print(f"STARTING DATA PROCESSING FOR {total_sheets} SHEETS")
    print(f"{'='*60}")
    
    for sheet_id, (sheet_name, sheet_data) in enumerate(sheets_data.items(), 1):
        print(f"\n{'#'*50}")
        print(f"PROCESSING SHEET {sheet_id}/{total_sheets}: {sheet_name}")
        print(f"{'#'*50}")
        
        # Use sheet name as collection name (clean it to be a valid identifier)
        collection_name = re.sub(r'[^a-zA-Z0-9_]', '_', sheet_name)
        # Ensure it starts with a letter and is not too long
        if not collection_name[0].isalpha():
            collection_name = f"collection_{collection_name}"
        collection_name = collection_name[:63]  # Max length for collection name
        
        print(f"Collection name: {collection_name}")
        print(f"Number of records: {len(sheet_data) if sheet_data else 0}")
        
        if not sheet_data:
            # Create an empty collection for empty sheets
            create_collection_for_sheet(collection_name)
            print("  Created empty collection for sheet")
            processed_sheets += 1
            continue
        
        # Process and insert data for this sheet
        try:
            inserted_count = insert_sheet_data(collection_name, sheet_data)
            if inserted_count > 0:
                processed_sheets += 1
                print(f"  Successfully processed {inserted_count} records")
            else:
                print("  No records were inserted")
                
        except Exception as e:
            print(f"  Error processing sheet {sheet_name}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"Total sheets processed: {processed_sheets}/{total_sheets}")
    print(f"{'='*60}")
    
    return processed_sheets

def main():
    # Connect to Milvus
    try:
        connect_to_milvus()
    except Exception as e:
        print(f"Error connecting to Milvus: {str(e)}")
        return
    
    # Process Excel files
    print("Starting to process Excel files...")
    try:
        sheets_data = process_excel_files('.')
    except Exception as e:
        print(f"Error processing Excel files: {str(e)}")
        return
    
    if not sheets_data:
        print("No data to process.")
        return
    
    # Process and insert data for all sheets
    try:
        process_and_insert_data(sheets_data)
    except Exception as e:
        print(f"Error processing and inserting data: {str(e)}")
        return
    
    print("\nProcessing complete. You can now query the collections.")
    print("Each sheet has been stored in a separate collection with names like 'email_12345678'.")

if __name__ == "__main__":
    main()
