import os
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility
)
import config.settings as settings

REQUIRED_COLS = ["subject", "email_body"]  # You can add more required columns here
MAX_VARCHAR_LEN = 65535  # Milvus 2.2+ limit for VARCHAR


def _sanitize(name):
    """
    Lowercase, replace non-alphanumeric with underscores, remove leading/trailing underscores.
    """
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def _truncate(text, max_len=MAX_VARCHAR_LEN):
    if not isinstance(text, str):
        text = str(text)
    if len(text) > max_len:
        print(f"⚠️ Truncating value from {len(text)} to {max_len} characters.")
        return text[:max_len]
    return text


class VectorIngestionEngine:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.embedding_dim = settings.EMBEDDING_DIM
        connections.connect(alias="default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        print("✅ Milvus connection established")

    def embed_batch(self, texts):
        if not texts:
            return []
        embeddings = self.model.encode(texts)
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return normalized.tolist()

    def ingest_excel_folder(self, folder_path="data/kb_sheets"):
        for file in os.listdir(folder_path):
            if not file.endswith(".xlsx"):
                continue
            file_path = os.path.join(folder_path, file)
            self.ingest_excel_file(file_path)

    def ingest_excel_file(self, file_path):
        excel = pd.ExcelFile(file_path)
        for sheet_name in excel.sheet_names:
            print(f"\n📄 Processing Sheet: {sheet_name}")
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=1, engine="openpyxl")
                df.columns = [_sanitize(c) for c in df.columns]
                collection_name = _sanitize(sheet_name)

                # Always create collection, even if empty or invalid
                if utility.has_collection(collection_name):
                    print(f"ℹ️ Collection '{collection_name}' already exists. Dropping and recreating...")
                    utility.drop_collection(collection_name)

                # Check for required columns
                print(f"🔍 Available columns in '{sheet_name}': {list(df.columns)}")
                print(f"🔍 Required columns: {REQUIRED_COLS}")
                
                # Check if required columns exist (with case-insensitive matching)
                missing_cols = []
                for required_col in REQUIRED_COLS:
                    if required_col not in df.columns:
                        # Try to find similar column names
                        similar_cols = [col for col in df.columns if required_col.lower() in col.lower() or col.lower() in required_col.lower()]
                        if similar_cols:
                            print(f"⚠️ Column '{required_col}' not found, but found similar: {similar_cols}")
                        missing_cols.append(required_col)
                
                if missing_cols:
                    print(f"❌ Missing required columns in '{sheet_name}': {missing_cols}")
                    print(f"❌ Skipping data insert for sheet '{sheet_name}'")
                    continue

                if df.empty:
                    print(f"⚠️ Sheet '{sheet_name}' is empty, but collection created.")
                    continue
                
                print(f"📊 DataFrame shape: {df.shape}")

                # Create embeddings from Subject and Email Body only
                texts = (df["subject"].fillna("") + " " + df["email_body"].fillna("")).tolist()
                embeddings = self.embed_batch(texts)
                print(f"🔍 Generated {len(embeddings)} vectors")

                # Dynamically build schema based on this sheet's columns
                schema = self._build_dynamic_schema(df)
                collection = Collection(name=collection_name, schema=schema)
                print(f"✅ Created collection '{collection_name}' with dynamic schema")

                # Prepare data for insertion
                insert_data = [embeddings]  # Start with embeddings
                
                # Add all metadata columns (excluding subject and email_body)
                metadata_columns = [col for col in df.columns if col not in REQUIRED_COLS]
                for col in metadata_columns:
                    insert_data.append(df[col].fillna("").astype(str).tolist())

                # Insert data
                collection.insert(insert_data)
                collection.create_index("embedding", {
                    "metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}
                })
                collection.load()
                print(f"📥 Inserted {len(df)} vectors into '{collection_name}' with {len(metadata_columns)} metadata fields")

            except Exception as e:
                print(f"❌ Error processing sheet '{sheet_name}': {e}")
                continue

    def verify_collection(self, collection_name, limit=10):
        """
        Verify insertion by querying a collection and displaying the entire stored data table.
        """
        try:
            if not utility.has_collection(collection_name):
                print(f"❌ Collection '{collection_name}' does not exist")
                return
            
            collection = Collection(collection_name)
            collection.load()
            
            # Query all data
            results = collection.query(
                expr="id >= 0", 
                output_fields=["*"], 
                limit=limit
            )
            
            print(f"\n📊 Verification Results for Collection '{collection_name}':")
            print(f"Total records retrieved: {len(results)}")
            
            if results:
                # Display first record structure
                first_record = results[0]
                print(f"\n📋 Record Structure:")
                for key, value in first_record.items():
                    if key == "embedding":
                        print(f"  {key}: [Vector with {len(value)} dimensions]")
                    else:
                        # Truncate long text for display
                        display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        print(f"  {key}: {display_value}")
                
                print(f"\n📋 All Records (showing first {limit}):")
                for i, record in enumerate(results):
                    print(f"\n--- Record {i+1} ---")
                    for key, value in record.items():
                        if key == "embedding":
                            print(f"  {key}: [Vector with {len(value)} dimensions]")
                        else:
                            # Truncate long text for display
                            display_value = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                            print(f"  {key}: {display_value}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error verifying collection '{collection_name}': {e}")
            return None

    def _build_dynamic_schema(self, df):
        """
        Dynamically build schema based on the DataFrame columns.
        Only Subject and Email Body are used for embeddings, all other columns become metadata.
        """
        # Start with ID and embedding fields
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        ]
        
        # Add all columns as metadata fields (excluding the ones used for embeddings)
        metadata_columns = [col for col in df.columns if col not in REQUIRED_COLS]
        for col in metadata_columns:
            fields.append(FieldSchema(name=col, dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LEN))
        
        print(f"🔧 Dynamic schema created with {len(metadata_columns)} metadata fields: {metadata_columns}")
        return CollectionSchema(fields) 