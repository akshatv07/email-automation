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

                schema = self._build_schema(df)
                collection = Collection(name=collection_name, schema=schema)
                print(f"✅ Created collection '{collection_name}'")

                # Check for required columns
                if not all(col in df.columns for col in REQUIRED_COLS):
                    print(f"❌ Missing required columns in '{sheet_name}', skipping data insert.")
                    continue

                if df.empty:
                    print(f"⚠️ Sheet '{sheet_name}' is empty, but collection created.")
                    continue

                # Combine and embed
                df["query_text"] = df["subject"].astype(str) + " " + df["email_body"].astype(str)
                df["query_text"] = df["query_text"].apply(lambda x: _truncate(x, MAX_VARCHAR_LEN))
                vectors = self.embed_batch(df["query_text"].tolist())

                # Prepare row-wise data for insert
                insert_data = []
                for i, row in df.iterrows():
                    row_data = [row["query_text"]]
                    for col in df.columns:
                        if col == "query_text":
                            continue
                        val = row.get(col, "")
                        row_data.append(_truncate(val, MAX_VARCHAR_LEN))
                    row_data.append(vectors[i])
                    insert_data.append(row_data)

                # Transpose to columnar format
                columns = ["query_text"] + [col for col in df.columns if col != "query_text"] + ["embedding"]
                milvus_data = list(map(list, zip(*insert_data)))

                collection.insert(milvus_data)
                collection.create_index("embedding", {
                    "metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}
                })
                collection.load()
                print(f"📥 Inserted {len(df)} vectors into '{collection_name}'")

            except Exception as e:
                print(f"❌ Error processing sheet '{sheet_name}': {e}")
                continue

    def _build_schema(self, df):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="query_text", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LEN)
        ]
        for col in df.columns:
            if col == "query_text":
                continue
            fields.append(FieldSchema(name=col, dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LEN))
        fields.append(FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim))
        return CollectionSchema(fields) 