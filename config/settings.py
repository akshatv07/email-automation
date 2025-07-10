import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File paths
CSV_DB_FILE = os.path.join(BASE_DIR, "data", "datadb.csv")
KB_FOLDER = os.path.join(BASE_DIR, "data", "kb_sheets")

# Embedding model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Milvus configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530" 