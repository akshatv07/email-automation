# Ticket Resolution System

## Prerequisites
- Windows 10/11
- Python 3.8+
- Docker Desktop (for Milvus)

## Setup Instructions

### 1. Install Python
1. Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"

### 2. Install Milvus
1. Install Docker Desktop from [Docker's website](https://www.docker.com/products/docker-desktop/)
2. Open PowerShell as Administrator
3. Pull Milvus standalone image:
   ```powershell
   docker pull milvusdb/milvus:latest
   ```
4. Run Milvus:
   ```powershell
   docker-compose up -d
   ```

### 3. Set Up Project
1. Clone the repository
2. Create a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

### 4. Run the Application
```powershell
python main.py
```

### 5. Test Semantic Response Engine
```powershell
python test_semantic_response.py
```

### 6. Run Integration Example
```powershell
python example_integration.py
```

## Project Structure
- `main.py`: Entry point for the application
- `config/settings.py`: Global configuration settings
- `data/`: Contains databases and knowledge base sheets
- `core/`: Core application components
  - `data_db_processor.py`: Processes ticket metadata from CSV
  - `vector_ingestion_engine.py`: Ingests knowledge base into Milvus
  - `semantic_response_engine.py`: Semantic search and response extraction
- `test_semantic_response.py`: Test script for semantic response engine
- `example_integration.py`: Example showing integration with existing components

## Components

### Semantic Response Engine
The `semantic_response_engine.py` component provides intelligent response generation based on:
- **Ticket Metadata**: Uses `data_db.csv` to determine appropriate status keys
- **Semantic Search**: Vectorizes queries and searches Milvus collections
- **Response Extraction**: Extracts relevant responses based on ticket type

**Key Features:**
- Automatic status key selection based on `data_from_IM_pls` field
- Cosine similarity search in Milvus collections
- Normalized vector embeddings for accurate matching
- Comprehensive error handling and logging

**Usage:**
```python
from core.semantic_response_engine import SemanticResponseEngine

engine = SemanticResponseEngine()
result = engine.get_response(
    ticket_id="3633261",
    category_name="Predisbursal_Loan_Query_IM+_instances", 
    query_text="Payment done but not reflected in loan"
)
```

## Troubleshooting
- Ensure Docker is running before starting Milvus
- Check Python and pip versions with `python --version` and `pip --version` 