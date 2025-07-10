import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import settings

def main():
    """
    Entry point for the ticket resolution system.
    Orchestrates the main logic of the application.
    """
    print("Ticket Resolution System")
    print(f"Database File: {settings.CSV_DB_FILE}")
    print(f"Knowledge Base Folder: {settings.KB_FOLDER}")
    print(f"Embedding Model: {settings.EMBEDDING_MODEL}")

if __name__ == "__main__":
    main()
