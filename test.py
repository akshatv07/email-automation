from core.data_db_processor import process_ticket_metadata
import json

def main():
    data = process_ticket_metadata(
        ticket_id="3633280",
        subject="Query regarding loan",
        body="I want to know the status of my repayment",
        category="Loan Queries"
    )

    print("Processed Ticket Metadata:")
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main() 