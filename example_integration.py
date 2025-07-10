#!/usr/bin/env python3
"""
Example integration showing how to use the Semantic Response Engine
with the existing data_db_processor
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.semantic_response_engine import SemanticResponseEngine
from core.data_db_processor import process_ticket_metadata
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def example_integration():
    """
    Example showing how to integrate semantic response engine with data_db_processor
    """
    try:
        print("🚀 Starting integration example...")
        
        # Initialize the semantic response engine
        semantic_engine = SemanticResponseEngine()
        
        # Example ticket data
        ticket_id = "3633261"
        subject = "Payment done but not reflected"
        body = "I have to paid my due in manually but not reflected in loan"
        category = "Predisbursal_Loan_Query_IM+_instances"
        
        print(f"\n📋 Processing ticket: {ticket_id}")
        print(f"Subject: {subject}")
        print(f"Category: {category}")
        
        # Step 1: Use data_db_processor to get ticket metadata
        print("\n🔍 Step 1: Getting ticket metadata from data_db_processor...")
        metadata = process_ticket_metadata(ticket_id, subject, body, category)
        print(f"Metadata: {metadata}")
        
        # Step 2: Use semantic response engine to get response
        print("\n🔍 Step 2: Getting semantic response...")
        query_text = f"{subject} {body}"
        result = semantic_engine.get_response(ticket_id, category, query_text)
        print(f"Semantic Response: {result}")
        
        # Step 3: Combine results
        print("\n🔍 Step 3: Combined results...")
        combined_result = {
            "ticket_id": ticket_id,
            "metadata": metadata,
            "semantic_response": result,
            "final_response": result.get("response", "No response found")
        }
        print(f"Combined Result: {combined_result}")
        
        print("\n✅ Integration example completed!")
        
    except Exception as e:
        print(f"❌ Error in integration example: {e}")
        import traceback
        traceback.print_exc()

def example_batch_processing():
    """
    Example showing batch processing of multiple tickets
    """
    try:
        print("\n🚀 Starting batch processing example...")
        
        # Initialize the semantic response engine
        semantic_engine = SemanticResponseEngine()
        
        # Sample batch of tickets
        tickets = [
            {
                "ticket_id": "3633261",
                "subject": "Payment done but not reflected",
                "body": "I have to paid my due in manually but not reflected in loan",
                "category": "Predisbursal_Loan_Query_IM+_instances"
            },
            {
                "ticket_id": "3633264",
                "subject": "Change bank account", 
                "body": "Hi team I want to change my Bank account pls help me with this",
                "category": "Update_-_Edit_details_Bank_Account_details_"
            },
            {
                "ticket_id": "3633263",
                "subject": "Sir mera emi ka date change krwana hai",
                "body": "Mera ko pard payment Krna hai qk mera abhi tbyat Boht khrb hai sir",
                "category": "Predisbursal_Loan_Query_IM+_instances"
            }
        ]
        
        results = []
        
        for i, ticket in enumerate(tickets, 1):
            print(f"\n📋 Processing ticket {i}/{len(tickets)}: {ticket['ticket_id']}")
            
            # Get metadata
            metadata = process_ticket_metadata(
                ticket['ticket_id'],
                ticket['subject'], 
                ticket['body'],
                ticket['category']
            )
            
            # Get semantic response
            query_text = f"{ticket['subject']} {ticket['body']}"
            semantic_result = semantic_engine.get_response(
                ticket['ticket_id'],
                ticket['category'],
                query_text
            )
            
            # Combine results
            combined = {
                "ticket_id": ticket['ticket_id'],
                "metadata": metadata,
                "semantic_response": semantic_result,
                "final_response": semantic_result.get("response", "No response found")
            }
            
            results.append(combined)
            print(f"✅ Processed: {combined['final_response']}")
        
        print(f"\n🎉 Batch processing completed! Processed {len(results)} tickets.")
        
        # Summary
        print("\n📊 Summary:")
        for result in results:
            print(f"Ticket {result['ticket_id']}: {result['final_response']}")
            
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run integration example
    example_integration()
    
    # Run batch processing example
    example_batch_processing() 