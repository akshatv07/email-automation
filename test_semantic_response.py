#!/usr/bin/env python3
"""
Test script for the Semantic Response Engine
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.semantic_response_engine import SemanticResponseEngine
import logging

# Configure logging to see detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_semantic_response_engine():
    """
    Test the semantic response engine with sample data
    """
    try:
        print("🚀 Initializing Semantic Response Engine...")
        engine = SemanticResponseEngine()
        
        # Test cases
        test_cases = [
            {
                "ticket_id": "3633261",
                "category_name": "Predisbursal_Loan_Query_IM+_instances",
                "query_text": "Payment done but not reflected I have to paid my due in manually but not reflected in loan"
            },
            {
                "ticket_id": "3633264", 
                "category_name": "Update_-_Edit_details_Bank_Account_details_",
                "query_text": "Change bank account Hi team I want to change my Bank account pls help me with this"
            },
            {
                "ticket_id": "3633263",
                "category_name": "Predisbursal_Loan_Query_IM+_instances", 
                "query_text": "Sir mera emi ka date change krwana hai Mera ko pard payment Krna hai qk mera abhi tbyat Boht khrb hai sir"
            }
        ]
        
        print("\n🧪 Running test cases...")
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Ticket ID: {test_case['ticket_id']}")
            print(f"Category: {test_case['category_name']}")
            print(f"Query: {test_case['query_text'][:100]}...")
            
            result = engine.get_response(
                ticket_id=test_case['ticket_id'],
                category_name=test_case['category_name'],
                query_text=test_case['query_text']
            )
            
            print(f"✅ Result: {result}")
            print("-" * 50)
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_semantic_response_engine() 