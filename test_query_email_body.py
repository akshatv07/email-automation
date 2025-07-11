#!/usr/bin/env python3
"""
Test Suite for Email Body Query Script

This test file provides comprehensive test cases for the EmailQueryManager
to ensure robust functionality across different scenarios.
"""

from query_email_body import EmailQueryManager, EmailQueryError
import pytest

class TestEmailQueryManager:
    @pytest.fixture
    def query_manager(self):
        """
        Fixture to create a query manager and establish Milvus connection
        """
        manager = EmailQueryManager()
        assert manager.connect(), "Failed to connect to Milvus"
        return manager

    def test_successful_query(self, query_manager):
        """
        Test a successful email query with valid parameters
        """
        query_params = {
            'category': 'predisbursal_loan_query_loan_ca',
            'subject': 'loan cancel request',
            'status': 'IMSanctioned'
        }
        
        results = query_manager.query_emails(query_params)
        
        assert isinstance(results, list), "Results should be a list"
        # Optional: Add more specific assertions based on expected results
        
    def test_missing_parameters(self, query_manager):
        """
        Test query with missing parameters
        """
        invalid_queries = [
            {},  # Empty dictionary
            {'category': 'test'},  # Missing subject and status
            {'subject': 'test'},  # Missing category and status
            {'status': 'test'}    # Missing category and subject
        ]
        
        for query_params in invalid_queries:
            with pytest.raises(EmailQueryError, match="Missing required parameters"):
                query_manager.query_emails(query_params)
    
    def test_non_existent_collection(self, query_manager):
        """
        Test query with a non-existent collection
        """
        query_params = {
            'category': 'non_existent_collection',
            'subject': 'test subject',
            'status': 'test status'
        }
        
        results = query_manager.query_emails(query_params)
        assert results == [], "Non-existent collection should return empty list"
    
    def test_display_results(self, query_manager, capsys):
        """
        Test the display_results method
        """
        # Prepare sample results
        sample_results = [
            {
                'subject': 'Test Subject',
                'status': 'TestStatus',
                'email_body': 'This is a test email body that might be quite long and needs truncation.'
            }
        ]
        
        query_manager.display_results(sample_results)
        
        # Capture output and check basic structure
        captured = capsys.readouterr()
        assert "Found 1 matching record(s)" in captured.out
        assert "Test Subject" in captured.out
        assert "TestStatus" in captured.out
    
    def test_connection_status(self):
        """
        Test Milvus connection handling
        """
        # Create manager without connecting
        manager = EmailQueryManager()
        
        with pytest.raises(EmailQueryError, match="Not connected to Milvus"):
            manager.query_emails({
                'category': 'test',
                'subject': 'test',
                'status': 'test'
            })

# Example usage scenarios for documentation
def example_usage():
    """
    Example usage scenarios demonstrating different ways to use EmailQueryManager
    """
    # Basic usage
    query_manager = EmailQueryManager()
    query_manager.connect()
    
    # Example 1: Standard query
    standard_query = {
        'category': 'predisbursal_loan_query_loan_ca',
        'subject': 'loan cancel request',
        'status': 'IMSanctioned'
    }
    results = query_manager.query_emails(standard_query)
    query_manager.display_results(results)
    
    # Example 2: Query with different collection
    alternative_query = {
        'category': 'another_collection',
        'subject': 'different subject',
        'status': 'different status'
    }
    results = query_manager.query_emails(alternative_query)
    query_manager.display_results(results)

if __name__ == "__main__":
    # This allows running the example usage directly
    example_usage() 