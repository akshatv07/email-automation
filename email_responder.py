import os
import argparse
import json
import sys
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv
import time
import boto3
from urllib.error import HTTPError
import bedrock  # Import the generate_llm_response_with_backoff function

# Load environment variables
load_dotenv()

class EmailTemplateGenerator:
    def __init__(self):
        # LLM Configuration from .env
        self.model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3.5-sonnet-20240620-v1:0')
        self.max_output_tokens = 2000
        self.temperature = 0.4
        self.top_p = 0.8
        self.top_k = 10

    def format_search_results(self, search_data: Dict[str, Any]) -> str:
        """Format search results for the LLM prompt."""
        if not search_data or 'results' not in search_data:
            return "No relevant information was found in our records."
        formatted = "Here are the relevant details from our system:\n\n"
        # Add metadata
        meta = search_data.get('metadata', {})
        if meta:
            formatted += f"Search Context:\n"
            if 'collection' in meta:
                formatted += f"- Collection: {meta['collection']}\n"
            if 'status_field' in meta:
                formatted += f"- Status Field: {meta['status_field']}\n"
            if 'results_count' in meta:
                formatted += f"- Found {meta['results_count']} results\n"
            formatted += "\n"
        # Add results
        for i, result in enumerate(search_data.get('results', [])[:3]):  # Limit to top 3
            formatted += f"Result {i+1}:\n"
            if 'status' in result and result['status']:
                formatted += f"- Status: {result['status']}\n"
            if 'fields' in result:
                for key, value in result['fields'].items():
                    if value and key.lower() != 'embedding':  # Skip embedding data
                        formatted += f"- {key}: {value}\n"
            formatted += "\n"
        return formatted

    def generate_response(self, search_data: Dict[str, Any], 
                         email_subject: str = "",
                         ticket_id: str = "") -> Dict[str, Any]:
        """
        Generate an email response based on search results using AWS Bedrock.
        """
        # Format the search results for the prompt
        knowledge_context = self.format_search_results(search_data)
        fallback_template_used = False
        fallback_template_content = None
        # Check for missing field error in search_data
        error_msg = None
        if isinstance(search_data, dict) and 'error' in search_data and 'Field' in search_data['error'] and 'not found in collection' in search_data['error']:
            error_msg = search_data['error']
            # Try to extract collection name from error message
            import re
            match = re.search(r"Field '.*' not found in collection '([^']+)'", error_msg)
            if match:
                collection_name = match.group(1)
                # Try to find a template file matching the collection name
                template_path = os.path.join('templates', f'{collection_name}.html')
                if os.path.exists(template_path):
                    with open(template_path, 'r', encoding='utf-8') as f:
                        fallback_template_content = f.read()
                        fallback_template_used = True
        try:
            if fallback_template_used and fallback_template_content:
                # Directly return the template content as the response, no LLM call
                return {
                    'status': 'success',
                    'email_response': fallback_template_content,
                    'metadata': {
                        'model': None,
                        'fallback_template_used': True,
                        'fallback_template': os.path.basename(template_path) if fallback_template_used else None,
                        'fallback_template_flag': 'FALL BACK TEMPLATE USED'
                    },
                    'search_metadata': search_data.get('metadata', {})
                }
            else:
                # Prepare the prompt for Bedrock as before
                prompt = f"""You are a helpful customer support agent.\nGenerate only the email body (do not include subject, ticket id, or any extra formatting).\nDo not include any label, heading, or prefix like 'Email Body:', 'Subject:', 'Ticket ID:', or '=== GENERATED EMAIL ==='. Absolutely do not include any such label. Output only the email content, starting directly with the greeting or first line of the email.\n\nRelevant Information:\n{knowledge_context}\n\nGuidelines:\n1. Acknowledge the customer's concern\n2. Provide clear information based on the search results\n3. Be concise but thorough\n4. Maintain a professional and helpful tone\n5. If the status is available, highlight it clearly\n6. End with a call to action or next steps\n\nPlease draft only the email body based on the above information."""
                start_time = time.time()
                # Use the Bedrock LLM
                generated_text = bedrock.generate_llm_response_with_backoff(
                    prompt,
                    max_tokens=self.max_output_tokens
                )
                elapsed = time.time() - start_time
                print(f"Time to generate LLM response: {elapsed:.2f} seconds")
                return {
                    'status': 'success',
                    'email_response': generated_text,
                    'metadata': {
                        'model': self.model_id,
                        'fallback_template_used': fallback_template_used,
                        'fallback_template': os.path.basename(template_path) if fallback_template_used else None
                    },
                    'search_metadata': search_data.get('metadata', {})
                }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'search_metadata': search_data.get('metadata', {}) if 'search_data' in locals() else {}
            }

def load_search_results(file_path: str) -> Optional[Dict[str, Any]]:
    """Load search results from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:  # Using utf-8-sig to handle BOM
            return json.load(f)
    except Exception as e:
        print(f"Error loading search results: {str(e)}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate email responses from search results')
    parser.add_argument('search_results', help='Path to JSON file with search results from search_emails.py')
    parser.add_argument('--subject', default='', help='Original email subject')
    parser.add_argument('--ticket-id', default='', help='Ticket ID for reference')
    parser.add_argument('--output', help='Output file path (optional)')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                      help='Output format (text or json)')
    args = parser.parse_args()
    search_data = load_search_results(args.search_results)
    if not search_data:
        print("Error: Could not load search results")
        return 1
    generator = EmailTemplateGenerator()
    result = generator.generate_response(
        search_data=search_data,
        email_subject=args.subject,
        ticket_id=args.ticket_id
    )
    if args.format == 'json':
        output = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        if result['status'] == 'success':
            output = f"=== GENERATED EMAIL ===\n\n"
            if args.ticket_id:
                output += f"Ticket ID: {args.ticket_id}\n"
            output += f"Subject: {args.subject or 'Re: Your Inquiry'}\n\n"
            output += result['email_response']
            output += f"\n\n=== END OF EMAIL ===\n"
        else:
            output = f"Error: {result.get('error', 'Unknown error')}"
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Response written to {args.output}")
    else:
        print(output)
    return 0

if __name__ == "__main__":
    sys.exit(main())
