import os
import argparse
import json
import sys
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv
import google.generativeai as genai
from prompt_templates import get_prompt

# Load environment variables
load_dotenv()

class EmailTemplateGenerator:
    def __init__(self):
        # LLM Configuration
        self.llm_config = {
            'api_key': 'AIzaSyBYvn_uYjajkf7_ELjqzsY2o0awESsqjSg',  # Using the provided API key
            'model': 'gemini-1.5-flash',
            'max_output_tokens': 1000,
            'temperature': 0.3
        }
        
        # Configure the Gemini client
        genai.configure(api_key=self.llm_config['api_key'])
        self.model = genai.GenerativeModel(self.llm_config['model'])
        self.generation_config = {
            'temperature': self.llm_config['temperature'],
            'max_output_tokens': self.llm_config['max_output_tokens']
        }
        
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
        Generate an email response based on search results.
        
        Args:
            search_data: Dictionary containing search results from search_emails.py
            email_subject: The original email subject
            ticket_id: Optional ticket ID for reference
            
        Returns:
            Dict containing the generated response and original search data
        """
        # Format the search results for the prompt
        knowledge_context = self.format_search_results(search_data)
        
        try:
            # Prepare the prompt for Gemini
            prompt = f"""You are a helpful customer support agent. 
Generate a professional and empathetic email response based on the provided information.
Focus on being clear, concise, and helpful.

Email Subject: {email_subject}
Ticket ID: {ticket_id if ticket_id else 'Not provided'}

Relevant Information:
{knowledge_context}

Guidelines:
1. Acknowledge the customer's concern
2. Provide clear information based on the search results
3. Be concise but thorough
4. Maintain a professional and helpful tone
5. If the status is available, highlight it clearly
6. End with a call to action or next steps

Please draft an email response based on the above information:"""
            
            # Generate response using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.llm_config['temperature'],
                    max_output_tokens=self.llm_config['max_output_tokens']
                )
            )
            
            # Extract the generated text safely
            try:
                generated_text = response.text.strip()
            except Exception as e:
                generated_text = str(response)
                
            # Try to get token usage if available
            tokens_used = None
            try:
                if hasattr(response, 'usage_metadata'):
                    if hasattr(response.usage_metadata, 'total_tokens'):
                        tokens_used = response.usage_metadata.total_tokens
                    elif hasattr(response.usage_metadata, 'candidates_token_count'):
                        tokens_used = response.usage_metadata.candidates_token_count
            except Exception:
                pass
                
            return {
                'status': 'success',
                'email_response': generated_text,
                'metadata': {
                    'model': self.llm_config['model'],
                    'tokens_used': tokens_used
                },
                'search_metadata': search_data.get('metadata', {})
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'search_metadata': search_data.get('metadata', {}) if 'search_data' in locals() else {}
            }

def load_search_results(file_path: str) -> Dict[str, Any]:
    """Load search results from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:  # Using utf-8-sig to handle BOM
            return json.load(f)
    except Exception as e:
        print(f"Error loading search results: {str(e)}")
        return None

def main():
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate email responses from search results')
    parser.add_argument('search_results', help='Path to JSON file with search results from search_emails.py')
    parser.add_argument('--subject', default='', help='Original email subject')
    parser.add_argument('--ticket-id', default='', help='Ticket ID for reference')
    parser.add_argument('--output', help='Output file path (optional)')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                      help='Output format (text or json)')
    
    args = parser.parse_args()
    
    # Load search results
    search_data = load_search_results(args.search_results)
    if not search_data:
        print("Error: Could not load search results")
        return 1
    
    # Initialize the template generator
    generator = EmailTemplateGenerator()
    
    # Generate the response
    result = generator.generate_response(
        search_data=search_data,
        email_subject=args.subject,
        ticket_id=args.ticket_id
    )
    
    # Output the results
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
    
    # Write to file or print to console
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Response written to {args.output}")
    else:
        print(output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
