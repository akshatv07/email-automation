"""
Prompt templates for email response generation.
These templates are used by the email responder to generate consistent and professional responses.
"""

# Base system prompt that sets the tone and behavior of the assistant
SYSTEM_PROMPT = """You are a professional and empathetic customer support representative.
Your goal is to provide clear, accurate, and helpful responses to customer inquiries.
Always maintain a professional yet friendly tone and ensure all information is accurate."""

# Main response generation prompt template
RESPONSE_TEMPLATE = """
TICKET DETAILS:
- Ticket ID: {ticket_id}
- Response Type: {response_type}
- Customer Query: {customer_query}

KNOWLEDGE BASE CONTEXT:
{knowledge_context}

INSTRUCTIONS:
1. Carefully analyze the customer's query and the provided knowledge base context.
2. Generate a professional and helpful response that addresses all points in the customer's query.
3. If the information isn't available in the knowledge base, be honest and let the customer know you'll look into it.
4. Keep the response concise but thorough (under 600 characters).
5. Use proper email etiquette with appropriate greeting and closing.
6. If relevant, include next steps or additional resources.

RESPONSE:
"""

# Error response template for when no knowledge base results are found
NO_KNOWLEDGE_RESPONSE = """
I'm sorry, but I couldn't find specific information related to your query in our knowledge base.
Our team will look into this and get back to you as soon as possible.

Thank you for your patience.
"""

def get_prompt(ticket_id: str, customer_query: str, knowledge_context: str, response_type: str = 'general') -> dict:
    """
    Get the complete prompt for the LLM based on the provided parameters.
    
    Args:
        ticket_id: The ticket ID for reference
        customer_query: The customer's query or email content
        knowledge_context: Relevant context from the knowledge base
        response_type: Type of response (e.g., 'general', 'technical', 'billing')
        
    Returns:
        dict: Dictionary containing 'system' and 'user' prompts
    """
    if not knowledge_context.strip():
        knowledge_context = "No relevant information found in the knowledge base."
    
    user_prompt = RESPONSE_TEMPLATE.format(
        ticket_id=ticket_id,
        customer_query=customer_query,
        knowledge_context=knowledge_context,
        response_type=response_type
    )
    
    return {
        'system': SYSTEM_PROMPT,
        'user': user_prompt
    }
