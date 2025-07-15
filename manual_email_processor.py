import os
import sys
import json
import logging
import subprocess
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('manual_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def run_subprocess(cmd, step_name):
    """Run a subprocess command with logging."""
    logging.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=True)
        if result.stdout:
            logging.info(f"{step_name} stdout:\n{result.stdout}")
        if result.stderr:
            logging.warning(f"{step_name} stderr:\n{result.stderr}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"{step_name} failed with exit code {e.returncode}")
        logging.error(f"stdout:\n{e.stdout}")
        logging.error(f"stderr:\n{e.stderr}")
        raise

def run_data_db_processor(ticket_id):
    """Run data_db_processor to get status and category."""
    logging.info(f"Running data_db_processor for ticket {ticket_id}")
    cmd = [
        'python', 'core/data_db_processor.py',
        '--ticket-id', ticket_id
    ]
    stdout = run_subprocess(cmd, "data_db_processor")
    output = json.loads(stdout)
    logging.info(f"data_db_processor complete. Status: {output['status']}, Category: {output['category']}")
    return output['status'], output['category']

def run_search_db_by_field(category, subject, body, status):
    """Run search_db_by_field to find relevant templates."""
    logging.info(f"Running search_db_by_field for collection '{category}'")
    output_file = f'search_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    # Sanitize inputs to prevent command injection or unexpected behavior
    def sanitize_input(input_str):
        # Remove or escape special characters that might cause issues in shell commands
        if input_str is None:
            return ''
        return input_str.replace('"', '\\"').replace('`', '\\`').replace('$', '\\$')
    
    category = sanitize_input(category)
    subject = sanitize_input(subject)
    body = sanitize_input(body)
    status = sanitize_input(status)
    
    # Fallback to subject if body is empty
    search_body = body if body and body.strip() else subject
    
    cmd = [
        'python', 'search_db_by_field.py',
        '--collection', category,
        '--subject', subject,
        '--body', search_body,
        '--metadata', status,
        '--json',
        '--output', output_file
    ]
    
    try:
        logging.info(f"Search command: {' '.join(cmd)}")
        run_subprocess(cmd, "search_db_by_field")
        logging.info(f"Search results saved to {output_file}")
        return output_file
    except Exception as e:
        # Log detailed error information
        logging.error(f"Search failed: {str(e)}")
        
        # Additional diagnostic logging
        logging.info("Attempting to list available collections...")
        try:
            list_cmd = ['python', 'inspect_collections.py']
            collections_output = subprocess.check_output(list_cmd, text=True)
            logging.info(f"Available collections:\n{collections_output}")
        except Exception as list_error:
            logging.error(f"Could not list collections: {list_error}")
        
        # Raise a more informative error
        raise RuntimeError(f"Search failed for collection {category}. See logs for details.") from e

def run_email_responder(search_results_file, subject, ticket_id=''):
    """Run email_responder to generate a response."""
    logging.info(f"Running email_responder with results from {search_results_file}")
    cmd = [
        'python', 'email_responder.py',
        search_results_file,
        '--subject', subject,
        '--ticket-id', ticket_id
    ]
    stdout = run_subprocess(cmd, "email_responder")
    logging.info("email_responder complete. Generated response.")
    return stdout

def process_manual_email(ticket, subject, email_body=None):
    """Process a single email with manual input."""
    results = []
    try:
        # Step 1: Get status and category
        try:
            status, category = run_data_db_processor(ticket)
            
            # Convert 'im_closed' to 'imclosed'
            if status == 'im_closed':
                logging.info(f"Converting status from 'im_closed' to 'imclosed'")
                status = 'imclosed'
        except Exception as e:
            response = f"Failed at data_db_processor: {str(e)}"
            logging.error(response)
            return [{
                'ticket': ticket,
                'subject': subject,
                'email_body': email_body or '',
                'Response': response
            }]

        # Log the final status after potential conversion
        logging.info(f"Final status for ticket {ticket}: {status}")

        # Map categories if needed
        if category == 'predisbursal_loan_query_im+_instances':
            logging.info("Mapping category 'predisbursal_loan_query_im+_instances' to 'predisbursal_loan_query_im_in_1'")
            category = 'predisbursal_loan_query_im_in_1'
        if category == 'update_-_edit_details_bank_account_details_':
            logging.info("Mapping category 'update_-_edit_details_bank_account_details_' to 'update_edit_details_bank_accou'")
            category = 'update_edit_details_bank_accou'

        # Skip certain categories
        skip_search_categories = [
            'post_loan_disbursal_query_payment_lndn_payment_',
            'predisbursal_loan_query_rf/vf_query_general_information',
            'escalations_rbi-cyber_cell_',
            'escalations_singledebt_',
            'post_loan_disbursal_query_payment_paytm_payment_not_updated',
            'other_kyc_issues',
            'predisbursal_loan_query_im+_instances_unable_to_place_withdrawal',
        ]

        if category in skip_search_categories:
            response = f"Skipped search_db_by_field for category: {category}"
            logging.info(response)
        elif not category:
            response = "Failed at category check: Category is empty. Cannot run search_db_by_field."
            logging.error(response)
        else:
            # Step 2: Search for template
            # Use subject as body if email_body is not provided
            search_body = email_body if email_body else subject
            try:
                search_results_file = run_search_db_by_field(category, subject, search_body, status)
            except Exception as e:
                response = f"Failed at search_db_by_field: {str(e)}"
                logging.error(response)
                return [{
                    'ticket': ticket,
                    'subject': subject,
                    'email_body': email_body or '',
                    'Response': response
                }]
            
            # Step 3: Generate LLM response
            try:
                resp = run_email_responder(search_results_file, subject, ticket)
                response = resp if resp else "Response generated successfully."
            except Exception as e:
                response = f"Failed at email_responder: {str(e)}"
                logging.error(response)

        # Append results
        results.append({
            'ticket': ticket,
            'subject': subject,
            'email_body': email_body or '',
            'Response': response
        })

        # Save results to Excel
        results_df = pd.DataFrame(results)
        results_file = 'manual_results.xlsx'
        results_df.to_excel(results_file, index=False)
        logging.info(f"Results saved to {results_file}")

        return results

    except Exception as e:
        logging.error(f"Workflow failed for ticket {ticket}: {e}")
        results.append({
            'ticket': ticket,
            'subject': subject,
            'email_body': email_body or '',
            'Response': f"Failed at unknown step: {e}"
        })
        return results

def main():
    """Interactive manual email processing."""
    print("Manual Email Processor")
    print("---------------------")
    
    while True:
        # Get user input
        ticket = input("Enter Ticket ID (or 'q' to quit): ").strip()
        if ticket.lower() == 'q':
            break
        
        subject = input("Enter Email Subject: ").strip()
        
        # Optional email body
        print("Enter Email Body (optional, type 'END' on a new line to skip or finish):")
        email_body_lines = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            email_body_lines.append(line)
        
        email_body = '\n'.join(email_body_lines) if email_body_lines else None
        
        # Process the email
        results = process_manual_email(ticket, subject, email_body)
        
        # Print the response
        for result in results:
            print("\nProcessing Result:")
            print(f"Ticket ID: {result['ticket']}")
            print(f"Subject: {result['subject']}")
            if result['email_body']:
                print(f"Email Body: {result['email_body']}")
            print(f"Response: {result['Response']}")
        
        print("\n")

if __name__ == '__main__':
    main() 