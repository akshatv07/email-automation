import os
import sys
import subprocess
import json
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import settings

# === GLOBAL PLACEHOLDERS ===
EMAIL_BODY = "Hi team , I want to change my Bank account, pls help me with this. Thanks Abhishek R 9880804843"
SUBJECT = "Change bank account"
TICKET_ID ="3632479"  # Placeholder

def run_subprocess(cmd, step_name):
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
    logging.info(f"Running search_db_by_field for collection '{category}'")
    cmd = [
        'python', 'search_db_by_field.py',
        '--collection', category,
        '--subject', subject,
        '--body', body,
        '--metadata', status
    ]
    stdout = run_subprocess(cmd, "search_db_by_field")
    logging.info(f"search_db_by_field complete. Results returned via stdout")
    return stdout

def run_email_responder(search_results_file, subject, ticket_id=''):
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

def main(input_file='test_data.csv', resume=False):
    logging.info(f"Starting batch processing from {input_file}")
    
    # Check if results file exists and resume is requested
    results_file = 'results.xlsx'
    if resume and os.path.exists(results_file):
        existing_results = pd.read_excel(results_file)
        logging.info(f"Resuming processing. Found {len(existing_results)} existing results.")
    else:
        existing_results = pd.DataFrame(columns=pd.Index(['Ticket ID', 'Subject', 'Email Body', 'Response']))
    
    # Read test data
    try:
        test_data = pd.read_csv(input_file)
        # Debug: Log exact column names
        logging.info(f"Columns found in CSV: {list(test_data.columns)}")
    except Exception as e:
        logging.error(f"Error reading input file {input_file}: {e}")
        return
    
    # Validate required columns
    required_columns = ['ticket', 'subject', 'email_body']
    
    # Rename columns to match expected names
    column_mapping = {
        'Ticket ID': 'ticket',
    }
    test_data.rename(columns=column_mapping, inplace=True)
    
    # Recheck missing columns after renaming
    missing_columns = [col for col in required_columns if col not in test_data.columns]
    if missing_columns:
        logging.error(f"Missing columns in input file: {missing_columns}")
        return
    
    results = list(existing_results.to_dict('records'))
    processed_tickets = set(str(ticket) for ticket in existing_results['ticket']) if not existing_results.empty else set()
    
    for idx, row in test_data.iterrows():
        ticket_id = str(row['ticket'])
        
        # Skip already processed tickets if resuming
        if ticket_id in processed_tickets:
            logging.info(f"Skipping already processed ticket {ticket_id}")
            continue
        
        subject = str(row['subject'])
        email_body = str(row['email_body'])
        
        try:
            # Step 1: Get status and category
            try:
                status, category = run_data_db_processor(ticket_id)
                
                # Convert 'im_closed' to 'imclosed'
                if status == 'im_closed':
                    logging.info(f"Converting status from 'im_closed' to 'imclosed'")
                    status = 'imclosed'
            except Exception as e:
                response = f"Failed at data_db_processor: {str(e)}"
                logging.error(response)
                results.append({
                    'ticket': ticket_id,
                    'subject': subject,
                    'email_body': email_body,
                    'Response Generated': response,
                    'Template referred': ''
                })
                continue

            # Map the fetched category to the correct Milvus collection name if needed
            if category == 'predisbursal_loan_query_im+_instances':
                logging.info("Mapping category 'predisbursal_loan_query_im+_instances' to 'predisbursal_loan_query_im_in_1'")
                category = 'predisbursal_loan_query_im_in_1'
            if category == 'update_-_edit_details_bank_account_details_':
                logging.info("Mapping category 'update_-_edit_details_bank_account_details_' to 'update_edit_details_bank_accou'")
                category = 'update_edit_details_bank_accou'

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
                template_subject = ''
                template_error_reason = ''
                try:
                    search_results_stdout = run_search_db_by_field(category, subject, email_body, status)
                    # Try to extract template subject or error from stdout
                    try:
                        search_data = None
                        try:
                            search_data = json.loads(search_results_stdout)
                        except Exception:
                            pass
                        if search_data and 'error' in search_data:
                            template_error_reason = search_data['error']
                        elif search_data and 'results' in search_data and isinstance(search_data['results'], list) and len(search_data['results']) > 0 and 'fields' in search_data['results'][0] and 'subject' in search_data['results'][0]['fields']:
                            template_subject = search_data['results'][0]['fields']['subject']
                        elif 'No matching records found' in search_results_stdout:
                            template_error_reason = 'No matching records found in search results.'
                        elif 'not found' in search_results_stdout:
                            template_error_reason = search_results_stdout.strip()
                    except Exception as e:
                        template_error_reason = f'Error extracting template subject: {e}'
                except Exception as e:
                    template_error_reason = str(e)
                    logging.error(f"No template found: {template_error_reason}")
                    results.append({
                        'ticket': ticket_id,
                        'subject': subject,
                        'email_body': email_body,
                        'Response Generated': f"No template found: {template_error_reason}",
                        'Template referred': template_subject
                    })
                    continue
                if template_error_reason:
                    results.append({
                        'ticket': ticket_id,
                        'subject': subject,
                        'email_body': email_body,
                        'Response Generated': f"No template found: {template_error_reason}",
                        'Template referred': template_subject
                    })
                    continue
                # Step 3: Generate LLM response
                responder_subject = subject
                if responder_subject.lower() == 'nan' or responder_subject.strip() == '' or responder_subject == 'none':
                    if email_body and email_body.lower() != 'nan' and email_body.strip() != '' and email_body.lower() != 'none':
                        responder_subject = email_body
                    else:
                        responder_subject = '[No Subject Provided]'
                try:
                    resp = run_email_responder(search_results_stdout, responder_subject, ticket_id)
                    response = resp if resp else "Response generated successfully."
                except Exception as e:
                    response = f"Failed to generate response: {str(e)}"
                    logging.error(response)
                results.append({
                    'ticket': ticket_id,
                    'subject': subject,
                    'email_body': email_body,
                    'Response Generated': response,
                    'Template referred': template_subject
                })
            
            # Write results after each successful processing to allow resuming
            results_df = pd.DataFrame(results)
            results_df.to_excel(results_file, index=False)
            
        except Exception as e:
            logging.error(f"Workflow failed for ticket {ticket_id}: {e}")
            results.append({
                'ticket': ticket_id,
                'subject': subject,
                'email_body': email_body,
                'Response Generated': f"Failed at unknown step: {e}",
                'Template referred': ''
            })
    
    logging.info("[INFO] All results written to results.xlsx")

if __name__ == '__main__':
    # Optional: Add command-line argument parsing for input file and resume option
    import argparse
    parser = argparse.ArgumentParser(description='Batch process email responses')
    parser.add_argument('--input', default='test_data.csv', help='Input CSV file')
    parser.add_argument('--resume', action='store_true', help='Resume from previous results')
    args = parser.parse_args()
    
    main(input_file=args.input, resume=args.resume)
