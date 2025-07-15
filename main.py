import os
import sys
import subprocess
import json
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import settings

# === GLOBAL PLACEHOLDERS ===
EMAIL_BODY = "Hi team , I want to change my Bank account, pls help me with this. Thanks Abhishek R 9880804843"
SUBJECT = "Change bank account"
TICKET_ID ="3632479"  # Placeholder


def run_subprocess(cmd, step_name):
    print(f"[INFO] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=True)
        if result.stdout:
            print(f"[INFO] {step_name} stdout:\n{result.stdout}")
        if result.stderr:
            print(f"[WARN] {step_name} stderr:\n{result.stderr}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {step_name} failed with exit code {e.returncode}")
        print(f"[ERROR] stdout:\n{e.stdout}")
        print(f"[ERROR] stderr:\n{e.stderr}")
        raise


def run_data_db_processor(ticket_id):
    print("[INFO] Running data_db_processor to get status and category...")
    cmd = [
        'python', 'core/data_db_processor.py',
        '--ticket-id', ticket_id
    ]
    stdout = run_subprocess(cmd, "data_db_processor")
    output = json.loads(stdout)
    print(f"[INFO] data_db_processor complete. Status: {output['status']}, Category: {output['category']}")
    return output['status'], output['category']


def run_search_db_by_field(category, subject, body, status):
    print(f"[INFO] Running search_db_by_field for collection '{category}'...")
    output_file = 'search_results.json'
    cmd = [
        'python', 'search_db_by_field.py',
        '--collection', category,
        '--subject', subject,
        '--body', body,
        '--metadata', status,
        '--json',
        '--output', output_file
    ]
    run_subprocess(cmd, "search_db_by_field")
    print(f"[INFO] search_db_by_field complete. Results saved to {output_file}")
    return output_file


def run_email_responder(search_results_file, subject, ticket_id=''):
    print(f"[INFO] Running email_responder with results from {search_results_file}...")
    cmd = [
        'python', 'email_responder.py',
        search_results_file,
        '--subject', subject,
        '--ticket-id', ticket_id
    ]
    stdout = run_subprocess(cmd, "email_responder")
    print("[INFO] email_responder complete. Generated response:")
    print(stdout)
    return stdout


def main():
    print("[INFO] Starting main workflow...")
    # Read test data
    test_data = pd.read_csv('test_data.csv')
    results = []
    for idx, row in test_data.iterrows():
        ticket_id = str(row['Ticket'])
        subject = str(row['Subject'])
        email_body = str(row['Email Body'])
        try:
            # Step 1: Get status and category
            try:
                status, category = run_data_db_processor(ticket_id)
            except Exception as e:
                response = f"Failed at data_db_processor: {str(e)}"
                print(response)
                results.append({
                    'Ticket ID': ticket_id,
                    'Subject': subject,
                    'Email Body': email_body,
                    'Response': response
                })
                continue

            # Map the fetched category to the correct Milvus collection name if needed
            if category == 'predisbursal_loan_query_im+_instances':
                print("[INFO] Mapping category 'predisbursal_loan_query_im+_instances' to 'predisbursal_loan_query_im_in_1'")
                category = 'predisbursal_loan_query_im_in_1'
            if category == 'update_-_edit_details_bank_account_details_':
                print("[INFO] Mapping category 'update_-_edit_details_bank_account_details_' to 'update_edit_details_bank_accou'")
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
                print(response)
            elif not category:
                response = "Failed at category check: Category is empty. Cannot run search_db_by_field."
                print(response)
            else:
                # Step 2: Search for template
                try:
                    search_results_file = run_search_db_by_field(category, subject, email_body, status)
                except Exception as e:
                    response = f"Failed at search_db_by_field: {str(e)}"
                    print(response)
                    results.append({
                        'Ticket ID': ticket_id,
                        'Subject': subject,
                        'Email Body': email_body,
                        'Response': response
                    })
                    continue
                # Step 3: Generate LLM response
                try:
                    resp = run_email_responder(search_results_file, subject, ticket_id)
                    response = resp if resp else "Response generated successfully."
                except Exception as e:
                    response = f"Failed at email_responder: {str(e)}"
            results.append({
                'Ticket ID': ticket_id,
                'Subject': subject,
                'Email Body': email_body,
                'Response': response
            })
        except Exception as e:
            print(f"[FATAL] Workflow failed for ticket {ticket_id}: {e}")
            results.append({
                'Ticket ID': ticket_id,
                'Subject': subject,
                'Email Body': email_body,
                'Response': f"Failed at unknown step: {e}"
            })
    # Write results to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel('results.xlsx', index=False)
    print("[INFO] All results written to results.xlsx")


if __name__ == '__main__':
    main()
