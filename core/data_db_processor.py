import pandas as pd
import re

# Category sanitization mapping
CATEGORY_SANITIZATION_MAP = {
    "predisbursal_loan_query_loan_cancellation_request": "predisbursal_loan_query_loan_ca",
    "collection_query": "collection_query",
    "data_erasure_request__": "data_erasure_request",
    "predisbursal_loan_query_credit_team_action_pending": "predisbursal_loan_query_credit",
    "predisbursal_loan_query_loan_approved_disbursed": "predisbursal_loan_query_loan_ap",
    "predisbursal_loan_query_other_bs_isssues": "predisbursal_loan_query_other_b",
    "predisbursal_loan_query_bs_issue_maximum_attempt_reach": "predisbursal_loan_query_bs_issu",
    "predisbursal_loan_query_im+_instances_loc-live-withdrawal_request_placed": "predisbursal_loan_query_im_inst",
    "post_loan_closure_queries_surrender_of_im+limit": "post_loan_closure_queries_surre",
    "post_loan_disbursal_query_basic_emi_-ecs_details_emi_amount": "post_loan_disbursal_query_basic",
    "post_loan_closure_queries": "post_loan_closure_queries",
    "post_loan_disbursal_query_basic_emi_-ecs_details_ecs_approved,_but_not_triggered": "post_loan_disbursal_query_bas_1",
    "post_loan_disbursal_query_payment_ecs_payment": "post_loan_disbursal_query_payme",
    "post_loan_disbursal_query": "post_loan_disbursal_query",
    "post_loan_disbursal_query_basic_emi_-ecs_details_ecs_status": "post_loan_disbursal_query_bas_2",
    "canceled_loan_checking_for_reason_": "canceled_loan_checking_for_reas",
    "post_loan_disbursal_query_basic_emi_-ecs_details_emi_date_change_request": "post_loan_disbursal_query_bas_3",
    "update_-_edit_details_bank_account_details_": "update_edit_details_bank_accou",
    "update_-_edit_details_mobile_number": "update_edit_details_mobile_num",
    "update_-_edit_details_email_id": "update_edit_details_email_id",
    "post_loan_disbursal_query_basic_emi_-ecs_details_loan_closure_amount": "post_loan_disbursal_query_bas_4",
    "post_loan_disbursal_query_payment_dual_payment_not_updated": "post_loan_disbursal_query_pay_1",
    "post_loan_disbursal_query_payment_refund_request": "post_loan_disbursal_query_pay_2",
    "predisbursal_loan_query_general_info": "predisbursal_loan_query_general",
    "predisbursal_loan_query_im+_instances": "predisbursal_loan_query_im_in_1",
    "predisbursal_loan_query_incomplete_profile": "predisbursal_loan_query_incompl",
    "predisbursal_loan_query_loan_approved_disbursal_in_progress": "predisbursal_loan_approved_disb",
    "stop_marketing_sms-emails_details_added_in_the_sheet_": "stop_marketing_sms_emails_detai",
    "unregistered-no_content_registered_credentials_needed": "unregistered_no_content_registe",
    "collection_queries_settlement_query": "collection_queries_settlement_q",
    "post_loan_closure_queries_credit_report_issues": "post_loan_closure_queries_credi",
    "post_loan_closure_queries_loan_related_documents_required_": "post_loan_closure_queries_loan",
    "predisbursal_loan_query_kyc_issue_pan-aadhar_exists": "predisbursal_loan_query_kyc_iss",
    "predisbursal_loan_query_nach_issue_general_information": "predisbursal_loan_query_nach_is",
    "predisbursal_loan_query_nach_issue_unable_to_proceed_enach": "predisbursal_loan_query_nach_1",
    "predisbursal_loan_query_rf-vf_query_general_information": "predisbursal_loan_query_rf_vf_q",
    "predisbursal_loan_query_rf-vf_query_rf-vf_paid_not_updated": "predisbursal_loan_query_rf_vf_p",
    "rejected_loan_cancel_enach_": "rejected_loan_cancel_enach",
    "rejected_loan_checking_for_reason": "rejected_loan_checking_for_reas",
    "rejected_loan_requesting_for_refund_": "rejected_loan_requesting_for_re",
    "rejected_loan_wants_to_re-apply-re-consider": "rejected_loan_wants_to_re_apply",
    "update_-_edit_details_address": "update_edit_details_address",
    "update_-_edit_details_name": "update_edit_details_name"
}

def process_ticket_metadata(ticket_id: str) -> dict:
    """
    Process ticket metadata from the data/datadb.csv file.
    Args:
        ticket_id (str): Ticket identifier
    Returns:
        dict: Processed ticket metadata with single line status
    """
    # Read the CSV file
    df = pd.read_csv('data/datadb.csv')
    ticket_id = str(ticket_id)
    row = df[df['Ticket ID'].astype(str) == ticket_id]
    if row.empty:
        row = df[df['ticket_id'].astype(str) == ticket_id]
    row = row.iloc[0] if not row.empty else None

    # New logic: check data_from (IM value)
    if row is not None:
        data_from_im = str(row.get('data_from', '')).strip()
        if data_from_im and data_from_im.lower() != 'nan':
            # Gather status list for processing
            status_list = []
            loan_status = str(row.get('Loan Status', '')).strip()
            repayment_status = str(row.get('repayment_status', '')).strip()
            if loan_status and loan_status.lower() != 'nan':
                status_list.append(loan_status)
            if repayment_status and repayment_status.lower() != 'nan':
                status_list.append(repayment_status)
            # Fallback to lr_status if status_list is empty
            if not status_list:
                lr_status = str(row.get('lr_status', '')).strip()
                if lr_status and lr_status.lower() != 'nan':
                    status_list.append(lr_status)
            status = create_single_line_status(data_from_im, status_list, row)
            # Sanitize status: remove trailing _<number> if present
            status_sanitized = re.sub(r'_[0-9]+$', '', status)
            category_raw = str(row.get('new', '')).lower()
            category_sanitized = CATEGORY_SANITIZATION_MAP.get(category_raw, category_raw)
            return {
                "status": status_sanitized,
                "category": category_sanitized
            }
        else:
            lr_status = str(row.get('lr_status', '')).strip()
            # Sanitize status: remove trailing _<number> if present
            status_sanitized = re.sub(r'_[0-9]+$', '', lr_status)
            category_raw = str(row.get('new', '')).lower()
            category_sanitized = CATEGORY_SANITIZATION_MAP.get(category_raw, category_raw)
            return {
                "status": status_sanitized,
                "category": category_sanitized
            }
    return {
        "status": "not_found",
        "category": "not_found"
    }

def create_single_line_status(im_value: str, status_list: list, row) -> str:
    """
    Create a single line status by combining IM value with relevant status information.
    
    Args:
        im_value (str): IM value (IM, IM+, etc.)
        status_list (list): List of status values
        row: DataFrame row containing ticket data
    
    Returns:
        str: Combined single line status
    """
    # Determine if underscore should be added based on IM value
    def format_status(base_status: str) -> str:
        # Add underscore only if '+' is in the original im_value
        return f"{im_value.replace('+', '_').lower()}{base_status}" if '+' in im_value else f"{im_value.lower()}{base_status}"
    
    if not status_list:
        return format_status("nostatus")
    
    # Get loan status and repayment status
    loan_status = ""
    repayment_status = ""
    
    for status in status_list:
        if status in ['DISBURSED', 'CLOSED', 'UNDER_REVIEW', 'REJECTED', 'EXPIRED']:
            loan_status = status.lower()
        elif status in ['REGULAR', 'DELAYED_1', 'DELAYED_3', 'WRITTEN_OFF']:
            repayment_status = status.lower()
    
    # Logic for different IM scenarios
    if im_value == 'IM':
        if loan_status == 'rejected':
            return format_status("rejected")
        elif loan_status == 'under_review':
            return format_status("underreview")
        elif loan_status == 'expired':
            return format_status("expired")
        elif loan_status == 'disbursed':
            if repayment_status:
                return format_status(f"disbursed{repayment_status}")
            else:
                return format_status("disbursed")
        elif loan_status == 'closed':
            return format_status("closed")
        else:
            return format_status(loan_status) if loan_status else format_status("nostatus")
    
    elif im_value == 'IM+':
        if loan_status == 'disbursed':
            if repayment_status:
                return format_status(f"disbursed{repayment_status}")
            else:
                return format_status("disbursed")
        elif loan_status == 'closed':
            return format_status("closed")
        elif loan_status == 'under_review':
            return format_status("underreview")
        else:
            return format_status(loan_status) if loan_status else format_status("nostatus")
    
    else:
        # For other IM values (IM++, IM-, etc.)
        im_sanitized = im_value.replace('+', '_').lower()
        if loan_status:
            return f"{im_sanitized}{loan_status}{repayment_status}" if repayment_status else f"{im_sanitized}{loan_status}"
        else:
            return f"{im_sanitized}_nostatus"

def input_and_process_ticket():
    """
    Interactively input a ticket ID and process its metadata.
    
    Prompts the user to enter a ticket ID and displays the processed ticket information.
    """
    try:
        # Prompt for ticket ID input
        ticket_id = input("Enter the Ticket ID: ").strip()
        
        # Process the ticket metadata
        ticket_info = process_ticket_metadata(ticket_id)
        
        # Print the processed ticket information in a readable format
        print("\n--- Ticket Metadata ---")
        for key, value in ticket_info.items():
            print(f"{key.capitalize()}: {value}")
        
        return ticket_info
    
    except Exception as e:
        print(f"An error occurred while processing the ticket: {e}")
        return None

# Allow the script to be run directly for ticket input
if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticket-id', type=str, required=True, help='Ticket ID to process')
    args = parser.parse_args()
    ticket_info = process_ticket_metadata(args.ticket_id)
    print(json.dumps(ticket_info))