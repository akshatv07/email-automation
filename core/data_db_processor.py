import pandas as pd

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
    
    # Convert ticket_id to string to ensure consistent matching
    ticket_id = str(ticket_id)
    
    # Find the matching row (case-insensitive)
    row = df[df['Ticket ID'].astype(str) == ticket_id]
    
    # If row is empty, try alternative column
    if row.empty:
        row = df[df['ticket_id'].astype(str) == ticket_id]
    
    # Get the first row if found
    row = row.iloc[0] if not row.empty else None
    
    # Initialize status list
    status = []
    
    # Determine 'im' value
    im_value = 'IM'
    if row is not None:
        # Get the raw value and convert to string
        raw_im = str(row.get('data_from_IM_pls', '')).strip()
        
        # Replace 'nan' with 'IM'
        im_value = raw_im if raw_im and raw_im.lower() != 'nan' else 'IM'
    
    # Define all possible status columns
    status_columns = [
        'Loan Status', 
        'repayment_status', 
        'last_stage_checklist',
        'lr_status', 
        'disbursement_completion_date'
    ]
    
    # Extract status from all specified columns
    if row is not None:
        # Extract non-empty, non-NaN values and remove duplicates
        status = list(dict.fromkeys(
            str(row.get(col, '')).strip() 
            for col in status_columns 
            if pd.notna(row.get(col, '')) and str(row.get(col, '')).strip()
        ))
    
    # Create single line status based on IM value and status combination
    single_line_status = create_single_line_status(im_value, status, row)
    
    # Construct and return the result dictionary
    return {
        "ticket_id": ticket_id,
        "status": single_line_status,
        "im": im_value
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
    input_and_process_ticket()