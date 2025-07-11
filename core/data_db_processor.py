import pandas as pd

def process_ticket_metadata(ticket_id: str, subject: str, body: str, category: str) -> dict:
    """
    Process ticket metadata from the data/datadb.csv file.
    
    Args:
        ticket_id (str): Ticket identifier
        subject (str): Ticket subject
        body (str): Ticket body
        category (str): Ticket category
    
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
        "query": f"{subject} {body}",
        "category": category,
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
    if not status_list:
        return f"{im_value}NoStatus"
    
    # Get loan status and repayment status
    loan_status = ""
    repayment_status = ""
    
    for status in status_list:
        if status in ['DISBURSED', 'CLOSED', 'UNDER_REVIEW', 'REJECTED', 'EXPIRED']:
            loan_status = status
        elif status in ['REGULAR', 'DELAYED_1', 'DELAYED_3', 'WRITTEN_OFF']:
            repayment_status = status
    
    # Logic for different IM scenarios
    if im_value == 'IM':
        if loan_status == 'REJECTED':
            return "IMRejectedBureau"
        elif loan_status == 'UNDER_REVIEW':
            return "IMUnderReview"
        elif loan_status == 'EXPIRED':
            return "IMExpired"
        elif loan_status == 'DISBURSED':
            if repayment_status:
                return f"IM{loan_status}{repayment_status}"
            else:
                return f"IM{loan_status}"
        elif loan_status == 'CLOSED':
            return f"IM{loan_status}"
        else:
            return f"IM{loan_status}" if loan_status else "IMNoStatus"
    
    elif im_value == 'IM+':
        if loan_status == 'DISBURSED':
            if repayment_status:
                return f"IM+{loan_status}{repayment_status}"
            else:
                return f"IM+{loan_status}"
        elif loan_status == 'CLOSED':
            return f"IM+{loan_status}"
        elif loan_status == 'UNDER_REVIEW':
            return "IM+UnderReview"
        else:
            return f"IM+{loan_status}" if loan_status else "IM+NoStatus"
    
    else:
        # For other IM values (IM++, IM-, etc.)
        if loan_status:
            return f"{im_value}{loan_status}{repayment_status}" if repayment_status else f"{im_value}{loan_status}"
        else:
            return f"{im_value}NoStatus"
