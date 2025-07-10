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
        dict: Processed ticket metadata
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
    
    # Construct and return the result dictionary
    return {
        "ticket_id": ticket_id,
        "query": f"{subject} {body}",
        "category": category,
        "status": status,
        "im": im_value
    }
