# Manual Email Processor

## Overview
`manual_email_processor.py` is an interactive script that allows you to manually process individual emails by providing ticket details.

## Features
- Manually input ticket ID, subject, and email body
- Process emails through the existing workflow
- Generate responses using the semantic response engine
- Save results to an Excel file
- Detailed logging of the processing steps

## Usage

### Running the Script
```bash
python manual_email_processor.py
```

### Workflow
1. Enter the Ticket ID
2. Enter the Email Subject
3. Enter the Email Body (optional)
   - Press 'END' on a new line to skip
   - Or provide multiple lines of text
4. View the generated response
5. Repeat or quit by entering 'q'

#### Input Scenarios
1. **Full Input**
```
Enter Ticket ID (or 'q' to quit): 3633276
Enter Email Subject: Loan Payment Issue
Enter Email Body (optional, type 'END' on a new line to skip or finish):
I paid my loan amount but it's not showing.
Please help me resolve this.
END
```

2. **Skipping Email Body**
```
Enter Ticket ID (or 'q' to quit): 3633276
Enter Email Subject: Loan Payment Issue
Enter Email Body (optional, type 'END' on a new line to skip or finish):
END
```

### Output
- Generates a `manual_results.xlsx` file with processed emails
- Creates a `manual_processing.log` for detailed logging

## Requirements
- Python 3.7+
- Pandas
- Existing project dependencies

## Notes
- Email body is completely optional
- If no email body is provided, the subject will be used for processing
- Supports multi-line email body input 