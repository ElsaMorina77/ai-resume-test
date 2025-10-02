from googleapiclient.errors import HttpError
import logging

from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SHEETS_DEFAULT_HEADER = [
    "MsgID", "From (Sender)", "Subject", "Date", "CV Link", "Name (CV)", "Email (CV)",
    "Phone", "Job Applied For", "Status", "Acknowledgment Email Sent", "Timestamp" 
]

def ensure_tab_exists(service, sheet_id=config.SHEETS_ID, tab_name=config.SHEETS_TAB_NAME):
    """
    Ensures that the specified tab/sheet exists within the Google Sheet.
    (This is an improvement over the monolithic script which didn't explicitly check for tab existence).
    """
    try:
        spreadsheet = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
        sheets = spreadsheet.get('sheets', [])
        
        tab_found = False
        for sheet in sheets:
            if sheet['properties']['title'] == tab_name:
                tab_found = True
                break
        
        if not tab_found:
            logging.info(f"Tab '{tab_name}' not found. Creating tab...")
            batch_update_request = {
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': tab_name
                        }
                    }
                }]
            }
            service.spreadsheets().batchUpdate(
                spreadsheetId=sheet_id,
                body=batch_update_request
            ).execute()
            logging.info(f"Tab '{tab_name}' created successfully.")
        else:
            logging.info(f"Tab '{tab_name}' already exists.")

    except HttpError as e:
        logging.error(f"Google Sheets API error ensuring tab existence: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred ensuring tab existence: {e}")
        raise


def ensure_header(service, sheet_id=config.SHEETS_ID, tab_name=config.SHEETS_TAB_NAME, header=SHEETS_DEFAULT_HEADER):
    """
    Ensures the header row in the Google Sheet matches the expected format.
    Matches the monolithic script's 'ensure_header' function, now using a defined header.
    """
    try:
        resp = service.spreadsheets().values().get(
            spreadsheetId=sheet_id, range=f"{tab_name}!A1:Z1" 
        ).execute()

        current_header = resp.get("values", [[]])[0] 
        
        if not current_header or len(current_header) < len(header) or current_header != header:
            logging.info("Sheet header is missing or incomplete. Updating header...")
            service.spreadsheets().values().update(
                spreadsheetId=sheet_id,
                range=f"{tab_name}!A1",
                valueInputOption="USER_ENTERED",
                body={"values":[header]}
            ).execute()
            logging.info("Sheet header updated successfully.")
        else:
            logging.info("Sheet header is up-to-date.")

    except HttpError as e:
        logging.error(f"Google Sheets API error ensuring header: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred ensuring header: {e}")
        raise


def append_rows(service, rows, sheet_id=config.SHEETS_ID, tab_name=config.SHEETS_TAB_NAME):
    """
    Appends multiple rows of data to the Google Sheet.
    Matches the monolithic script's 'append_rows' function.
    """
    if not rows:
        logging.info("No rows to append to Google Sheet.")
        return
    try:
        service.spreadsheets().values().append(
            spreadsheetId=sheet_id,
            range=f"{tab_name}!A1", 
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values":rows}
        ).execute()
        logging.info(f"Successfully appended {len(rows)} row(s) to Google Sheet.")
    except HttpError as e:
        logging.error(f"Google Sheets API error appending rows: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred appending rows: {e}")
        raise

