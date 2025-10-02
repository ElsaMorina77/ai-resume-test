from googleapiclient.errors import HttpError
import logging

from config import config
from src.sheets_api.sheet_writer import SHEETS_DEFAULT_HEADER 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_existing_applicants_data(service, sheet_id=config.SHEETS_ID, tab_name=config.SHEETS_TAB_NAME):
    """
    Fetches existing applicant data from the Google Sheet for duplicate detection purposes.
    The deduplication logic is aligned with the monolithic script, but the return structure
    is a dictionary for better organization and to avoid the previous "tuple indices" error.

    Args:
        service (googleapiclient.discovery.Resource): Authenticated Google Sheets service object.
        sheet_id (str): The ID of the Google Sheet.
        tab_name (str): The name of the tab to read from.

    Returns:
        dict: A dictionary containing sets of unique identifiers and a map from email to CV link:
              {
                  'msg_ids': set of message IDs (str),
                  'names_emails': set of (name_lower, email_lower) tuples,
                  'phones': set of phone numbers (str, lowercased),
                  'email_to_cv_link': dict mapping email (lowercase) to CV link (str)
              }
              Returns empty sets/dict if no data or an error occurs.
    """

    last_column_char = chr(ord('A') + len(SHEETS_DEFAULT_HEADER) - 1)
    range_to_fetch = f"{tab_name}!A2:{last_column_char}" 
    
    existing_msg_ids = set()
    existing_names_emails = set()
    existing_phones = set()
    existing_email_to_cv_link = {}

    try:
        resp = service.spreadsheets().values().get(
            spreadsheetId=sheet_id, range=range_to_fetch
        ).execute()
        
        for row in resp.get("values", []):

            msg_id = row[0] if len(row) > 0 else ""
            cv_link = row[4] if len(row) > 4 else ""
            applicant_name_from_cv = row[5].strip().lower() if len(row) > 5 else ""
            applicant_email_cv = row[6].strip().lower() if len(row) > 6 else ""
            applicant_phone = row[7].strip().lower() if len(row) > 7 else ""

            if msg_id:
                existing_msg_ids.add(msg_id)

            if applicant_name_from_cv and applicant_email_cv:
                existing_names_emails.add((applicant_name_from_cv, applicant_email_cv))
                if applicant_email_cv not in existing_email_to_cv_link and cv_link:
                    existing_email_to_cv_link[applicant_email_cv] = cv_link
            
            if applicant_phone:
                existing_phones.add(applicant_phone)

        logging.info(f"Loaded {len(existing_msg_ids)} existing message IDs for deduplication.")
        logging.info(f"Loaded {len(existing_names_emails)} existing name-email combos for deduplication.")
        logging.info(f"Loaded {len(existing_phones)} existing phone numbers for deduplication.")

        return {
            'msg_ids': existing_msg_ids,
            'names_emails': existing_names_emails,
            'phones': existing_phones,
            'email_to_cv_link': existing_email_to_cv_link
        }

    except HttpError as e:
        logging.error(f"Google Sheets API error while fetching existing applicant data: {e}")
        return {
            'msg_ids': set(),
            'names_emails': set(),
            'phones': set(),
            'email_to_cv_link': {}
        }
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching existing applicant data: {e}")
        return {
            'msg_ids': set(),
            'names_emails': set(),
            'phones': set(),
            'email_to_cv_link': {}
        }

