from google.oauth2 import service_account
from googleapiclient.discovery import build
import logging

from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def authenticate_sheets_service():
    """Authenticates with Google Sheets API using a service account."""
    try:
        creds = service_account.Credentials.from_service_account_file(
            config.SHEETS_SERVICE_ACCOUNT_FILE, scopes=config.SHEETS_SCOPES
        )
        service = build("sheets", "v4", credentials=creds)
        logging.info("Google Sheets service authenticated successfully.")
        return service
    except Exception as e:
        logging.error(f"Error authenticating Google Sheets service: {e}")
        raise 

