from google.oauth2 import service_account
from googleapiclient.discovery import build
import logging

from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def authenticate_drive_service():
    """Authenticates with Google Drive API using a service account."""
    try:
        creds = service_account.Credentials.from_service_account_file(
            config.SHEETS_SERVICE_ACCOUNT_FILE, scopes=config.DRIVE_SCOPES 
        )
        service = build("drive", "v3", credentials=creds)
        logging.info("Google Drive service authenticated successfully.")
        return service
    except Exception as e:
        logging.error(f"Error authenticating Google Drive service: {e}")
        raise 
