import io
import mimetypes
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.errors import HttpError
import logging

from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""def upload_cv_to_drive(drive_service, data: bytes, filename: str) -> str:
    #
    #Uploads a CV to Google Drive and returns its webViewLink.
    #Matches the monolithic script's 'upload_cv_to_drive' function.
    #
    try:
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime)
        
        up = drive_service.files().create(
            media_body=media,
            body={"name": filename, "parents":[config.DRIVE_FOLDER_ID]}, 
            fields="webViewLink"
        ).execute()
        return up["webViewLink"]
    except HttpError as e:
        logging.error(f"Google Drive API error uploading CV '{filename}': {e}")
        raise 
    except Exception as e:
        logging.error(f"An unexpected error occurred uploading CV '{filename}': {e}")
        raise

"""


def upload_cv_to_drive(drive_service, data: bytes, filename: str) -> str:
    """
    Uploads a CV to a Shared Drive folder and returns its webViewLink.
    """
    try:
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime)
        
        up = drive_service.files().create(
            media_body=media,
            body={
                "name": filename,
                "parents": [config.DRIVE_FOLDER_ID]
            },
            supportsAllDrives=True,  # <- crucial for Shared Drives
            fields="webViewLink"
        ).execute()
        
        return up["webViewLink"]
    
    except HttpError as e:
        logging.error(f"Google Drive API error uploading CV '{filename}': {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred uploading CV '{filename}': {e}")
        raise
