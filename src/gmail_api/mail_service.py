import base64
import re
import logging
from googleapiclient.errors import HttpError
from pathlib import Path 
from typing import Tuple

from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_email_body_text(message_payload) -> str:
    """
    Extracts the plain text body from a Gmail message payload.
    Handles multipart messages by looking for the 'text/plain' part.
    """
    if 'parts' in message_payload:
        for part in message_payload['parts']:
            mime_type = part.get('mimeType')
            if mime_type == 'text/plain' and 'body' in part and 'data' in part['body']:
                return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
            if 'parts' in part:
                nested_body = get_email_body_text(part)
                if nested_body:
                    return nested_body
    elif 'body' in message_payload and 'data' in message_payload['body']:
        return base64.urlsafe_b64decode(message_payload['body']['data']).decode('utf-8')
    return ""


def get_sender_email(message_payload) -> str:
    """
    Extracts the sender's email address from a Gmail API message payload's 'From' header.
    Matches the monolithic script's regex for sender email.
    """
    headers = message_payload.get('headers', [])
    for header in headers:
        if header.get('name') == 'From':
            from_value = header.get('value', '')
            match = re.search(r'<(.*?)>', from_value)
            if match:
                return match.group(1)
            else:
                return from_value.strip() 
    return ""


def parse_message_headers(message_payload) -> Tuple[str, str, str]:
    """
    Extracts 'From' (full string), 'Subject', and 'Date' headers from a Gmail message payload.
    Matches the monolithic script's 'read_headers' functionality.
    """
    headers = message_payload.get("headers", [])
    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "")
    sender_full_string = next((h["value"] for h in headers if h["name"] == "From"), "")
    date = next((h["value"] for h in headers if h["name"] == "Date"), "")
    return sender_full_string, subject, date


def get_message_details(gmail_service, message_id: str):
    """
    Fetches the full details of a specific Gmail message.
    Matches the inline logic in the monolithic script's run_once.
    """
    try:
        message = gmail_service.users().messages().get(userId='me', id=message_id, format='full').execute()
        return message
    except HttpError as e:
        if e.resp.status == 404:
            logging.warning(f"Message ID {message_id} not found. It might have been deleted or moved. Error: {e}")
        else:
            logging.error(f"Error fetching message details for ID {message_id}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching message details for ID {message_id}: {e}")
        return None


def fetch_messages(gmail_service, query: str = config.GMAIL_EMAIL_QUERY, max_results: int = config.GMAIL_MAX_FETCH):
    """
    Fetches a list of message IDs from Gmail based on a query.
    Matches the monolithic script's 'fetch_messages' function.
    """
    try:
        results = gmail_service.users().messages().list(
            userId="me", q=query, maxResults=max_results
        ).execute()
        return results.get("messages", [])
    except Exception as e:
        logging.error(f"ERROR: An error occurred while fetching message list: {e}")
        return []


def get_attachments(gmail_service, message_id: str):
    """
    Fetches attachments from a specific Gmail message.
    Returns a list of dictionaries, each containing 'filename' and 'data' (bytes).
    Matches the inline attachment logic in the monolithic script's run_once.
    """
    attachments_list = []
    try:
        message = gmail_service.users().messages().get(userId='me', id=message_id, format='full').execute()
        parts = message['payload'].get('parts', [])

        for part in parts:
            if part.get('filename') and part['body'].get('attachmentId'):
                ext = Path(part['filename']).suffix.lower() 
                if ext in config.ALLOWED_RESUME_EXTENSIONS: 
                    attachment_id = part['body']['attachmentId']
                    att = gmail_service.users().messages().attachments().get(
                        userId='me', messageId=message_id, id=attachment_id
                    ).execute()
                    file_data = base64.urlsafe_b64decode(att['data'])
                    attachments_list.append({
                        'filename': part['filename'],
                        'data': file_data
                    })
                    break 

    except HttpError as e:
        logging.error(f"Google API error fetching attachments for message ID {message_id}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching attachments for message ID {message_id}: {e}")
    return attachments_list


def mark_message_as_read(gmail_service, message_id: str):
    """
    Marks a Gmail message as read.
    (This functionality is an improvement over the monolithic script, which didn't mark as read).
    """
    try:
        gmail_service.users().messages().modify(
            userId='me',
            id=message_id,
            body={'removeLabelIds': ['UNREAD']}
        ).execute()
        logging.info(f"Message ID {message_id} marked as read.")
    except HttpError as e:
        logging.error(f"Google API error marking message ID {message_id} as read: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while marking message ID {message_id} as read: {e}")

