import base64
from email.mime.text import MIMEText
from googleapiclient.errors import HttpError
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def send_acknowledgment_email(gmail_service, recipient_email: str, subject: str, body: str) -> bool:
    """
    Sends an acknowledgment email using the Gmail API.
    Returns True if successful, False otherwise.
    Matches the monolithic script's 'send_acknowledgment_email' function.
    """
    try:
        message = MIMEText(body, 'plain')
        message['to'] = recipient_email
        message['from'] = 'me' 
        message['subject'] = subject
        
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        send_message = gmail_service.users().messages().send(
            userId="me",
            body={'raw': raw_message}
        ).execute()
        logging.info(f"Sent acknowledgment to: {recipient_email}")
        return True
    except HttpError as error:
        logging.error(f"Failed to send acknowledgment email to {recipient_email}: {error}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while sending email to {recipient_email}: {e}")
        return False

