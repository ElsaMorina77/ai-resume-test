
import time
import os
import re 
from datetime import datetime
import logging
from googleapiclient.errors import HttpError
import sys 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from config import config
from src.gmail_api import auth as gmail_auth, mail_service, mail_sender
from src.sheets_api import auth as sheets_auth, sheet_writer, sheet_reader
from src.drive_api import auth as drive_auth, file_uploader
from src.resume_parser import data_parser 
from src.bot_logic import job_matcher
from googleapiclient.discovery import build 


def process_email_and_attachment(gmail_service, sheets_service, drive_service, message, sheet_id, tab_name, existing_data):
    """
    Processes a single email, extracts applicant data, handles attachments,
    updates Google Sheets, and sends acknowledgment emails.
    Orchestrates the logic from the monolithic script's main loop for one email.
    """
    msg_id = message['id']
    logging.info(f"Processing email with MsgID: {msg_id}")

    if msg_id in existing_data['msg_ids']:
        logging.info(f"MsgID {msg_id} already exists in Google Sheet. Skipping.")
        return

    full_message_details = mail_service.get_message_details(gmail_service, msg_id)
    if not full_message_details:
        logging.error(f"Could not retrieve full message details for MsgID {msg_id}. Skipping.")
        return

    sender_full_string, subject_line, date_header = mail_service.parse_message_headers(full_message_details['payload'])
    sender_email_for_reply = mail_service.get_sender_email(full_message_details['payload'])
    email_body_content = mail_service.get_email_body_text(full_message_details['payload'])


    attachments = mail_service.get_attachments(gmail_service, msg_id)
    cv_attachment = next((a for a in attachments if a['filename'].lower().endswith(tuple(config.ALLOWED_RESUME_EXTENSIONS))), None)

    if not cv_attachment:
        logging.warning(f"No valid CV attachment found for email MsgID: {msg_id}. Skipping.")
        return

    cv_filename = cv_attachment['filename']
    cv_data = cv_attachment['data']
    logging.info(f"Found CV attachment: {cv_filename}")

    parsed_applicant_data = data_parser.parse_resume_data(cv_data, cv_filename)
    logging.debug(f"Parsed data for message ID {msg_id}: {parsed_applicant_data}")

    applicant_name_from_cv = parsed_applicant_data.get('name', '')
    applicant_email_from_cv = parsed_applicant_data.get('email_cv', '')
    applicant_phone_from_cv = parsed_applicant_data.get('phone', '')
    full_cv_text_content = parsed_applicant_data.get('full_text_content', '') 

    if not applicant_name_from_cv and not applicant_email_from_cv:
        logging.warning(f"Could not extract sufficient applicant info (name/email) from CV for MsgID {msg_id}. Skipping.")
        return

    inferred_job_title = job_matcher.infer_job_title(subject_line, email_body_content, full_cv_text_content)
    if not inferred_job_title:
        inferred_job_title = "Unspecified"

    is_duplicate = False
    cv_link_for_sheet = ""

    if applicant_name_from_cv and applicant_email_from_cv:
        name_email_combo_tuple = (applicant_name_from_cv.lower().strip(), applicant_email_from_cv.lower().strip())
        if name_email_combo_tuple in existing_data['names_emails']:
            is_duplicate = True
            cv_link_for_sheet = existing_data['email_to_cv_link'].get(applicant_email_from_cv.lower(), "")
            logging.info(f"Applicant name-email combo '{name_email_combo_tuple}' already exists (duplicate).")

    if not is_duplicate and applicant_phone_from_cv:
        if applicant_phone_from_cv.lower().strip() in existing_data['phones']:
            is_duplicate = True
            logging.info(f"Applicant phone '{applicant_phone_from_cv}' already exists (duplicate).")

    if is_duplicate:
        logging.info(f"Skipping processing for MsgID {msg_id} due to duplicate applicant detection.")
        return
    try:
        cv_link_for_sheet = file_uploader.upload_cv_to_drive(drive_service, cv_data, cv_filename)
        logging.info(f"CV uploaded to Google Drive: {cv_link_for_sheet}")
    except Exception as e:
        logging.error(f"Error uploading CV for MsgID {msg_id}: {e}")
        cv_link_for_sheet = "Upload Failed"


    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    acknowledgment_sent_status = "No" 
    personalized_body = config.AUTO_REPLY_BODY_PLAIN.format(applicant_name=applicant_name_from_cv.split(' ')[0] if applicant_name_from_cv else "Applicant")

   
    # UNCOMMENT THE LINE BELOW TO PREVENT EMAILS FROM BEING SENT IN PRODUCTION
    email_sent_successfully = False # Keeping this for explicit control, as per original script's comment
    # COMMENT OUT THE LINE BELOW TO PREVENT EMAILS FROM BEING SENT
    # email_sent_successfully = mail_sender.send_acknowledgment_email(gmail_service, sender_email_for_reply, config.AUTO_REPLY_SUBJECT, personalized_body) 


    if sender_email_for_reply: 
        try:
            if email_sent_successfully:
                acknowledgment_sent_status = "Yes"
                logging.info(f"Acknowledgment email sent to {sender_email_for_reply} for MsgID: {msg_id}")
            else: 
                acknowledgment_sent_status = "Skipped/Failed"
                logging.warning(f"Acknowledgment email to {sender_email_for_reply} for MsgID: {msg_id} was not sent (controlled by script).")
        except Exception as e:
            logging.error(f"Error sending acknowledgment email to {sender_email_for_reply} for MsgID {msg_id}: {e}")
            acknowledgment_sent_status = "Error"
    else:
        logging.warning(f"No sender email found for message ID {msg_id}. Skipping acknowledgment email.")
        acknowledgment_sent_status = "No Reply Address"


    phone_txt_for_sheet = applicant_phone_from_cv
    if phone_txt_for_sheet and not phone_txt_for_sheet.startswith("'"):
        phone_txt_for_sheet = f"'{phone_txt_for_sheet}"

    row_data = [
        str(msg_id),
        str(sender_full_string),
        str(subject_line),
        str(date_header), 
        str(cv_link_for_sheet),
        str(applicant_name_from_cv),
        str(applicant_email_from_cv), 
        str(phone_txt_for_sheet), 
        str(inferred_job_title),
        "New Application", 
        str(acknowledgment_sent_status), 
        str(current_timestamp) 
    ]

    try:
        sheet_writer.append_rows(sheets_service, [row_data], sheet_id, tab_name) # append_rows expects a list of lists
        logging.info(f"Data written to Google Sheet for MsgID: {msg_id}")
        
        existing_data['msg_ids'].add(msg_id)
        if applicant_name_from_cv and applicant_email_from_cv:
            existing_data['names_emails'].add((applicant_name_from_cv.lower().strip(), applicant_email_from_cv.lower().strip()))
            existing_data['email_to_cv_link'][applicant_email_from_cv.lower()] = cv_link_for_sheet
        if applicant_phone_from_cv:
            existing_data['phones'].add(applicant_phone_from_cv.lower().strip())

    except Exception as e:
        logging.error(f"Error writing data to Google Sheet for MsgID {msg_id}: {e}")


def run_once():
    """
    Executes the bot's logic once: fetches new emails, processes them, and updates sheets.
    Mirrors the monolithic script's 'run_once' function.
    """
    logging.info("Starting a single run of the recruitment bot.")

    try:
        gmail_creds = gmail_auth.authenticate_gmail()
        gmail_service = build('gmail', 'v1', credentials=gmail_creds)

        sheets_service = sheets_auth.authenticate_sheets_service()
        drive_service = drive_auth.authenticate_drive_service()
    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        return
    

    sheet_id = config.SHEETS_ID
    tab_name = config.SHEETS_TAB_NAME

    try:
        sheet_writer.ensure_tab_exists(sheets_service, sheet_id, tab_name)
        sheet_writer.ensure_header(sheets_service, sheet_id, tab_name, sheet_writer.SHEETS_DEFAULT_HEADER) 
        existing_data = sheet_reader.get_existing_applicants_data(sheets_service, sheet_id, tab_name)
        logging.info(f"Fetched existing entries from Google Sheet. MsgIDs: {len(existing_data['msg_ids'])}, Name-Emails: {len(existing_data['names_emails'])}, Phones: {len(existing_data['phones'])}.")
    except Exception as e:
        logging.error(f"Error initializing Google Sheet or fetching existing data: {e}")
        return

    query_for_emails = config.GMAIL_EMAIL_QUERY
    
    messages = mail_service.fetch_messages(gmail_service, query=query_for_emails, max_results=config.GMAIL_MAX_FETCH)
    if not messages:
        logging.info("No new emails found matching the query.")
        return

    logging.info(f"Found {len(messages)} emails matching the query.")

    for message in messages:
        try:
            process_email_and_attachment(gmail_service, sheets_service, drive_service, message, sheet_id, tab_name, existing_data)
        

        except Exception as e:
            logging.error(f"Failed to process message {message.get('id', 'N/A')}: {e}")

    logging.info("Single run of the recruitment bot finished.")


def main():
    """
    Main function to run the bot continuously or once based on command line arguments.
    Matches the monolithic script's main execution block.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Recruitment bot to process emails and update Google Sheets.")
    parser.add_argument("--once", action="store_true", help="Run the bot once and exit.")
    args = parser.parse_args()

    if args.once:
        try:
            run_once()
            sys.exit(0) 
        except HttpError as e:
            logging.error(f"Google API error during --once run: {e}")
            sys.exit(1) 
        except Exception as ex:
            logging.error(f"An unexpected error occurred during --once run: {ex}")
            sys.exit(1)
    else:
        logging.info(f"Recruitment bot running every {config.CHECK_INTERVAL_SECONDS} seconds (Ctrl-C to stop)...")
        while True:
            try:
                run_once()
            except HttpError as e:
                logging.error(f"Google API error: {e}")
            except Exception as ex:
                logging.error(f"Unexpected error: {ex}")
            time.sleep(config.CHECK_INTERVAL_SECONDS)

if __name__ == '__main__':
    main()