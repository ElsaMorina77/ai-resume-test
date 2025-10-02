# src/gmail_api/auth.py

import os
import sys
import pickle

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Determine the project root dynamically
# This script is in 'src/gmail_api'.
# project_root should be 'C:\Users\Lenovo\Desktop\New folder'
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

# Add project_root to sys.path to allow importing modules from the root directory
# This ensures 'config' can be found.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import config # This import should now work!

# If modifying these scopes, delete the file token.json.
SCOPES = config.GMAIL_SCOPES

def authenticate_gmail():
    """Authenticates with Google Gmail API using OAuth 2.0 flow.
    Manages token.json for persistent credentials."""
    creds = None
    
    token_path = os.path.join(project_root, 'token.json')
    client_secret_path = os.path.join(project_root, 'config', 'client_secret.json') 

    if os.path.exists(token_path):
        try:
            with open(token_path, 'rb') as token_file:
                creds = pickle.load(token_file)
        except Exception as e:
            print(f"Error loading token.json: {e}. Deleting token.json and re-authenticating.")
            try:
                os.remove(token_path)
            except OSError as del_e:
                print(f"Error deleting corrupted token.json: {del_e}")
            creds = None 

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing existing Gmail token...")
            creds.refresh(Request())
        else:
            if not os.path.exists(client_secret_path):
                print(f"Error: 'client_secret.json' not found at expected path: {client_secret_path}")
                print("Please download your OAuth 2.0 Client ID JSON file from Google Cloud Console and place it in your project's 'config' folder.")
                sys.exit(1) 

            print(f"Starting new Gmail authentication flow. Please visit the URL in your browser.")
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open(token_path, 'wb') as token_file:
            pickle.dump(creds, token_file)
        print(f"New/refreshed Gmail credentials saved to {token_path}")
    
    return creds

if __name__ == '__main__':
    print("Running Gmail authentication script directly (src/gmail_api/auth.py)...")
    try:
        credentials = authenticate_gmail() 
        if credentials:
            print("Direct Gmail authentication successful!")
        else:
            print("Direct Gmail authentication failed.")
    except Exception as e:
        print(f"An error occurred during direct Gmail authentication: {e}")