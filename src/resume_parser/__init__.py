import spacy
import sys


try:
    nlp = spacy.load("en_core_web_lg")
    print("INFO: spaCy model 'en_core_web_lg' loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load spaCy model 'en_core_web_lg'. Please ensure it's installed: {e}")
    print("Run `python -m spacy download en_core_web_lg` from your terminal.")
    sys.exit(1) 

