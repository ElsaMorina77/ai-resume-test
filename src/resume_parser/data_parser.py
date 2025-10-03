import logging
import re
import spacy
from pyresparser import ResumeParser
from pathlib import Path
from typing import Optional, List, Dict, Any 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Attempt to load the larger, more accurate model first
print(f"DEBUG: Attempting to load spaCy model 'en_core_web_lg' from {__file__}")

nlp = None
try:
    # Try loading the larger model for better NER accuracy
    nlp = spacy.load('en_core_web_lg') 
    logger.info("spaCy model 'en_core_web_lg' loaded successfully.")
except Exception as e:
    logger.warning(f"Could not load spaCy model 'en_core_web_lg': {e}. Falling back to 'en_core_web_sm'.")
    try:
        # Fallback to the smaller model
        nlp = spacy.load('en_core_web_sm') 
        logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
    except Exception as e_sm:
        logger.error(f"Could not load spaCy model 'en_core_web_sm'. Please ensure it's installed: {e_sm}")
        nlp = None 

# Assuming this module exists in your project structure
from src.resume_parser import text_extractor 


def extract_phone_number(text: str) -> Optional[str]: 
    """
    Extracts a phone number from the given text using a comprehensive regex pattern.
    Returns the most plausible phone number found or None.
    """

    phone_regex = re.compile(
        r'''
        (?:(?:\+|00)\d{1,3}[-.\s]?)? 
        (?:                          
            \(?0?\d{2,5}\)?          
            [-.\s]?                  
        )?
        \d{2,4}[-.\s]?               
        \d{2,4}[-.\s]?               
        \d{2,4}                      
        |                           
        \b(?<!\d)\d{7,15}\b          
        ''',
        re.VERBOSE | re.IGNORECASE
    )

    potential_numbers = []
    for match in phone_regex.finditer(text):
        found_num = match.group(0) 
        # Cleaned_num keeps '+' and '()' for common formats
        cleaned_num = re.sub(r'[^\d+()-.\s]', '', found_num).strip()
        digits_only = re.sub(r'\D', '', cleaned_num) 

        if 7 <= len(digits_only) <= 15:
            potential_numbers.append(cleaned_num)
            # Removed redundant logging here to keep log clean during main loop scan

    if potential_numbers:
        # Select the number with the longest sequence of digits
        best_phone = max(potential_numbers, key=lambda x: len(re.sub(r'\D', '', x)))
        logger.debug(f"Selected phone number: {best_phone}")
        return best_phone

    logger.debug("No phone number found using robust regex.")
    return None


def extract_email_address(text: str) -> Optional[str]:
    """
    Extracts an email address from the given text using a standard regex pattern.
    Returns the first match found or None.
    """
    email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    match = email_regex.search(text)
    if match:
        # Removed redundant logging here to keep log clean during main loop scan
        return match.group(0)
    logger.debug("No email address found with regex.")
    return None

def extract_name_with_spacy(text: str) -> Optional[str]:
    """
    Extracts a person's name from the text using spaCy's Named Entity Recognition (NER).
    Returns the first PERSON entity found or None.
    """
    if nlp is None:
        logger.warning("spaCy model not loaded, cannot extract name with spaCy.")
        return None

    # Process only the first few KB of text for faster and more relevant NER
    doc = nlp(text[:2048]) 
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Removed redundant logging here to keep log clean
            return ent.text
    logger.debug("No PERSON entity found with spaCy NER.")
    return None











def parse_resume_data(bytes_: bytes, filename: str) -> Dict[str, Any]:
    """
    Uses pyresparser to parse various resume fields, and enhances with custom
    phone and email extraction, and spaCy for name extraction.
    """

    # --- 1. Setup and Pyresparser Parsing (Unchanged) ---
    temp_pyresparser_file = Path("_tmp_pyresparser_" + Path(filename).name)
    temp_pyresparser_file.write_bytes(bytes_)

    parsed_data_pyresparser = {}
    try:
        # ... [Pyresparser parsing logic, including custom_nlp and fallbacks] ...
        if nlp is not None:
            try:
                parser = ResumeParser(str(temp_pyresparser_file), custom_nlp=nlp)
                parsed_data_pyresparser = parser.get_extracted_data() or {}
                logger.debug(f"pyresparser (with custom_nlp) raw parsed_data: {parsed_data_pyresparser}")
            except TypeError as e_type:
                if "custom_nlp" in str(e_type):
                    logger.warning(f"Pyresparser version does not support 'custom_nlp' argument. Trying without it for '{filename}'.")
                    try:
                        parser = ResumeParser(str(temp_pyresparser_file))
                        parsed_data_pyresparser = parser.get_extracted_data() or {}
                        logger.debug(f"pyresparser (without custom_nlp) raw parsed_data: {parsed_data_pyresparser}")
                    except Exception as e_no_nlp_fallback:
                        logger.error(f"Error parsing resume with pyresparser (fallback, no custom_nlp) for '{filename}': {e_no_nlp_fallback}")
                        logger.error("Suggestion: Try 'pip uninstall pyresparser' then 'pip install pyresparser'.")
                        parsed_data_pyresparser = {}
                else:
                    logger.error(f"Unexpected TypeError during pyresparser parsing for '{filename}': {e_type}")
                    parsed_data_pyresparser = {}
        else:
            logger.warning("spaCy model not loaded (nlp is None), trying pyresparser without custom_nlp.")
            try:
                parser = ResumeParser(str(temp_pyresparser_file))
                parsed_data_pyresparser = parser.get_extracted_data() or {}
                logger.debug(f"pyresparser (nlp None, no custom_nlp) raw parsed_data: {parsed_data_pyresparser}")
            except Exception as e_no_nlp:
                logger.error(f"Error parsing resume with pyresparser (nlp None, no custom_nlp) for '{filename}': {e_no_nlp}")
                logger.error("Suggestion: Try 'pip uninstall pyresparser' then 'pip install pyresparser'.")
                parsed_data_pyresparser = {}

    except Exception as e:
        logger.error(f"General error parsing resume with pyresparser for '{filename}': {e}")
        parsed_data_pyresparser = {}
    finally:
        temp_pyresparser_file.unlink(missing_ok=True) 

    full_text_content = text_extractor.extract_text_from_resume(bytes_, filename)
    if not full_text_content:
        logger.warning(f"Full text content is empty for {filename}. Custom extraction will be limited.")

    # --- 2. Contact Extraction (Required for Name Heuristic) ---
    pyresparser_phone = parsed_data_pyresparser.get("mobile_number", "")
    custom_extracted_phone = extract_phone_number(full_text_content) if full_text_content else None
    final_phone = pyresparser_phone
    if custom_extracted_phone:
        pyresparser_digits = re.sub(r'\D', '', str(pyresparser_phone or ""))
        custom_digits = re.sub(r'\D', '', custom_extracted_phone)
        if not pyresparser_phone and len(custom_digits) >= 7:
            final_phone = custom_extracted_phone
        elif len(custom_digits) > len(pyresparser_digits) and len(custom_digits) >= 7:
            final_phone = custom_extracted_phone

    pyresparser_email = parsed_data_pyresparser.get("email", "")
    custom_extracted_email = extract_email_address(full_text_content) if full_text_content else None
    final_email = pyresparser_email
    if custom_extracted_email:
        if not pyresparser_email or not re.match(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', str(pyresparser_email)):
            final_email = custom_extracted_email

    # --------------------------------------------------------------------------
    # --- 3. NAME EXTRACTION LOGIC (Highest Priority on Email Proximity) -------
    # --------------------------------------------------------------------------
    extracted_name = ""
    generic_keywords = [
        "experienced", "developer", "professional", "resume", "cv", "curriculum vitae", "profile", 
        "contact", "summary", "education", "experience", "skills", "projects", "full-stack", 
        "front-end", "back-end", "manager", "engineer", "analyst", "github", "linkedin",
        "mobile", "number", "email", "mail", "phone", "address", "objective",
        "about me", "references", "work history", "career summary", "technical skills", 
        "personal details", "city", "street", "vue js", "git", "don bosko", "altina avdyli",
        "portfolio", "designer", "freelancer", "tech nexus", "axians", "kutia", "ubt"
    ] 
    
    generic_keywords_lower = set(kw.lower() for kw in generic_keywords)
    all_lines = [line.strip() for line in (full_text_content or "").split('\n') if line.strip()]
    lines_top_of_resume = all_lines[:5]
    
    # Helper function to check if a candidate name is too generic
    def is_generic_or_bad(text: str) -> bool:
        text_lower = text.lower()
        if any(kw in text_lower for kw in generic_keywords_lower):
            return True
        # Check for non-alphabetic characters that aren't spaces, hyphens, or common delimiters
        if re.search(r'[^\w\s-]', text) and not re.search(r'^\w+@\w+', text): # Allow emails but flag other symbols
            return True
        return False
    
    # --- Highest Priority: Name next to Email ---
    if final_email:
        for i, line in enumerate(all_lines):
            # Check if the line contains the final, confirmed email
            if final_email in line:
                # The name is often on the line *immediately preceding* the email or on the email line itself
                
                # Check line before (Most common location for floating name)
                if i > 0:
                    candidate_name = all_lines[i-1]
                    words = candidate_name.split()

                    if 2 <= len(words) <= 5 and \
                       (all(word[0].isupper() or len(word) <= 2 for word in words) or candidate_name.isupper()) and \
                       not is_generic_or_bad(candidate_name):
                        extracted_name = candidate_name
                        logger.debug(f"Selected name from Email Proximity Heuristic (Prev Line): '{extracted_name}'")
                        break
                        
                # Check current line (less common, usually separated by a pipe '|' or spaces)
                if not extracted_name and nlp is not None:
                    # Run spaCy only on the immediate line containing the email
                    doc_line = nlp(line)
                    for ent in doc_line.ents:
                        if ent.label_ == "PERSON":
                            candidate = ent.text.replace('\n', ' ').strip()
                            if not is_generic_or_bad(candidate):
                                extracted_name = candidate
                                logger.debug(f"Selected name from Email Proximity Heuristic (SpaCy on Email Line): '{extracted_name}'")
                                break
                    if extracted_name:
                        break


    # --- 1st Fallback: Simple heuristic on the very first non-empty line (De-Spaced) ---
    if not extracted_name and lines_top_of_resume:
        first_line = lines_top_of_resume[0]
        despaced_line = re.sub(r'\s+', ' ', first_line).strip() 
        words = despaced_line.split()
        
        if 2 <= len(words) <= 5 and \
           (all(word[0].isupper() or len(word) <= 2 for word in words) or despaced_line.isupper()) and \
           not any(char.isdigit() for char in despaced_line) and \
           not is_generic_or_bad(despaced_line):
            extracted_name = despaced_line 
            logger.debug(f"Selected name from simple first-line heuristic (1st Fallback): '{extracted_name}'")
            
    # --- 2nd Fallback: Pyresparser Name (Strictly Filtered) ---
    if not extracted_name:
        pyresparser_name = parsed_data_pyresparser.get("name", "").strip()
        logger.debug(f"Initial name from pyresparser: '{pyresparser_name}'")
        if pyresparser_name:
            words = pyresparser_name.split()
            # New Filter: If the pyresparser name is a weak/generic term OR is a substring of the email (common error)
            if len(words) >= 2 and not is_generic_or_bad(pyresparser_name) and pyresparser_name.lower() not in final_email.lower():
                extracted_name = pyresparser_name
                logger.debug(f"Selected pyresparser name (2nd Fallback): '{extracted_name}'")
            else:
                logger.debug(f"Pyresparser name ('{pyresparser_name}') seems weak, generic, or too similar to email/title, skipping.")

    # --- 3rd Fallback: SpaCy NER on Top Lines ---
    if not extracted_name and nlp is not None:
        potential_name_from_top = None
        for line in lines_top_of_resume:
            doc_line = nlp(line)
            for ent in doc_line.ents:
                if ent.label_ == "PERSON":
                    candidate = ent.text.replace('\n', ' ').strip()
                    words = candidate.split()
                    
                    if len(words) >= 2 and not is_generic_or_bad(candidate):
                        potential_name_from_top = candidate
                        break 
            if potential_name_from_top:
                extracted_name = potential_name_from_top
                logger.debug(f"Selected name from top lines (3rd Fallback, spaCy NER): '{extracted_name}'")
                break 


    # --- 4th Fallback: SpaCy NER on Full Text (Last Resort) ---
    if not extracted_name and nlp is not None:
        spacy_extracted_name_full = extract_name_with_spacy(full_text_content)
        if spacy_extracted_name_full:
            words = spacy_extracted_name_full.split()
            if 2 <= len(words) <= 5 and not is_generic_or_bad(spacy_extracted_name_full):
                extracted_name = spacy_extracted_name_full.replace('\n', ' ').strip()
                logger.debug(f"Selected spaCy name from full text (Last Resort): '{extracted_name}'")
            else:
                logger.debug(f"SpaCy full text name ('{spacy_extracted_name_full}') seems weak/generic, skipping.")


    # --- 4. Final Cleanup and Return ---
    final_name = str(extracted_name).replace('\n', ' ').strip() if extracted_name else ""
    final_email_cv = str(final_email).replace('\n', ' ').strip() if final_email else ""
    final_phone_cleaned = str(final_phone).replace('\n', ' ').strip() if final_phone else ""

    return {
        "name"              : final_name,
        "email_cv"          : final_email_cv,
        "phone"             : final_phone_cleaned,
        "skills"            : parsed_data_pyresparser.get("skills", []),
        "education"         : parsed_data_pyresparser.get("education", []),
        "experience"        : parsed_data_pyresparser.get("experience", []),
        "total_experience"  : parsed_data_pyresparser.get("total_experience", 0.0),
        "full_text_content" : full_text_content or "",
    }

"""def parse_resume_data(bytes_: bytes, filename: str) -> Dict[str, Any]:
    
    #Uses pyresparser to parse various resume fields, and enhances with custom
    #phone and email extraction, and spaCy for name extraction.
    

   
    temp_pyresparser_file = Path("_tmp_pyresparser_" + Path(filename).name)
    temp_pyresparser_file.write_bytes(bytes_)

    parsed_data_pyresparser = {}
    try:
        if nlp is not None:
            try:
                parser = ResumeParser(str(temp_pyresparser_file), custom_nlp=nlp)
                parsed_data_pyresparser = parser.get_extracted_data() or {}
                logger.debug(f"pyresparser (with custom_nlp) raw parsed_data: {parsed_data_pyresparser}")
            except TypeError as e_type:
                if "custom_nlp" in str(e_type):
                    logger.warning(f"Pyresparser version does not support 'custom_nlp' argument. Trying without it for '{filename}'.")
                    try:
                        parser = ResumeParser(str(temp_pyresparser_file))
                        parsed_data_pyresparser = parser.get_extracted_data() or {}
                        logger.debug(f"pyresparser (without custom_nlp) raw parsed_data: {parsed_data_pyresparser}")
                    except Exception as e_no_nlp_fallback:
                        logger.error(f"Error parsing resume with pyresparser (fallback, no custom_nlp) for '{filename}': {e_no_nlp_fallback}")
                        logger.error("Suggestion: Try 'pip uninstall pyresparser' then 'pip install pyresparser'.")
                        parsed_data_pyresparser = {}
                else:
                    logger.error(f"Unexpected TypeError during pyresparser parsing for '{filename}': {e_type}")
                    parsed_data_pyresparser = {}
        else:
            logger.warning("spaCy model not loaded (nlp is None), trying pyresparser without custom_nlp.")
            try:
                parser = ResumeParser(str(temp_pyresparser_file))
                parsed_data_pyresparser = parser.get_extracted_data() or {}
                logger.debug(f"pyresparser (nlp None, no custom_nlp) raw parsed_data: {parsed_data_pyresparser}")
            except Exception as e_no_nlp:
                logger.error(f"Error parsing resume with pyresparser (nlp None, no custom_nlp) for '{filename}': {e_no_nlp}")
                logger.error("Suggestion: Try 'pip uninstall pyresparser' then 'pip install pyresparser'.")
                parsed_data_pyresparser = {}

    except Exception as e:
        logger.error(f"General error parsing resume with pyresparser for '{filename}': {e}")
        parsed_data_pyresparser = {}
    finally:
        temp_pyresparser_file.unlink(missing_ok=True) 

    full_text_content = text_extractor.extract_text_from_resume(bytes_, filename)
    if not full_text_content:
        logger.warning(f"Full text content is empty for {filename}. Custom extraction will be limited.")
    # Removed snippet logging to keep output cleaner

    # --- Phone/Email Extraction Logic (Combined for efficiency) ---
    pyresparser_phone = parsed_data_pyresparser.get("mobile_number", "")
    custom_extracted_phone = extract_phone_number(full_text_content) if full_text_content else None
    final_phone = pyresparser_phone
    if custom_extracted_phone:
        pyresparser_digits = re.sub(r'\D', '', str(pyresparser_phone or ""))
        custom_digits = re.sub(r'\D', '', custom_extracted_phone)
        if not pyresparser_phone and len(custom_digits) >= 7:
            final_phone = custom_extracted_phone
        elif len(custom_digits) > len(pyresparser_digits) and len(custom_digits) >= 7:
            final_phone = custom_extracted_phone

    pyresparser_email = parsed_data_pyresparser.get("email", "")
    custom_extracted_email = extract_email_address(full_text_content) if full_text_content else None
    final_email = pyresparser_email
    if custom_extracted_email:
        if not pyresparser_email or not re.match(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', str(pyresparser_email)):
            final_email = custom_extracted_email

    # --------------------------------------------------------------------------
    # --- NAME EXTRACTION LOGIC (Fixing the Out-of-Place Name Issue) -----------
    # --------------------------------------------------------------------------
    extracted_name = ""
    generic_keywords = [
        "experienced", "developer", "professional", "resume", "cv", "curriculum vitae", "profile", 
        "contact", "summary", "education", "experience", "skills", "projects", "full-stack", 
        "front-end", "back-end", "manager", "engineer", "analyst", "github", "linkedin",
        "mobile", "number", "email", "mail", "phone", "address", "objective",
        "about me", "references", "work history", "career summary", "technical skills", 
        "personal details", "city", "street", "vue js", "git", "don bosko", "altina avdyli",
        "portfolio", "designer", "freelancer", "tech nexus", "axians", "kutia", "ubt" # Added more specific role/company keywords
    ] 
    
    generic_keywords_lower = set(kw.lower() for kw in generic_keywords)
    all_lines = [line.strip() for line in (full_text_content or "").split('\n') if line.strip()]
    lines_top_of_resume = all_lines[:5]
    
    # Helper function to check if a candidate name is too generic
    def is_generic_or_bad(text: str) -> bool:
        text_lower = text.lower()
        if any(kw in text_lower for kw in generic_keywords_lower):
            return True
        # Check for non-alphabetic characters that aren't spaces, hyphens, or common delimiters
        if re.search(r'[^\w\s-]', text) and not re.search(r'^\w+@\w+', text): # Allow emails but flag other symbols
            return True
        return False

    # 1. Highest Priority A: Simple heuristic on the very first non-empty line
    if lines_top_of_resume:
        first_line = lines_top_of_resume[0]
        despaced_line = re.sub(r'\s+', ' ', first_line).strip() 
        words = despaced_line.split()
        
        if 2 <= len(words) <= 5 and \
           (all(word[0].isupper() or len(word) <= 2 for word in words) or despaced_line.isupper()) and \
           not any(char.isdigit() for char in despaced_line) and \
           not is_generic_or_bad(despaced_line):
            extracted_name = despaced_line 
            logger.debug(f"Selected name from simple first-line heuristic (Highest Priority A, De-Spaced): '{extracted_name}'")
            
    # 1. Highest Priority B (New): Contact Block Heuristic (for names at the bottom/middle)
    # Scans all lines for a contact block pattern (Name + Title/Location/Contact on adjacent lines)
    if not extracted_name:
        for i in range(1, len(all_lines)):
            current_line = all_lines[i]
            prev_line = all_lines[i-1]
            
            # Check if the current line contains either a phone or email (the marker for contact info)
            if extract_email_address(current_line) or extract_phone_number(current_line):
                candidate_name = prev_line
                words = candidate_name.split()

                # Check the line *before* the contact info:
                # 1. Must have 2-5 words.
                # 2. Must be capitalized (or all caps).
                # 3. Must NOT be a generic keyword.
                if 2 <= len(words) <= 5 and \
                   (all(word[0].isupper() or len(word) <= 2 for word in words) or candidate_name.isupper()) and \
                   not is_generic_or_bad(candidate_name):
                    extracted_name = candidate_name
                    logger.debug(f"Selected name from Contact Block Heuristic (Highest Priority B): '{extracted_name}' from line: '{prev_line}'")
                    # Also try to grab the name if it's on the line *containing* the contact info
                    if not extracted_name and nlp is not None:
                        doc_line = nlp(current_line)
                        for ent in doc_line.ents:
                            if ent.label_ == "PERSON":
                                extracted_name = ent.text.replace('\n', ' ').strip()
                                logger.debug(f"Selected name from Contact Block Heuristic (SpaCy on Contact Line): '{extracted_name}'")
                                break
                    break 

    # 2. Pyresparser Name (Mid-High Priority)
    if not extracted_name:
        pyresparser_name = parsed_data_pyresparser.get("name", "").strip()
        logger.debug(f"Initial name from pyresparser: '{pyresparser_name}'")
        if pyresparser_name:
            words = pyresparser_name.split()
            if len(words) >= 2 and not is_generic_or_bad(pyresparser_name):
                extracted_name = pyresparser_name
                logger.debug(f"Selected pyresparser name: '{extracted_name}'")
            else:
                logger.debug(f"Pyresparser name ('{pyresparser_name}') seems weak or generic, skipping.")

    # 3. SpaCy NER on Top Lines (Mid-Low Priority)
    if not extracted_name and nlp is not None:
        potential_name_from_top = None
        for line in lines_top_of_resume:
            doc_line = nlp(line)
            for ent in doc_line.ents:
                if ent.label_ == "PERSON":
                    candidate = ent.text.replace('\n', ' ').strip()
                    words = candidate.split()
                    
                    if len(words) >= 2 and not is_generic_or_bad(candidate):
                        potential_name_from_top = candidate
                        break 
            if potential_name_from_top:
                extracted_name = potential_name_from_top
                logger.debug(f"Selected name from top lines (spaCy NER): '{extracted_name}'")
                break 


    # 4. SpaCy NER on Full Text (Lowest priority/last resort)
    if not extracted_name and nlp is not None:
        spacy_extracted_name_full = extract_name_with_spacy(full_text_content)
        if spacy_extracted_name_full:
            words = spacy_extracted_name_full.split()
            if 2 <= len(words) <= 5 and not is_generic_or_bad(spacy_extracted_name_full):
                extracted_name = spacy_extracted_name_full.replace('\n', ' ').strip()
                logger.debug(f"Selected spaCy name from full text (Last Resort): '{extracted_name}'")
            else:
                logger.debug(f"SpaCy full text name ('{spacy_extracted_name_full}') seems weak/generic, skipping.")


    # --- Final Cleanup ---

    final_name = str(extracted_name).replace('\n', ' ').strip() if extracted_name else ""
    final_email_cv = str(final_email).replace('\n', ' ').strip() if final_email else ""
    final_phone_cleaned = str(final_phone).replace('\n', ' ').strip() if final_phone else ""

    return {
        "name"              : final_name,
        "email_cv"          : final_email_cv,
        "phone"             : final_phone_cleaned,
        "skills"            : parsed_data_pyresparser.get("skills", []),
        "education"         : parsed_data_pyresparser.get("education", []),
        "experience"        : parsed_data_pyresparser.get("experience", []),
        "total_experience"  : parsed_data_pyresparser.get("total_experience", 0.0),
        "full_text_content" : full_text_content or "",
    }

    """


    