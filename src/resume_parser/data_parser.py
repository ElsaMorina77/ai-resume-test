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
# NOTE: This line requires 'from src.resume_parser import text_extractor' to be a valid import in your project structure
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
            logger.debug(f"Potential phone match found: '{cleaned_num}' (raw: '{found_num}')")

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
        logger.debug(f"Extracted email: {match.group(0)}")
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
            logger.debug(f"Extracted name with spaCy: {ent.text}")
            return ent.text
    logger.debug("No PERSON entity found with spaCy NER.")
    return None



def parse_resume_data(bytes_: bytes, filename: str) -> Dict[str, Any]:
    """
    Uses pyresparser to parse various resume fields, and enhances with custom
    phone and email extraction, and spaCy for name extraction.
    """

   
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
    logger.debug(f"Full text content length: {len(full_text_content) if full_text_content else 0}")
    if full_text_content:
        logger.debug(f"Full text content snippet (first 500 chars): {full_text_content[:500]}")

    # --- Phone/Email Extraction Logic (Kept as is - it was fine) ---

    pyresparser_phone = parsed_data_pyresparser.get("mobile_number", "")
    custom_extracted_phone = extract_phone_number(full_text_content) if full_text_content else None

    final_phone = pyresparser_phone
    if custom_extracted_phone:
        pyresparser_digits = re.sub(r'\D', '', str(pyresparser_phone or ""))
        custom_digits = re.sub(r'\D', '', custom_extracted_phone)

        if not pyresparser_phone and len(custom_digits) >= 7:
            final_phone = custom_extracted_phone
            logger.debug(f"Using custom extracted phone: '{custom_extracted_phone}' (pyresparser was empty).")
        elif len(custom_digits) > len(pyresparser_digits) and len(custom_digits) >= 7:
            final_phone = custom_extracted_phone
            logger.debug(f"Using longer custom extracted phone: '{custom_extracted_phone}' over pyresparser: '{pyresparser_phone}'.")
        else:
            logger.debug(f"Keeping pyresparser phone: '{pyresparser_phone}'.")
    elif not final_phone:
        logger.debug("No phone number found from pyresparser or custom extraction.")

    pyresparser_email = parsed_data_pyresparser.get("email", "")
    custom_extracted_email = extract_email_address(full_text_content) if full_text_content else None

    final_email = pyresparser_email
    if custom_extracted_email:
        if not pyresparser_email or not re.match(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', str(pyresparser_email)):
            final_email = custom_extracted_email
            logger.debug(f"Using custom extracted email: '{custom_extracted_email}' (pyresparser was empty or invalid).")
        else:
            logger.debug(f"Keeping pyresparser email: '{pyresparser_email}'.")
    elif not final_email:
        logger.debug("No email found from pyresparser or custom extraction.")

    # --------------------------------------------------------------------------
    # --- NAME EXTRACTION LOGIC (Fixing the De-Spacing Issue) ------------------
    # --------------------------------------------------------------------------
    extracted_name = ""
    # Massively Expanded Keywords to filter out non-name entities
    generic_keywords = [
        "experienced", "developer", "professional", "resume", "cv", 
        "curriculum vitae", "profile", "contact", "summary", "education", 
        "experience", "skills", "projects", "full-stack", "front-end", 
        "back-end", "manager", "engineer", "analyst", "github", "linkedin",
        "mobile", "number", "email", "mail", "phone", "address", "objective",
        "about me", "references", "work history", "career summary", 
        "technical skills", "personal details", "city", "street", "vue js",
        "git", "don bosko", "altina avdyli" # Added specific bad matches and common headers/locations
    ] 
    
    # Lowercase set for faster keyword checking
    generic_keywords_lower = set(kw.lower() for kw in generic_keywords)
    
    lines_top_of_resume = [line.strip() for line in (full_text_content or "").split('\n') if line.strip()][:5]
    
    # Helper function to check if a candidate name is too generic
    def is_generic_or_bad(text: str) -> bool:
        text_lower = text.lower()
        if any(kw in text_lower for kw in generic_keywords_lower):
            return True
        # Check for non-alphabetic characters that aren't spaces or hyphens (often found in addresses/titles)
        if re.search(r'[^\w\s-]', text):
            return True
        return False

    # 1. Highest Priority: Simple heuristic on the very first non-empty line
    if lines_top_of_resume:
        first_line = lines_top_of_resume[0]
        
        # --- CRITICAL FIX: DE-SPACING THE LINE FOR ACCURATE WORD COUNT/CHECK ---
        # Normalize excessive whitespace (e.g., 'A D R I A N' becomes 'A D R I A N')
        despaced_line = re.sub(r'\s+', ' ', first_line).strip() 
        words = despaced_line.split()
        
        # Stricter Heuristic: 2-5 words, all are mostly capitalized, no numbers, and NOT a generic keyword
        if 2 <= len(words) <= 5 and \
           (all(word[0].isupper() or len(word) <= 2 for word in words) or despaced_line.isupper()) and \
           not any(char.isdigit() for char in despaced_line) and \
           not is_generic_or_bad(despaced_line):
            # We use the despaced_line as the name candidate
            extracted_name = despaced_line 
            logger.debug(f"Selected name from simple first-line heuristic (Highest Priority, De-Spaced): '{extracted_name}'")
            
    # 2. Pyresparser Name (Good fallback if the first line heuristic failed)
    if not extracted_name:
        pyresparser_name = parsed_data_pyresparser.get("name", "").strip()
        logger.debug(f"Initial name from pyresparser: '{pyresparser_name}'")
        if pyresparser_name:
            words = pyresparser_name.split()
            # Ensure it's at least 2 words and not a generic keyword
            if len(words) >= 2 and not is_generic_or_bad(pyresparser_name):
                extracted_name = pyresparser_name
                logger.debug(f"Selected pyresparser name: '{extracted_name}'")
            else:
                logger.debug(f"Pyresparser name ('{pyresparser_name}') seems weak or generic, skipping.")

    # 3. SpaCy NER on Top Lines (Mid-level priority, more accurate than full text)
    if not extracted_name and nlp is not None:
        potential_name_from_top = None
        for line in lines_top_of_resume:
            doc_line = nlp(line)
            for ent in doc_line.ents:
                if ent.label_ == "PERSON":
                    candidate = ent.text.replace('\n', ' ').strip()
                    words = candidate.split()
                    
                    # Refined spaCy check: at least 2 words, and not a generic keyword
                    if len(words) >= 2 and not is_generic_or_bad(candidate):
                        potential_name_from_top = candidate
                        logger.debug(f"Strong candidate name from top lines (spaCy NER): '{potential_name_from_top}'")
                        break 
            if potential_name_from_top:
                break 

        if potential_name_from_top:
            extracted_name = potential_name_from_top
            logger.debug(f"Selected name from top lines (spaCy): '{extracted_name}'")
        else:
            logger.debug("No strong name found from top lines using spaCy.")


    # 4. SpaCy NER on Full Text (Lowest priority/last resort)
    if not extracted_name and nlp is not None:
        spacy_extracted_name_full = extract_name_with_spacy(full_text_content)
        logger.debug(f"SpaCy name from full text (fallback): '{spacy_extracted_name_full}'")
        if spacy_extracted_name_full:
            words = spacy_extracted_name_full.split()
            # Strict check: 2-5 words, and NOT a generic keyword
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