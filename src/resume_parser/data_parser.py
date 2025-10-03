import logging
import re
import spacy
from pyresparser import ResumeParser
from pathlib import Path
from typing import Optional, List, Dict, Any 
from nameparser import HumanName 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 

# --- 1. SETUP: SPACY MODEL LOADING ---
print(f"DEBUG: Attempting to load spaCy model 'en_core_web_lg' from {__file__}")

nlp = None
try:
    nlp = spacy.load('en_core_web_lg') 
    logger.info("spaCy model 'en_core_web_lg' loaded successfully.")
except Exception as e:
    logger.warning(f"Could not load spaCy model 'en_core_web_lg': {e}. Falling back to 'en_core_web_sm'.")
    try:
        nlp = spacy.load('en_core_web_sm') 
        logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
    except Exception as e_sm:
        logger.error(f"Could not load spaCy model 'en_core_web_sm'. Please ensure it's installed: {e_sm}")
        nlp = None 

# Assuming this module exists in your project structure
# Replace this with a placeholder to avoid breaking the code if the module is missing
class MockTextExtractor:
    @staticmethod
    def extract_text_from_resume(bytes_, filename):
        return ""
try:
    from src.resume_parser import text_extractor 
except ImportError:
    text_extractor = MockTextExtractor()
    logger.warning("Using MockTextExtractor as src.resume_parser.text_extractor could not be imported.")

# --- 2. HELPER FUNCTIONS ---

def extract_phone_number(text: str) -> Optional[str]: 
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
        cleaned_num = re.sub(r'[^\d+()-.\s]', '', found_num).strip()
        digits_only = re.sub(r'\D', '', cleaned_num) 
        if 7 <= len(digits_only) <= 15:
            potential_numbers.append(cleaned_num)
    if potential_numbers:
        best_phone = max(potential_numbers, key=lambda x: len(re.sub(r'\D', '', x)))
        logger.debug(f"Selected phone number: {best_phone}")
        return best_phone
    logger.debug("No phone number found using robust regex.")
    return None


def extract_email_address(text: str) -> Optional[str]:
    email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    match = email_regex.search(text)
    if match:
        logger.debug(f"Extracted email: {match.group(0)}")
        return match.group(0)
    logger.debug("No email address found with regex.")
    return None

def extract_name_with_spacy(text: str) -> Optional[str]:
    global nlp 

    if nlp is None:
        logger.warning("spaCy model not loaded, cannot extract name with spaCy.")
        return None

    doc = nlp(text[:2048]) 
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            logger.debug(f"Extracted name with spaCy from full text: {ent.text}")
            return ent.text
    logger.debug("No PERSON entity found with spaCy NER from full text.")
    return None

def is_valid_name_format(name_candidate: str) -> bool:
    """Uses nameparser to check if a string has a recognizable first and last name structure."""
    if not name_candidate:
        return False
    
    # Allow up to 6 words for complex Albanian names
    if len(name_candidate.split()) > 6: 
        return False
    
    if re.search(r'[^\w\s-]', name_candidate) and not re.search(r'^\w+@\w+', name_candidate):
        return False
        
    name_parts = HumanName(name_candidate)
    
    # Require at least a first name and a last name, each being more than 1 char
    if name_parts.first and name_parts.last and len(name_parts.first) > 1 and len(name_parts.last) > 1:
        logger.debug(f"Name candidate '{name_candidate}' validated by nameparser: {name_parts.first} {name_parts.last}")
        return True
    
    logger.debug(f"Name candidate '{name_candidate}' failed nameparser validation.")
    return False


# --- 3. MAIN PARSING FUNCTION ---

def parse_resume_data(bytes_: bytes, filename: str) -> Dict[str, Any]:
    # --- 3.1 PYRESPARSER EXECUTION ---
    temp_pyresparser_file = Path("_tmp_pyresparser_" + Path(filename).name)
    temp_pyresparser_file.write_bytes(bytes_)
    parsed_data_pyresparser = {}
    
    try:
        if nlp is not None:
            try:
                parser = ResumeParser(str(temp_pyresparser_file), custom_nlp=nlp)
                parsed_data_pyresparser = parser.get_extracted_data() or {}
            except Exception:
                try:
                    parser = ResumeParser(str(temp_pyresparser_file))
                    parsed_data_pyresparser = parser.get_extracted_data() or {}
                except:
                    parsed_data_pyresparser = {}
        else:
            try:
                parser = ResumeParser(str(temp_pyresparser_file))
                parsed_data_pyresparser = parser.get_extracted_data() or {}
            except Exception:
                parsed_data_pyresparser = {}
    except Exception as e:
        logger.error(f"General error parsing resume with pyresparser for '{filename}': {e}")
        parsed_data_pyresparser = {}
    finally:
        temp_pyresparser_file.unlink(missing_ok=True) 

    # Extract full text content
    full_text_content = text_extractor.extract_text_from_resume(bytes_, filename)
    if not full_text_content:
        logger.warning(f"Full text content is empty for {filename}. Custom extraction will be limited.")

    # --- 3.2 CONTACT EXTRACTION ---
    pyresparser_phone = parsed_data_pyresparser.get("mobile_number", "")
    custom_extracted_phone = extract_phone_number(full_text_content) if full_text_content else None
    final_phone = custom_extracted_phone if custom_extracted_phone else pyresparser_phone 

    pyresparser_email = parsed_data_pyresparser.get("email", "")
    custom_extracted_email = extract_email_address(full_text_content) if full_text_content else None
    final_email = custom_extracted_email if custom_extracted_email else pyresparser_email

    # --- 3.3 ENHANCED NAME EXTRACTION LOGIC ---
    extracted_name = ""
    
    # Generic Keywords
    generic_keywords = [
        "experienced", "developer", "professional", "resume", "cv", 
        "curriculum vitae", "profile", "contact", "summary", "education", 
        "experience", "skills", "projects", "full-stack", "front-end", 
        "back-end", "manager", "engineer", "analyst", "github", "linkedin",
        "mobile", "number", "email", "mail", "phone", "address", "objective",
        "about me", "references", "work history", "career summary", 
        "technical skills", "personal details", "city", "street", "vue js",
        "git", "don bosko", "portfolio", "designer", 
        "freelancer", "tech nexus", "axians", "kutia", "ubt", "product",
        "bachelor", "master", "doctor", "phd", "pristina", "kosovo",
        "personal info", "personal details", "contact info",
        "more information",
    ] 
    generic_keywords_lower = set(kw.lower() for kw in generic_keywords)
    
    all_lines = [line.strip() for line in (full_text_content or "").split('\n') if line.strip()]
    lines_top_of_resume = all_lines[:10] 
    
    def is_generic_or_bad(text: str) -> bool:
        """Checks if a candidate line is a job title, common header, or otherwise invalid."""
        text_lower = text.lower()
        if any(kw in text_lower for kw in generic_keywords_lower):
            return True
        # Reject if it contains numbers/symbols other than spaces/hyphens
        if re.search(r'[^\w\s-]', text) and not re.search(r'^\w+@\w+', text):
            return True
        return False
        
    def validate_candidate(candidate: str, source: str) -> Optional[str]:
        """Validates a candidate against heuristics and nameparser."""
        candidate = candidate.replace('\n', ' ').strip()
        
        # 1. Heuristic Filter (length, bad keywords)
        words = candidate.split()
        # Check: 2-6 words
        if not (2 <= len(words) <= 6):
            logger.debug(f"Candidate '{candidate}' from {source} failed word count (2-6).")
            return None
        if is_generic_or_bad(candidate):
            logger.debug(f"Candidate '{candidate}' from {source} failed generic keyword check.")
            return None
            
        # 2. Nameparser Validation
        if not is_valid_name_format(candidate):
            logger.debug(f"Candidate '{candidate}' from {source} failed nameparser validation.")
            return None
            
        logger.debug(f"VALIDATED name candidate from {source}: '{candidate}'")
        return candidate
        
    def aggressively_despace_name(text: str) -> Optional[str]:
        """
        Handles names with excessive single-space spacing (A D R I A N).
        """
        words = text.split()
        
        # Check if the line matches the single-letter-spacing pattern 
        if len(words) >= 4 and all(len(word) <= 2 and word.isupper() for word in words):
            merged_name = "".join(words)
            
            # Try to insert a space at common split points (4 to 9 characters)
            for split in range(4, min(10, len(merged_name) - 3)):
                candidate_with_space = merged_name[:split] + " " + merged_name[split:]
                
                # Check if adding a space makes it a valid name 
                if is_valid_name_format(candidate_with_space):
                    logger.debug(f"Aggressively de-spaced, inserted space, and validated name: '{candidate_with_space}'")
                    return candidate_with_space.title() # Return in Title case
            
            logger.debug(f"Aggressively de-spaced name '{merged_name}' could not be parsed even with split attempts.")
                
        return None


    # --- NAME EXTRACTION PRIORITY CHAIN ---

    # 1. Highest Priority: Email Proximity Heuristic (For floating names)
    if final_email:
        for i, line in enumerate(all_lines):
            if final_email in line:
                if i > 0:
                    candidate = all_lines[i-1]
                    extracted_name = validate_candidate(candidate, "Email Proximity - Prev Line")
                    if extracted_name:
                        break
                if not extracted_name and nlp is not None:
                    doc_line = nlp(line)
                    for ent in doc_line.ents:
                        if ent.label_ == "PERSON":
                            candidate = ent.text
                            extracted_name = validate_candidate(candidate, "Email Proximity - SpaCy on Email Line")
                            if extracted_name:
                                break
                    if extracted_name:
                        break
        
    # 1a. NEW: Labeled Name Heuristic (For "Full Name: John Doe" type formats)
    if not extracted_name:
        # Regex looks for "name", "full name", "applicant", etc., followed by optional colon/spaces and then the name content
        name_label_regex = re.compile(r'(\b(?:full\s*name|name|applicant|candidate|personal\s*details)\s*[:]?\s*)(.+)', re.IGNORECASE)
        
        for line in lines_top_of_resume:
            match = name_label_regex.search(line)
            if match:
                candidate = match.group(2).strip()
                # Check if the extracted part is just more contact info
                if any(kw in candidate.lower() for kw in ["email", "phone", "contact", "@", "address", "linkedin"]):
                     continue

                extracted_name = validate_candidate(candidate, "Labeled Name Heuristic")
                if extracted_name:
                    logger.debug(f"Selected name from Labeled Name Heuristic: '{extracted_name}'")
                    break

    # 2. 1st Fallback: Top Lines Heuristic (Captures names placed correctly)
    if not extracted_name:
        for line in lines_top_of_resume:
            despaced_line = re.sub(r'\s+', ' ', line).strip()
            
            # Aggressive De-Spacing Check (Handles A D R I A N S H A L A)
            aggressively_despaced_name = aggressively_despace_name(line)
            if aggressively_despaced_name:
                extracted_name = aggressively_despaced_name
                logger.debug(f"Selected name from Aggressive De-Spacing: '{extracted_name}'")
                break
            
            # Original Simple Capitalization Check 
            if all(word[0].isupper() or len(word) <= 2 for word in despaced_line.split()) or despaced_line.isupper():
                extracted_name = validate_candidate(despaced_line, "Top Lines Heuristic - De-Spaced")
                if extracted_name:
                    break
                    
            # SpaCy Check on Top Lines (if capitalization check failed)
            if not extracted_name and nlp is not None:
                doc_line = nlp(line)
                for ent in doc_line.ents:
                    if ent.label_ == "PERSON":
                        extracted_name = validate_candidate(ent.text, "Top Lines Heuristic - SpaCy")
                        if extracted_name:
                            break
                if extracted_name:
                    break
            
    # 3. 2nd Fallback: Pyresparser Name (Strictly Filtered)
    if not extracted_name:
        pyresparser_name = parsed_data_pyresparser.get("name", "").strip()
        if pyresparser_name and pyresparser_name.lower() not in (final_email or "").lower():
            extracted_name = validate_candidate(pyresparser_name, "Pyresparser Name (2nd Fallback)")

    # 4. 3rd Fallback: SpaCy on Full Text (Last Resort)
    if not extracted_name and nlp is not None:
        spacy_extracted_name_full = extract_name_with_spacy(full_text_content)
        if spacy_extracted_name_full:
            extracted_name = validate_candidate(spacy_extracted_name_full, "SpaCy Full Text (Last Resort)")
            
    # 5. LOWEST PRIORITY FALLBACK: Name from Email Address 
    if not extracted_name and final_email:
        email_prefix = final_email.split('@')[0]

        # Step 1: Handle Common Separators (e.g., alba.krasniqi or alba-krasniqi)
        cleaned_prefix_separated = re.sub(r'[\d_]', ' ', email_prefix)
        cleaned_prefix_separated = re.sub(r'[\.-]', ' ', cleaned_prefix_separated).strip()
        cleaned_prefix_separated = re.sub(r'\s+', ' ', cleaned_prefix_separated) 
        
        if is_valid_name_format(cleaned_prefix_separated):
            extracted_name = cleaned_prefix_separated.title()
            logger.debug(f"Selected name from Email Prefix (Separated) Fallback: '{extracted_name}'")

        # Step 2: Handle Concatenated Names (e.g., albakrasniqi)
        if not extracted_name and len(email_prefix.split()) == 1:
            merged_name = email_prefix
            
            # Test common first name lengths (4 to 8 characters)
            for split in range(4, min(9, len(merged_name) - 3)):
                candidate_with_space = merged_name[:split] + " " + merged_name[split:]
                
                if is_valid_name_format(candidate_with_space):
                    extracted_name = candidate_with_space.title()
                    logger.debug(f"Selected name from Email Prefix (Concatenated Heuristic) Fallback: '{extracted_name}'")
                    break 


    # --- 3.4 FINAL CLEANUP AND RETURN ---
    
    final_name = str(extracted_name).replace('\n', ' ').strip() if extracted_name else ""
    
    # FINAL FIX: Remove any trailing numbers (e.g., Alba Krasniqi1999 -> Alba Krasniqi)
    if final_name:
        final_name = re.sub(r'\s*\d+$', '', final_name).strip()
    
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