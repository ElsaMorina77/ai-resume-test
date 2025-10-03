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
nlp = None
try:
    nlp = spacy.load('en_core_web_lg') 
except Exception:
    try:
        nlp = spacy.load('en_core_web_sm') 
    except Exception:
        logger.error("Could not load spaCy model. Please ensure it's installed.")
        nlp = None 

class MockTextExtractor:
    @staticmethod
    def extract_text_from_resume(bytes_, filename):
        # Placeholder for actual text extraction logic
        return ""
try:
    from src.resume_parser import text_extractor 
except ImportError:
    text_extractor = MockTextExtractor()

# --- 2. HELPER FUNCTIONS (No change to internals, just for completeness) ---

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
        return max(potential_numbers, key=lambda x: len(re.sub(r'\D', '', x)))
    return None


def extract_email_address(text: str) -> Optional[str]:
    email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    match = email_regex.search(text)
    if match:
        return match.group(0)
    return None

def extract_name_with_spacy(text: str) -> Optional[str]:
    global nlp 
    if nlp is None:
        return None
    doc = nlp(text[:2048]) 
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def aggressively_despace_name(text: str) -> Optional[str]:
    words = text.split()
    if len(words) >= 4 and all(len(word) <= 2 and word.isupper() for word in words):
        merged_name = "".join(words)
        for split in range(4, min(10, len(merged_name) - 3)):
            candidate_with_space = merged_name[:split] + " " + merged_name[split:]
            if is_valid_name_format(candidate_with_space):
                return candidate_with_space.title()
    
    if len(words) == 1 and len(text) > 7:
        merged_name = text.lower()
        for split in range(4, min(9, len(merged_name) - 3)):
            candidate_with_space = merged_name[:split].title() + " " + merged_name[split:].title()
            if is_valid_name_format(candidate_with_space):
                return candidate_with_space
            
    return None

def extract_name_from_filename(filename: str) -> Optional[str]:
    name_part = Path(filename).stem
    name_part = re.sub(r'(?i)CV|RESUME|PORTFOLIO|DOKUMENT|(?:\s*\(\d+\))|\d{4,}', ' ', name_part).strip()
    cleaned_name = re.sub(r'[\s\._-]+', ' ', name_part).strip()
    
    if is_valid_name_format(cleaned_name):
        return cleaned_name.title()
    
    if len(cleaned_name.split()) == 1:
        concatenated_split = aggressively_despace_name(cleaned_name)
        if concatenated_split:
            return concatenated_split
        
    return None


def is_valid_name_format(name_candidate: str) -> bool:
    if not name_candidate:
        return False
    
    if len(name_candidate.split()) > 6: 
        return False
    
    if re.search(r'[^\w\s-]', name_candidate) and not re.search(r'^\w+@\w+', name_candidate):
        return False
        
    name_parts = HumanName(name_candidate)
    
    if name_parts.first and name_parts.last and len(name_parts.first) > 1 and len(name_parts.last) > 1:
        return True
    
    return False

# --- 3. MAIN PARSING FUNCTION ---

def parse_resume_data(bytes_: bytes, filename: str) -> Dict[str, Any]:
    # --- 3.1 PYRESPARSER EXECUTION ---
    temp_pyresparser_file = Path("_tmp_pyresparser_" + Path(filename).name)
    temp_pyresparser_file.write_bytes(bytes_)
    parsed_data_pyresparser = {}
    
    try:
        # ... (Pyresparser setup, same as before) ...
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

    full_text_content = text_extractor.extract_text_from_resume(bytes_, filename)
    pre_processed_text = full_text_content # Start with the original text

    # --- 3.2 CONTACT EXTRACTION ---
    pyresparser_phone = parsed_data_pyresparser.get("mobile_number", "")
    custom_extracted_phone = extract_phone_number(full_text_content) if full_text_content else None
    final_phone = custom_extracted_phone if custom_extracted_phone else pyresparser_phone 

    pyresparser_email = parsed_data_pyresparser.get("email", "")
    custom_extracted_email = extract_email_address(full_text_content) if full_text_content else None
    final_email = custom_extracted_email if custom_extracted_email else pyresparser_email

    # NEW PRE-PROCESSING STEP: Separate Concatenated Name/Email (Addresses Adrian Shalaadrianxshalax@gmail.com)
    if final_email and pre_processed_text:
        # Check if email is preceded by a non-space character (e.g., 'a' in 'Shalaa' or 'a' in 'Brahaa')
        concat_pattern = re.compile(r'(\S)' + re.escape(final_email))
        
        match = concat_pattern.search(pre_processed_text)
        # Ensure the preceding character isn't a likely separator (like a colon or bracket)
        if match and not match.group(1).isspace() and match.group(1) not in (':', ';', '('):
            # Insert a space between the preceding character and the email
            # We replace the concatenated part with the preceding char + space + email
            # e.g., 'aaltinabraha15@gmail.com' -> 'a altinabraha15@gmail.com'
            pre_processed_text = pre_processed_text.replace(final_email, ' ' + final_email, 1)
            logger.debug(f"Applied concatenation fix for email: inserted space before {final_email}")

    # Use the pre-processed text for line-based analysis
    all_lines = [line.strip() for line in (pre_processed_text or "").split('\n') if line.strip()]
    lines_top_of_resume = all_lines[:10] 

    # --- 3.3 ENHANCED NAME EXTRACTION LOGIC ---
    extracted_name = ""
    
    # Generic Keywords (UPDATED with new document/header terms from latest log)
    generic_keywords = [
        "experienced", "developer", "professional", "resume", "cv", "curriculum vitae", "profile", 
        "contact", "summary", "education", "experience", "skills", "projects", "full-stack", 
        "front-end", "back-end", "manager", "engineer", "analyst", "github", "linkedin",
        "mobile", "number", "email", "mail", "phone", "address", "objective", "about me", 
        "references", "work history", "career summary", "technical skills", "personal details", 
        "city", "street", "vue js", "git", "don bosko", "portfolio", "designer", 
        "freelancer", "tech nexus", "axians", "kutia", "ubt", "product", "bachelor", 
        "master", "doctor", "phd", "pristina", "kosovo", "personal info", "contact info",
        "more information", "internship", "internships", "professional experience", "e-commerce", 
        "developer position", "project manager", "assistant project manager",
        
        # NEW HARDENING (from chat log analysis)
        "booklet", "certified", "expert", "digital transformation", "toolkit", 
        "assessors nominations form", "nomination form for final", "leter motivuese",
        "troubleshooting enthusiast", "javafx interaction desktop app", "cover letter",
        "zhvillues softueri", "introduction", "copy of ahmetaj", "training flyer",
        "reference by jone cd", "cv by", "cv for", "motivation letter"
    ] 
    generic_keywords_lower = set(kw.lower() for kw in generic_keywords)
    
    def is_generic_or_bad(text: str) -> bool:
        # ... (same logic as before) ...
        text_lower = text.lower()
        if any(kw in text_lower for kw in generic_keywords_lower):
            return True
        if len(text.split()) == 1 and text_lower in generic_keywords_lower:
            return True
        if re.search(r'[^\w\s-]', text) and not re.search(r'^\w+@\w+', text):
            return True
        return False
        
    def validate_candidate(candidate: str, source: str) -> Optional[str]:
        # ... (same logic as before) ...
        candidate = candidate.replace('\n', ' ').strip()
        words = candidate.split()
        if not (1 <= len(words) <= 6):
            return None
        if is_generic_or_bad(candidate):
            return None
        # Use aggressively_despace_name for one-word candidates that might be concatenated
        if len(words) == 1:
             concatenated_split = aggressively_despace_name(candidate)
             if concatenated_split:
                 candidate = concatenated_split
                 words = candidate.split() # Re-check after splitting
                 if not (2 <= len(words) <= 6): return None
        
        if not is_valid_name_format(candidate):
            return None
        return candidate
        
    # --- NAME EXTRACTION PRIORITY CHAIN ---
    # The pre-processing step above ensures that the proximity heuristics
    # below will now see a space between the name and email/phone, 
    # making them much more reliable.

    # 1. Email Proximity Heuristic 
    if final_email:
        # ... (rest of the logic is the same, using the now clean all_lines) ...
        for i, line in enumerate(all_lines):
            if final_email in line:
                if i > 0:
                    candidate = all_lines[i-1]
                    extracted_name = validate_candidate(candidate, "Email Proximity - Prev Line")
                    if extracted_name: break
                if not extracted_name and nlp is not None:
                    doc_line = nlp(line)
                    for ent in doc_line.ents:
                        if ent.label_ == "PERSON":
                            extracted_name = validate_candidate(ent.text, "Email Proximity - SpaCy on Email Line")
                            if extracted_name: break
                    if extracted_name: break
        
    # 1a. Labeled Name Heuristic (No change)
    if not extracted_name:
        name_label_regex = re.compile(r'(\b(?:full\s*name|name|applicant|candidate|personal\s*details)\s*[:]?\s*)(.+)', re.IGNORECASE)
        for line in lines_top_of_resume:
            match = name_label_regex.search(line)
            if match:
                candidate = match.group(2).strip()
                if any(kw in candidate.lower() for kw in ["email", "phone", "contact", "@", "address", "linkedin"]):
                     continue
                extracted_name = validate_candidate(candidate, "Labeled Name Heuristic")
                if extracted_name: break

    # 1b. Name from File Name Heuristic (No change)
    if not extracted_name and filename:
        name_from_file = extract_name_from_filename(filename)
        if name_from_file:
            extracted_name = name_from_file
            
    # 1c. High-Confidence Name-Label Extraction (Clean-Up) (No change, remains effective)
    if not extracted_name:
        for line in lines_top_of_resume:
            temp_candidate = line
            
            if final_email:
                temp_candidate = temp_candidate.replace(final_email, ' ')
            if final_phone:
                temp_candidate = temp_candidate.replace(final_phone, ' ')

            temp_candidate = re.sub(r'(?i)^\s*(email|phone|contact|mobile|linkedin|github)\s*[:\s]*', '', temp_candidate).strip()
            temp_candidate = re.sub(r'(?i)\s*[:\s]*(email|phone|contact|mobile|linkedin|github)\s*$', '', temp_candidate).strip()
            
            cleaned_candidate = re.sub(r'\s+', ' ', temp_candidate).strip()
            
            is_capitalized = all(word[0].isupper() or len(word) <= 2 for word in cleaned_candidate.split()) or cleaned_candidate.isupper()
            
            if len(cleaned_candidate.split()) >= 2 and is_capitalized:
                extracted_name = validate_candidate(cleaned_candidate, "High-Confidence Name-Label Clean-Up")
                if extracted_name:
                    break

            
    # 2. 1st Fallback: Top Lines Heuristic 
    if not extracted_name:
        for line in lines_top_of_resume:
            despaced_line = re.sub(r'\s+', ' ', line).strip()
            
            aggressively_despaced_name = aggressively_despace_name(line)
            if aggressively_despaced_name:
                extracted_name = aggressively_despaced_name
                break
            
            if all(word[0].isupper() or len(word) <= 2 for word in despaced_line.split()) or despaced_line.isupper():
                extracted_name = validate_candidate(despaced_line, "Top Lines Heuristic - De-Spaced")
                if extracted_name:
                    break
                    
            if not extracted_name and nlp is not None:
                doc_line = nlp(line) 
                for ent in doc_line.ents:
                    if ent.label_ == "PERSON":
                        extracted_name = validate_candidate(ent.text, "Top Lines Heuristic - SpaCy")
                        if extracted_name:
                            break
                if extracted_name:
                    break
            
    # 3. 2nd Fallback: Pyresparser Name (No change)
    if not extracted_name:
        pyresparser_name = parsed_data_pyresparser.get("name", "").strip()
        if pyresparser_name and pyresparser_name.lower() not in (final_email or "").lower():
            extracted_name = validate_candidate(pyresparser_name, "Pyresparser Name (2nd Fallback)")

    # 4. 3rd Fallback: SpaCy on Full Text (Last Resort) (No change)
    if not extracted_name and nlp is not None:
        spacy_extracted_name_full = extract_name_with_spacy(full_text_content)
        if spacy_extracted_name_full:
            extracted_name = validate_candidate(spacy_extracted_name_full, "SpaCy Full Text (Last Resort)")
            
    # 5. LOWEST PRIORITY FALLBACK: Name from Email Address (No change)
    if not extracted_name and final_email:
        email_prefix = final_email.split('@')[0]
        cleaned_prefix_separated = re.sub(r'[\d_]', ' ', email_prefix)
        cleaned_prefix_separated = re.sub(r'[\.-]', ' ', cleaned_prefix_separated).strip()
        cleaned_prefix_separated = re.sub(r'\s+', ' ', cleaned_prefix_separated) 
        
        if is_valid_name_format(cleaned_prefix_separated):
            extracted_name = cleaned_prefix_separated.title()

        if not extracted_name and len(email_prefix.split()) == 1:
            merged_name = email_prefix
            for split in range(4, min(9, len(merged_name) - 3)):
                candidate_with_space = merged_name[:split] + " " + merged_name[split:]
                if is_valid_name_format(candidate_with_space):
                    extracted_name = candidate_with_space.title()
                    break 


    # --- 3.4 FINAL CLEANUP AND RETURN ---
    
    final_name = str(extracted_name).replace('\n', ' ').strip() if extracted_name else ""
    
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