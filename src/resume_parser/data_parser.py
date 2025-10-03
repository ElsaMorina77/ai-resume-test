import logging
import re
import spacy
from pyresparser import ResumeParser
from pathlib import Path
from typing import Optional, List, Dict, Any 
from nameparser import HumanName 


# --- 1. LOGGER SETUP ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 

# --- 2. HUGGING FACE IMPORTS AND SETUP ---
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    import torch
    
    HF_MODEL_ID = "yashpwr/resume-ner-bert-v2" 
    
    hf_pipeline = None
    HF_MODEL_LOADED = False
    
    def load_hf_pipeline():
        """Loads the Hugging Face model and sets up the pipeline."""
        global hf_pipeline, HF_MODEL_LOADED
        try:
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
            model = AutoModelForTokenClassification.from_pretrained(HF_MODEL_ID)
            hf_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
            HF_MODEL_LOADED = True
            logger.info(f"Successfully loaded Hugging Face NER model: {HF_MODEL_ID}")
        except Exception as e:
            logger.warning(f"Failed to load Hugging Face model {HF_MODEL_ID}. Reverting to standard fallbacks. Error: {e}")
            HF_MODEL_LOADED = False
            hf_pipeline = None

    load_hf_pipeline()
except ImportError:
    logger.warning("Hugging Face (transformers/torch) not installed. Skipping HF fallback.")
    HF_MODEL_LOADED = False
except Exception as e:
    logger.warning(f"An unexpected error occurred during HF model setup: {e}")
    HF_MODEL_LOADED = False

# --- 3. SPACY SETUP ---
nlp = None
try:
    nlp = spacy.load('en_core_web_lg') 
except Exception:
    try:
        nlp = spacy.load('en_core_web_sm') 
    except Exception:
        logger.error("Could not load spaCy model.")
        nlp = None 

class MockTextExtractor:
    @staticmethod
    def extract_text_from_resume(bytes_, filename):
        return ""
try:
    # Assuming text_extractor utility is available in src.resume_parser
    from src.resume_parser import text_extractor 
except ImportError:
    text_extractor = MockTextExtractor()

# --- 4. HELPER FUNCTIONS ---

def extract_phone_number(text: str) -> Optional[str]: 
    # ... (Keep existing extract_phone_number logic)
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
    # ... (Keep existing extract_email_address logic)
    email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    match = email_regex.search(text)
    if match:
        return match.group(0)
    return None

def extract_name_with_spacy(text: str) -> Optional[str]:
    # ... (Keep existing extract_name_with_spacy logic)
    global nlp 
    if nlp is None:
        return None
    doc = nlp(text[:2048]) 
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def extract_name_with_hf_ner(lines: List[str]) -> Optional[str]:
    # ... (Keep existing extract_name_with_hf_ner logic)
    global hf_pipeline
    if not HF_MODEL_LOADED or hf_pipeline is None:
        return None

    text_to_analyze = "\n".join(lines[:20]) 
    
    try:
        ner_results = hf_pipeline(text_to_analyze)
        
        person_entities = [
            entity['word'].strip() 
            for entity in ner_results 
            if entity['entity_group'].upper() == 'NAME' or entity['entity_group'].upper() == 'PER'
        ]
        
        if person_entities:
            candidates = set()
            for i in range(len(person_entities)):
                if i < len(person_entities) - 1:
                    two_word = f"{person_entities[i]} {person_entities[i+1]}".replace(' ##', '').replace('#', '').strip()
                    if len(two_word.split()) >= 2:
                         candidates.add(two_word)
                candidates.add(person_entities[i].replace(' ##', '').replace('#', '').strip())

            best_name = None
            for candidate in sorted(candidates, key=len, reverse=True):
                # Clean before validating
                cleaned_candidate = re.sub(r'(?i)\s*(CV|RESUME|COMPRESSED|MOTIVATION LETTER|LINKPLUSIT|BOOKLET|CERTIFIED|DIGITAL|TRANSFORMATION|EXPERT)\s*', ' ', candidate).strip()
                if is_valid_name_format(cleaned_candidate):
                    best_name = cleaned_candidate
                    logger.debug(f"HF NER successfully extracted: {best_name}")
                    return best_name
            
            logger.debug(f"HF NER found entities but none passed name validation: {candidates}")
            return None

    except Exception as e:
        logger.error(f"Error during Hugging Face NER inference: {e}")
        return None
    
    return None


def aggressively_despace_name(text: str) -> Optional[str]:
    # ... (Keep existing aggressively_despace_name logic)
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
    # ... (Keep existing extract_name_from_filename logic, but add 'COMPRESSED')
    name_part = Path(filename).stem
    # ADDED 'COMPRESSED' to file name cleanup
    name_part = re.sub(r'(?i)CV|RESUME|PORTFOLIO|DOKUMENT|(?:\s*\(\d+\))|\d{4,}|COMPRESSED', ' ', name_part).strip() 
    cleaned_name = re.sub(r'[\s\._-]+', ' ', name_part).strip()
    
    if is_valid_name_format(cleaned_name):
        return cleaned_name.title()
    
    if len(cleaned_name.split()) == 1:
        concatenated_split = aggressively_despace_name(cleaned_name)
        if concatenated_split:
            return concatenated_split
        
    return None


def is_valid_name_format(name_candidate: str) -> bool:
    # ... (Keep existing is_valid_name_format logic)
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

# --- 5. MAIN PARSING FUNCTION ---

def parse_resume_data(bytes_: bytes, filename: str) -> Dict[str, Any]:
    # ... (Keep Pyresparser execution section)
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

    full_text_content = text_extractor.extract_text_from_resume(bytes_, filename)
    pre_processed_text = full_text_content 

    # --- 5.2 CONTACT EXTRACTION AND PRE-PROCESSING ---
    pyresparser_phone = parsed_data_pyresparser.get("mobile_number", "")
    custom_extracted_phone = extract_phone_number(full_text_content) if full_text_content else None
    final_phone = custom_extracted_phone if custom_extracted_phone else pyresparser_phone 

    pyresparser_email = parsed_data_pyresparser.get("email", "")
    custom_extracted_email = extract_email_address(full_text_content) if full_text_content else None
    final_email = custom_extracted_email if custom_extracted_email else pyresparser_email

    if final_email and pre_processed_text:
        concat_pattern = re.compile(r'(\S)' + re.escape(final_email))
        match = concat_pattern.search(pre_processed_text)
        if match and not match.group(1).isspace() and match.group(1) not in (':', ';', '('):
            pre_processed_text = pre_processed_text.replace(final_email, ' ' + final_email, 1)

    all_lines = [line.strip() for line in (pre_processed_text or "").split('\n') if line.strip()]
    lines_top_of_resume = all_lines[:15]

    # --- 5.3 ENHANCED NAME EXTRACTION LOGIC ---
    extracted_name = ""
    
    # --- EXPANDED BLACKLIST ---
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
        "booklet", "certified", "expert", "digital transformation", "toolkit", 
        "assessors nominations form", "nomination form for final", "leter motivuese",
        "troubleshooting enthusiast", "javafx interaction desktop app", "cover letter",
        "zhvillues softueri", "introduction", "copy of ahmetaj", "training flyer",
        "reference by jone cd", "cv by", "cv for", "motivation letter", "docu ment", 
        "about myself", "tailwind", "creating", "university", "college", "key facts", # <- ADDED
    ] 
    generic_keywords_lower = set(kw.lower() for kw in generic_keywords)
    
    def is_generic_or_bad(text: str) -> bool:
        text_lower = text.lower()
        if any(kw in text_lower for kw in generic_keywords_lower): return True
        if len(text.split()) == 1 and text_lower in generic_keywords_lower: return True
        if re.search(r'[^\w\s-]', text) and not re.search(r'^\w+@\w+', text): return True
        return False
        
    def validate_candidate(candidate: str, source: str) -> Optional[str]:
        candidate = candidate.replace('\n', ' ').strip()
        
        # --- NEW CLEANING STEP ---
        # Remove common non-name words from the candidate string before checking
        candidate = re.sub(r'(?i)\s*(CV|RESUME|COMPRESSED|MOTIVATION LETTER|LINKPLUSIT|BOOKLET|CERTIFIED|DIGITAL|TRANSFORMATION|EXPERT)\s*', ' ', candidate).strip()
        
        words = candidate.split()
        if not (1 <= len(words) <= 6): return None
        if is_generic_or_bad(candidate): return None
        
        if len(words) == 1:
             concatenated_split = aggressively_despace_name(candidate)
             if concatenated_split:
                 candidate = concatenated_split
                 words = candidate.split()
                 if not (2 <= len(words) <= 6): return None
        
        if not is_valid_name_format(candidate): return None
        logger.debug(f"VALIDATED name candidate from {source}: '{candidate}'")
        return candidate
        
    # --- NAME EXTRACTION PRIORITY CHAIN ---

    # 1. Email Proximity Heuristic 
    if final_email:
        for i, line in enumerate(all_lines):
            if final_email in line:
                if i > 0:
                    extracted_name = validate_candidate(all_lines[i-1], "Email Proximity - Prev Line")
                    if extracted_name: break
                if not extracted_name and nlp is not None:
                    doc_line = nlp(line)
                    for ent in doc_line.ents:
                        if ent.label_ == "PERSON":
                            extracted_name = validate_candidate(ent.text, "Email Proximity - SpaCy on Email Line")
                            if extracted_name: break
                    if extracted_name: break
        
    # 1a. Labeled Name Heuristic
    if not extracted_name:
        name_label_regex = re.compile(r'(\b(?:full\s*name|name|applicant|candidate|personal\s*details)\s*[:]?\s*)(.+)', re.IGNORECASE)
        for line in lines_top_of_resume:
            match = name_label_regex.search(line)
            if match:
                candidate = match.group(2).strip()
                if any(kw in candidate.lower() for kw in ["email", "phone", "contact", "@", "address", "linkedin"]): continue
                extracted_name = validate_candidate(candidate, "Labeled Name Heuristic")
                if extracted_name: break

    # 1b. Name from File Name Heuristic
    if not extracted_name and filename:
        name_from_file = extract_name_from_filename(filename)
        if name_from_file:
            extracted_name = name_from_file
            
    # 1c. High-Confidence Name-Label Extraction
    if not extracted_name:
        for line in lines_top_of_resume:
            temp_candidate = line
            if final_email: temp_candidate = temp_candidate.replace(final_email, ' ')
            if final_phone: temp_candidate = temp_candidate.replace(final_phone, ' ')
            temp_candidate = re.sub(r'(?i)^\s*(email|phone|contact|mobile|linkedin|github)\s*[:\s]*', '', temp_candidate).strip()
            temp_candidate = re.sub(r'(?i)\s*[:\s]*(email|phone|contact|mobile|linkedin|github)\s*$', '', temp_candidate).strip()
            cleaned_candidate = re.sub(r'\s+', ' ', temp_candidate).strip()
            is_capitalized = all(word[0].isupper() or len(word) <= 2 for word in cleaned_candidate.split()) or cleaned_candidate.isupper()
            if len(cleaned_candidate.split()) >= 2 and is_capitalized:
                extracted_name = validate_candidate(cleaned_candidate, "High-Confidence Name-Label Clean-Up")
                if extracted_name: break

    # 2. NEW: HUGGING FACE NER FALLBACK (High-Accuracy - MOVED UP IN PRIORITY)
    if not extracted_name and HF_MODEL_LOADED:
        hf_extracted_name = extract_name_with_hf_ner(lines_top_of_resume)
        if hf_extracted_name:
            # Note: Validation happens inside extract_name_with_hf_ner now, but we re-validate here for safety
            extracted_name = validate_candidate(hf_extracted_name, "Hugging Face NER Fallback") 
            if extracted_name:
                logger.debug(f"Selected name from HF NER Fallback: '{extracted_name}'")
    
    # 3. 2nd Fallback: Top Lines Heuristic (Original #2, now #3)
    if not extracted_name:
        for line in lines_top_of_resume:
            despaced_line = re.sub(r'\s+', ' ', line).strip()
            
            aggressively_despaced_name = aggressively_despace_name(line)
            if aggressively_despaced_name:
                extracted_name = aggressively_despaced_name
                break
            
            if all(word[0].isupper() or len(word) <= 2 for word in despaced_line.split()) or despaced_line.isupper():
                extracted_name = validate_candidate(despaced_line, "Top Lines Heuristic - De-Spaced")
                if extracted_name: break
                    
            if not extracted_name and nlp is not None:
                doc_line = nlp(line) 
                for ent in doc_line.ents:
                    if ent.label_ == "PERSON":
                        extracted_name = validate_candidate(ent.text, "Top Lines Heuristic - SpaCy")
                        if extracted_name: break
                if extracted_name: break
            
    # 4. 3rd Fallback: Pyresparser Name (Original #3, now #4)
    if not extracted_name:
        pyresparser_name = parsed_data_pyresparser.get("name", "").strip()
        if pyresparser_name and pyresparser_name.lower() not in (final_email or "").lower():
            extracted_name = validate_candidate(pyresparser_name, "Pyresparser Name (3rd Fallback)")


    # 5. 4th Fallback: SpaCy on Full Text (Last Resort - Original #5, now #5)
    if not extracted_name and nlp is not None:
        spacy_extracted_name_full = extract_name_with_spacy(full_text_content)
        if spacy_extracted_name_full:
            extracted_name = validate_candidate(spacy_extracted_name_full, "SpaCy Full Text (Last Resort)")
            
    # 6. LOWEST PRIORITY FALLBACK: Name from Email Address
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


    # --- 5.4 FINAL CLEANUP AND RETURN ---
    
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