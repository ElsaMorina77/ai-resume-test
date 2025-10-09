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
    
    HF_MODEL_IDS = [
        "yashpwr/resume-ner-bert-v2", # resume specific BERT
        "dslim/bert-base-NER", # BERT general purpose NER
        "Jean-Baptiste/roberta-large-ner-english", # RoBERTa NER 
        "Davlan/distilbert-base-multilingual-case-ner-hrl", #Multilingual fallback
        ]
    
    hf_pipelines = {}
    #HF_MODEL_LOADED = False
    
    def load_hf_pipeline(model_id: str):
        """Loads the Hugging Face model and sets up the pipeline."""
        #global hf_pipeline, HF_MODEL_LOADED
        if model_id in hf_pipelines:
            return hf_pipelines[model_id]
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForTokenClassification.from_pretrained(model_id)
            pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
            hf_pipelines[model_id] = pipe
            logger.info(f"Successfully loaded Hugging Face NER model: {model_id}")
            return pipe
        except Exception as e:
            logger.warning(f"Failed to load Hugging Face model {model_id}. Error: {e}")
            #HF_MODEL_LOADED = False
            #hf_pipeline = None
            return None

    #load_hf_pipeline()
except ImportError:
    logger.warning("Hugging Face (transformers/torch) not installed. Skipping HF fallback.")
    hf_pipelines = {}
    #HF_MODEL_LOADED = False
#except Exception as e:
    #logger.warning(f"An unexpected error occurred during HF model setup: {e}")
    #HF_MODEL_LOADED = False

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
    from src.resume_parser import text_extractor 
except ImportError:
    text_extractor = MockTextExtractor()

# --- 4. HELPER FUNCTIONS ---

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

def extract_name_with_hf_ner(lines: List[str]) -> Optional[str]:
    """Try multiple Hugging Face models to extract a name."""
    text_to_analyze = "\n".join(lines[:100])  # Limit scope
    
    for model_id in HF_MODEL_IDS:
        pipe = load_hf_pipeline(model_id)
        if pipe is None:
            continue
        
        try:
            ner_results = pipe(text_to_analyze)
            person_entities = []
            for entity in ner_results:
                label = entity['entity_group'].upper()
                if label in ['NAME', 'PER', 'PERSON']:
                    clean_word = entity['word'].replace('##', '').replace('Ġ', '').strip()
                    if clean_word:
                        person_entities.append(clean_word)
            
            if person_entities:
                # Same combining logic as before
                candidates = set(person_entities)
                for i in range(len(person_entities) - 1):
                    candidates.add(f"{person_entities[i]} {person_entities[i+1]}")
                for i in range(len(person_entities) - 2):
                    candidates.add(f"{person_entities[i]} {person_entities[i+1]} {person_entities[i+2]}")
                
                valid_candidates = [
                    clean_name_string(cand) for cand in candidates
                    if is_valid_name_format(clean_name_string(cand))
                ]
                
                if valid_candidates:
                    best = max(valid_candidates, key=lambda x: len(x.split()))
                    logger.info(f"✓ Name from {model_id}: {best}")
                    return best

        except Exception as e:
            logger.error(f"Error during inference with {model_id}: {e}")
    
    return None


def clean_name_string(name: str) -> str:
    """Clean up a name string by removing common artifacts."""
    if not name:
        return ""
    
    # Remove multiple spaces
    name = re.sub(r'\s+', ' ', name)
    
    # Remove leading/trailing punctuation
    name = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', name)
    
    # Remove digits
    name = re.sub(r'\d+', '', name)
    
    # Title case
    name = name.strip().title()
    
    return name


def is_valid_name_format(name_candidate: str) -> bool:
    """Enhanced name validation with better logic."""
    if not name_candidate:
        return False
    
    # Clean the candidate first
    cleaned = clean_name_string(name_candidate)
    words = cleaned.split()
    
    # Must have 2-4 words (First Last, or First Middle Last, etc.)
    if not (2 <= len(words) < 4):
        return False
    
    # Each word should be at least 2 characters
    if any(len(word) < 2 for word in words):
        return False
    
    # Check for invalid characters (but allow hyphens and apostrophes in names)
    if re.search(r'[^a-zA-Z\s\'-]', cleaned):
        return False
    
    # Use HumanName parser for validation
    parsed_name = HumanName(cleaned)
    
    # Must have at least first and last name
    if parsed_name.first and parsed_name.last:
        # Both must be at least 2 characters
        if len(parsed_name.first) >= 2 and len(parsed_name.last) >= 2:
            return True
    
    return False


def extract_name_from_filename(filename: str) -> Optional[str]:
    """Extract name from filename."""
    name_part = Path(filename).stem
    
    # Remove common resume keywords
    keywords_to_remove = [
        r'(?i)CV', r'(?i)RESUME', r'(?i)PORTFOLIO', r'(?i)DOCUMENT',
        r'\(\d+\)', r'\d{4,}', r'(?i)_final', r'(?i)_updated'
    ]
    
    for pattern in keywords_to_remove:
        name_part = re.sub(pattern, ' ', name_part)
    
    # Replace separators with spaces
    name_part = re.sub(r'[\s\._-]+', ' ', name_part).strip()
    
    cleaned = clean_name_string(name_part)
    
    if is_valid_name_format(cleaned):
        return cleaned
        
    return None


def is_generic_text(text: str) -> bool:
    """Check if text contains generic resume keywords."""
    generic_keywords = {
        "experienced", "developer", "professional", "resume", "cv", "curriculum vitae",
        "profile", "contact", "summary", "education", "experience", "skills", "projects",
        "full-stack", "front-end", "back-end", "manager", "engineer", "analyst",
        "github", "linkedin", "mobile", "number", "email", "phone", "address",
        "objective", "about me", "references", "work history", "career summary",
        "technical skills", "personal details", "portfolio", "cover letter", "personal information",
        "recommendations", "certifications", "achievements", "awards", "publications",
        "languages", "interests", "hobbies", "volunteer", "internship", "troubleshooting enthusiast",
        "recommendation letter", "leter motivuese", "certificate", "emri dhe mbiemri", "zhvillues softueri",
        "Xhafer Mustafa Street", "Prishtine", "Kosove", "Rruga", "Nr", "Shkolla e mesme", "Universiteti", "Diplome",
        "Internal Applications", "Motivationletterlp", "Personal Info", "Xhevdet Doda", "Graphic Designer"
    }
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in generic_keywords)


def extract_name_near_email(lines: List[str], email: str) -> Optional[str]:
    """Extract name from lines near the email address."""
    for i, line in enumerate(lines):
        if email in line:
            # Check previous line
            if i > 0:
                candidate = lines[i - 1]
                cleaned = clean_name_string(candidate)
                if not is_generic_text(cleaned) and is_valid_name_format(cleaned):
                    logger.debug(f"Found name above email: {cleaned}")
                    return cleaned
            
            # Check same line (before email)
            before_email = line.split(email)[0].strip()
            cleaned = clean_name_string(before_email)
            if not is_generic_text(cleaned) and is_valid_name_format(cleaned):
                logger.debug(f"Found name on email line: {cleaned}")
                return cleaned
    
    return None


def extract_name_from_labeled_field(lines: List[str]) -> Optional[str]:
    """Extract name from fields labeled as 'Name:', 'Full Name:', etc."""
    name_patterns = [
        r'(?i)^(?:full\s+)?name\s*[:\-]?\s*(.+)$',
        r'(?i)^(?:applicant|candidate)\s*[:\-]?\s*(.+)$',
    ]
    
    for line in lines:
        for pattern in name_patterns:
            match = re.search(pattern, line)
            if match:
                candidate = match.group(1).strip()
                # Remove common suffixes like email/phone from the same line
                candidate = re.sub(r'[\|\-].*$', '', candidate).strip()
                cleaned = clean_name_string(candidate)
                if not is_generic_text(cleaned) and is_valid_name_format(cleaned):
                    logger.debug(f"Found labeled name: {cleaned}")
                    return cleaned
    
    return None


# --- 5. MAIN PARSING FUNCTION ---

def parse_resume_data(bytes_: bytes, filename: str) -> Dict[str, Any]:
    """Main function to parse resume data."""
    
    # --- PyResParser Execution ---
    temp_file = Path("_tmp_pyresparser_" + Path(filename).name)
    temp_file.write_bytes(bytes_)
    parsed_data = {}
    
    try:
        if nlp is not None:
            try:
                parser = ResumeParser(str(temp_file), custom_nlp=nlp)
                parsed_data = parser.get_extracted_data() or {}
            except Exception:
                parser = ResumeParser(str(temp_file))
                parsed_data = parser.get_extracted_data() or {}
        else:
            parser = ResumeParser(str(temp_file))
            parsed_data = parser.get_extracted_data() or {}
    except Exception as e:
        logger.error(f"Error parsing resume with pyresparser: {e}")
    finally:
        temp_file.unlink(missing_ok=True)

    # Extract full text
    full_text = text_extractor.extract_text_from_resume(bytes_, filename)
    
    # Extract contact info
    email = extract_email_address(full_text) or parsed_data.get("email", "")
    phone = extract_phone_number(full_text) or parsed_data.get("mobile_number", "")
    
    # Get lines for processing
    lines = [line.strip() for line in full_text.split('\n') if line.strip()]
    top_lines = lines[:25]  # Extended window
    
    # --- NAME EXTRACTION PRIORITY CHAIN ---
    extracted_name = None
    
    # 1. Try labeled field extraction (highest confidence)
    if not extracted_name:
        extracted_name = extract_name_from_labeled_field(top_lines)
        if extracted_name:
            logger.info(f"✓ Name from labeled field: {extracted_name}")
    
    # 2. Try name near email
    if not extracted_name and email:
        extracted_name = extract_name_near_email(lines, email)
        if extracted_name:
            logger.info(f"✓ Name near email: {extracted_name}")
    
    # 3. Try Hugging Face NER (high accuracy for resumes)
    if not extracted_name and hf_pipelines:
        extracted_name = extract_name_with_hf_ner(top_lines)
        if extracted_name:
            logger.info(f"✓ Name from HF NER: {extracted_name}")
    
    # 4. Try first line if it looks like a name
    if not extracted_name and top_lines:
        first_line_cleaned = clean_name_string(top_lines[0])
        if not is_generic_text(first_line_cleaned) and is_valid_name_format(first_line_cleaned):
            extracted_name = first_line_cleaned
            logger.info(f"✓ Name from first line: {extracted_name}")
    
    # 5. Try spaCy on top section
    if not extracted_name and nlp:
        top_text = "\n".join(top_lines[:10])
        spacy_name = extract_name_with_spacy(top_text)
        if spacy_name:
            cleaned = clean_name_string(spacy_name)
            if not is_generic_text(cleaned) and is_valid_name_format(cleaned):
                extracted_name = cleaned
                logger.info(f"✓ Name from spaCy: {extracted_name}")
    
    # 6. Try PyResParser result
    if not extracted_name:
        pyres_name = parsed_data.get("name", "")
        if pyres_name:
            cleaned = clean_name_string(pyres_name)
            if not is_generic_text(cleaned) and is_valid_name_format(cleaned):
                extracted_name = cleaned
                logger.info(f"✓ Name from PyResParser: {extracted_name}")
    
    # 7. Try filename
    if not extracted_name:
        file_name = extract_name_from_filename(filename)
        if file_name:
            extracted_name = file_name
            logger.info(f"✓ Name from filename: {extracted_name}")
    
    # 8. Last resort: extract from email
    if not extracted_name and email:
        email_prefix = email.split('@')[0]
        # Split by dots, underscores, hyphens
        parts = re.split(r'[._-]', email_prefix)
        if len(parts) >= 2:
            candidate = ' '.join(parts[:2])  # Take first two parts
            cleaned = clean_name_string(candidate)
            if is_valid_name_format(cleaned):
                extracted_name = cleaned
                logger.info(f"✓ Name from email (last resort): {extracted_name}")
    
    if not extracted_name:
        logger.warning(f"Could not extract name from resume: {filename}")
    
    # Final cleanup
    final_name = extracted_name if extracted_name else ""
    
    return {
        "name": final_name,
        "email_cv": email or "",
        "phone": phone or "",
        "skills": parsed_data.get("skills", []),
        "education": parsed_data.get("education", []),
        "experience": parsed_data.get("experience", []),
        "total_experience": parsed_data.get("total_experience", 0.0),
        "full_text_content": full_text or "",
    }