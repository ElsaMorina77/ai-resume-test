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
# ... (Keep this section as is) ...
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

    def load_hf_pipeline(model_id: str):
        """Loads the Hugging Face model and sets up the pipeline."""
        if model_id in hf_pipelines:
            return hf_pipelines[model_id]

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForTokenClassification.from_pretrained(model_id)
            # Use 'max' aggregation for the best-scoring entity span
            pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
            hf_pipelines[model_id] = pipe
            logger.info(f"Successfully loaded Hugging Face NER model: {model_id}")
            return pipe
        except Exception as e:
            logger.warning(f"Failed to load Hugging Face model {model_id}. Error: {e}")
            return None

except ImportError:
    logger.warning("Hugging Face (transformers/torch) not installed. Skipping HF fallback.")
    hf_pipelines = {}

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
# ... (Keep this function as is) ...
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
# ... (Keep this function as is) ...
    email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    match = email_regex.search(text)
    if match:
        return match.group(0)
    return None

def extract_name_with_spacy(text: str) -> Optional[str]:
# ... (Keep this function as is) ...
    global nlp
    if nlp is None:
        return None
    doc = nlp(text[:2048])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def clean_name_string(name: str) -> str:
    """Clean up a name string by removing common artifacts."""
    if not name:
        return ""

    # Remove artifacts from tokenized NER
    name = name.replace('##', '').replace('Ġ', '').strip()

    # Remove multiple spaces
    name = re.sub(r'\s+', ' ', name)

    # Remove leading/trailing non-alphabetic/space characters (allow internal hyphens/apostrophes)
    name = re.sub(r'^[^a-zA-Z\s]+|[^a-zA-Z\s]+$', '', name)

    # Remove common separators if they appear as leading/trailing chars
    name = re.sub(r'^[-_,.:;]+|[-_,.:;]+$', '', name)

    # Remove digits
    name = re.sub(r'\d+', '', name)

    # Title case
    name = name.strip().title()

    # Aggressively remove single trailing non-alphabetic characters unless they are part of a name (e.g., O'Neil)
    if len(name) > 2 and name[-1] in '.,:;':
         name = name[:-1]

    return name.strip()


def is_valid_name_format(name_candidate: str) -> bool:
    """Enhanced name validation with better logic."""
    if not name_candidate:
        return False

    cleaned = clean_name_string(name_candidate)
    words = cleaned.split()

    # 1. Word Count Check: Must have 2-4 words (First Last, First Middle Last, etc.).
    if not (2 <= len(words) < 5):
        return False

    # 2. Length Check: Each word should be at least 2 characters.
    if any(len(word) < 2 for word in words):
        return False

    # 3. Invalid Character Check: Allow letters, spaces, hyphens, and apostrophes only.
    if re.search(r'[^a-zA-Z\s\'-]', cleaned):
        return False

    # 4. Use HumanName parser for structural validation
    try:
        parsed_name = HumanName(cleaned)
    except Exception:
        return False

    # 5. Must have at least a First and Last name component
    if parsed_name.first and parsed_name.last:
        # Both must be at least 2 characters (already checked, but safety check)
        if len(parsed_name.first) >= 2 and len(parsed_name.last) >= 2:
             # Additional check: names like 'A B' or 'T A' are valid here but often resume artifacts.
             # We rely on is_generic_text to catch most of these, but a quick check:
             if any(len(word) == 1 for word in words) and len(words) > 2:
                 # Allow initials, but be cautious with too many single letters
                 pass
             return True

    return False

# --- Stronger Filtering for Generic/Junk Text ---
def is_generic_text(text: str) -> bool:
    """Check if text contains generic resume keywords or common junk."""
    generic_keywords = {
        "experienced", "developer", "professional", "resume", "cv", "curriculum vitae",
        "profile", "contact", "summary", "education", "experience", "skills", "projects",
        "full-stack", "front-end", "back-end", "manager", "engineer", "analyst",
        "github", "linkedin", "mobile", "number", "email", "phone", "address",
        "objective", "about me", "references", "work history", "career summary",
        "technical skills", "personal details", "portfolio", "cover letter", "personal information",
        "recommendations", "certifications", "achievements", "awards", "publications",
        "languages", "interests", "hobbies", "volunteer", "internship",
        "recommendation letter", "leter motivuese", "certificate", "emri dhe mbiemri", "zhvillues softueri",
        "street", "prishtine", "kosove", "rruga", "nr", "shkolla e mesme", "universiteti", "diplome",
        "internal applications", "motivationletterlp", "personal info", "graphic designer",
        "xhevdet doda", "need more information", "networking opportunities",
        "more information", "cmc global", "troubleshooting enthusiast" , "curriculum vit", "podujeva kosova"
        "web cacttus", "exper ience", "experience", "mysql for this project", "reference available upon request"# These are common artifacts in your examples
    }

    text_lower = text.lower().replace('.', ' ').strip()
    words = text_lower.split()

    # Rule 1: Check against the list of known generic/junk phrases
    if any(keyword in text_lower for keyword in generic_keywords):
        return True

    # Rule 2: Aggressive check for common headings/titles (e.g., if it's one of the first few words)
    if words and (words[0] in {"personal", "contact", "full", "mobile", "phone", "email", "address", "about"} or words[-1] in {"info", "details", "letter", "cv"}):
        return True

    # Rule 3: Check for single-word entities that are common placeholders (less likely with word count check, but a safety)
    if len(words) == 1 and text_lower in {"name", "title", "contact", "profile", "applicant", "candidate"}:
         return True

    return False


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
                # Check for high confidence if available, though aggregation strategy 'simple' usually helps
                if label in ['NAME', 'PER', 'PERSON']:
                    clean_word = clean_name_string(entity['word']) # Use the improved cleaner early
                    if clean_word and len(clean_word.split()) >= 2: # Only consider multi-word entities from NER
                        person_entities.append(clean_word)

            if person_entities:
                # Deduplicate, don't re-combine to avoid mixing unrelated entities, as the model should find the full name
                valid_candidates = [
                    cand for cand in set(person_entities)
                    if not is_generic_text(cand) and is_valid_name_format(cand)
                ]

                if valid_candidates:
                    # Select the longest and most syntactically valid (via is_valid_name_format)
                    best = max(valid_candidates, key=lambda x: len(x.split()))
                    logger.info(f"✓ Name from {model_id}: {best}")
                    return best

        except Exception as e:
            logger.error(f"Error during inference with {model_id}: {e}")

    return None

def extract_name_from_filename(filename: str) -> Optional[str]:
# ... (Keep this function as is) ...
    """Extract name from filename."""
    name_part = Path(filename).stem

    # Remove common resume keywords
    keywords_to_remove = [
        r'(?i)CV', r'(?i)RESUME', r'(?i)PORTFOLIO', r'(?i)DOCUMENT',
        r'\(\d+\)', r'\d{4,}', r'(?i)_final', r'(?i)_updated', r'(?i)MotivationLetterlp',
        r'(?i)Certificate', r'(?i)RecommendationLetter', r'(?i)InternalApplications'
    ]

    for pattern in keywords_to_remove:
        name_part = re.sub(pattern, ' ', name_part)

    # Replace separators with spaces
    name_part = re.sub(r'[\s\._-]+', ' ', name_part).strip()

    cleaned = clean_name_string(name_part)

    if is_valid_name_format(cleaned):
        return cleaned

    return None


def extract_name_near_email(lines: List[str], email: str) -> Optional[str]:
    """Extract name from lines near the email address."""
    for i, line in enumerate(lines):
        if email in line:
            # Check previous line
            if i > 0:
                candidate = lines[i - 1]
                cleaned = clean_name_string(candidate)
                # Ensure it's not generic AND is a valid name format
                if not is_generic_text(cleaned) and is_valid_name_format(cleaned):
                    logger.debug(f"Found name above email: {cleaned}")
                    return cleaned

            # Check same line (before email)
            before_email = line.split(email)[0].strip()
            # aggressive split on common separators near email that might be non-name text
            before_email = re.split(r'[:;,\|\-]', before_email)[-1].strip()
            cleaned = clean_name_string(before_email)
            if not is_generic_text(cleaned) and is_valid_name_format(cleaned):
                logger.debug(f"Found name on email line: {cleaned}")
                return cleaned

    return None


def extract_name_from_labeled_field(lines: List[str]) -> Optional[str]:
    """Extract name from fields labeled as 'Name:', 'Full Name:', etc."""
    name_patterns = [
        r'(?i)^(?:full\s+)?name\s*[:\-\|]?\s*(.+)$', # Added |
        r'(?i)^(?:applicant|candidate)\s*[:\-\|]?\s*(.+)$',
        r'(?i)^(?:emri\s+dhe\s+mbiemri)\s*[:\-\|]?\s*(.+)$' # Added Albanian translation
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
# ... (Keep this function as is, but it will use the improved helpers) ...
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