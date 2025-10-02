import logging
import re
import spacy
from pyresparser import ResumeParser
from pathlib import Path
from typing import Optional, Dict, Any
from src.resume_parser import text_extractor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SPACY_MODEL = "en_core_web_md"

try:
    nlp = spacy.load(SPACY_MODEL)
    logger.info(f"Loaded spaCy model: {SPACY_MODEL}")
except Exception as e_md:
    logger.warning(f"Could not load '{SPACY_MODEL}', falling back to 'en_core_web_sm': {e_md}")
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model: en_core_web_sm")
    except Exception as e_sm:
        logger.error(f"Could not load any spaCy model: {e_sm}")
        nlp = None


"""def extract_name_with_spacy(text: str) -> Optional[str]:
    if not text or nlp is None:
        return None

    generic_keywords = [
        "experienced", "developer", "professional", "resume", "cv",
        "curriculum vitae", "profile", "contact", "summary",
        "education", "experience", "skills", "projects"
    ]

    lines = [line.strip() for line in text.split("\n") if line.strip()][:5]
    for line in lines:
        doc = nlp(line)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                candidate = ent.text.strip()
                words = candidate.split()
                if len(words) >= 2 and not any(kw in candidate.lower() for kw in generic_keywords):
                    return candidate

    doc_full = nlp(text)
    for ent in doc_full.ents:
        if ent.label_ == "PERSON":
            candidate = ent.text.strip()
            if len(candidate.split()) >= 2 and not any(kw in candidate.lower() for kw in generic_keywords):
                return candidate

    return None"""

def extract_name_with_spacy(text: str) -> Optional[str]:
    if not text or nlp is None:
        return None

    generic_keywords = [
        "experienced", "developer", "professional", "resume", "cv",
        "curriculum vitae", "profile", "contact", "summary",
        "education", "experience", "skills", "projects"
    ]

    lines = [line.strip() for line in text.split("\n") if line.strip()][:10]  # top 10 lines
    for line in lines:
        doc = nlp(line)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                candidate = ent.text.strip()
                words = candidate.split()
                # Filter: at least 2 words, no generic keywords, no digits, each word starts with uppercase
                if (len(words) >= 2 and
                    all(word[0].isupper() or len(word)<=2 for word in words) and
                    not any(char.isdigit() for char in candidate) and
                    not any(kw in candidate.lower() for kw in generic_keywords)):
                    return candidate

    # Full text fallback
    doc_full = nlp(text)
    for ent in doc_full.ents:
        if ent.label_ == "PERSON":
            candidate = ent.text.strip()
            words = candidate.split()
            if (len(words) >= 2 and
                all(word[0].isupper() or len(word)<=2 for word in words) and
                not any(char.isdigit() for char in candidate) and
                not any(kw in candidate.lower() for kw in generic_keywords)):
                return candidate

    return None



"""def extract_name_from_email(email: str, text: str) -> Optional[str]:
    if not email:
        return None
    base = email.split('@')[0].replace('.', ' ').replace('_', ' ').title()
    # Check if this name exists in the first 10 lines
    for line in text.split('\n')[:10]:
        if all(word in line for word in base.split()):
            return line.strip()
    return base"""


def extract_name_from_email(email: str) -> Optional[str]:
    if not email:
        return None
    # Take part before @, replace separators with spaces
    base = email.split('@')[0].replace('.', ' ').replace('_', ' ').title()
    # Only accept if it looks like a real name: no digits, at least 2 words
    if any(char.isdigit() for char in base):
        return None
    if len(base.split()) < 2:
        return None
    return base



def extract_phone_number(text: str) -> Optional[str]:
    phone_regex = re.compile(
        r'''
        (?:(?:\+|00)\d{1,3}[-.\s]?)?
        (?:\(0?\d{2,5}\)?[-.\s]?)?
        \d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}
        |
        \b(?<!\d)\d{7,15}\b
        ''', re.VERBOSE
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
        return best_phone

    return None


def normalize_phone(phone: str) -> str:
    if not phone:
        return ""
    digits = re.sub(r'\D', '', phone)
    if digits.startswith('0') and len(digits) == 9:
        return '+383 ' + digits[1:4] + ' ' + digits[4:6] + ' ' + digits[6:]
    elif digits.startswith('383') and len(digits) == 12:
        return '+383 ' + digits[3:6] + ' ' + digits[6:9] + ' ' + digits[9:]
    return phone


def extract_email_address(text: str) -> Optional[str]:
    email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    match = email_regex.search(text)
    if match:
        return match.group(0)
    return None


"""def parse_resume_data(bytes_: bytes, filename: str) -> Dict[str, Any]:
    temp_file = Path("_tmp_pyresparser_" + Path(filename).name)
    temp_file.write_bytes(bytes_)

    parsed_data_pyresparser = {}
    try:
        if nlp is not None:
            try:
                parser = ResumeParser(str(temp_file), custom_nlp=nlp)
                parsed_data_pyresparser = parser.get_extracted_data() or {}
            except TypeError:
                parser = ResumeParser(str(temp_file))
                parsed_data_pyresparser = parser.get_extracted_data() or {}
        else:
            parser = ResumeParser(str(temp_file))
            parsed_data_pyresparser = parser.get_extracted_data() or {}
    except Exception as e:
        logger.error(f"Error parsing resume '{filename}': {e}")
        parsed_data_pyresparser = {}
    finally:
        temp_file.unlink(missing_ok=True)

    full_text = text_extractor.extract_text_from_resume(bytes_, filename) or ""

    # Phone
    pyresparser_phone = parsed_data_pyresparser.get("mobile_number", "")
    custom_phone = extract_phone_number(full_text)
    final_phone = pyresparser_phone or custom_phone
    final_phone_cleaned = normalize_phone(final_phone)

    # Email
    pyresparser_email = parsed_data_pyresparser.get("email", "")
    custom_email = extract_email_address(full_text)
    final_email = pyresparser_email or custom_email
    final_email_cleaned = re.sub(r'[^\w@.+-]', '', final_email) if final_email else ""

    # Name
    extracted_name = ""

    # Priority 1: Top 5 lines spaCy
    if nlp is not None:
        lines_top = [line.strip() for line in full_text.split('\n') if line.strip()][:5]
        for line in lines_top:
            doc = nlp(line)
            for ent in doc.ents:
                if ent.label_ == "PERSON" and len(ent.text.split()) >= 2:
                    extracted_name = ent.text.strip()
                    break
            if extracted_name:
                break

    # Priority 2: pyresparser
    if not extracted_name:
        name_pyresparser = parsed_data_pyresparser.get("name", "").strip()
        if name_pyresparser:
            extracted_name = name_pyresparser

    # Priority 3: email-based
    if not extracted_name:
        extracted_name = extract_name_from_email(final_email_cleaned, full_text)

    # Priority 4: spaCy full-text fallback
    if not extracted_name and nlp is not None:
        extracted_name = extract_name_with_spacy(full_text)

    final_name = extracted_name.replace('\n', ' ').strip() if extracted_name else ""

    return {
        "name": final_name,
        "email_cv": final_email_cleaned,
        "phone": final_phone_cleaned,
        "skills": parsed_data_pyresparser.get("skills", []),
        "education": parsed_data_pyresparser.get("education", []),
        "experience": parsed_data_pyresparser.get("experience", []),
        "total_experience": parsed_data_pyresparser.get("total_experience", 0.0),
        "full_text_content": full_text,
    }"""


def parse_resume_data(bytes_: bytes, filename: str) -> Dict[str, Any]:
    temp_file = Path("_tmp_pyresparser_" + Path(filename).name)
    temp_file.write_bytes(bytes_)

    parsed_data_pyresparser = {}
    try:
        if nlp is not None:
            try:
                parser = ResumeParser(str(temp_file), custom_nlp=nlp)
                parsed_data_pyresparser = parser.get_extracted_data() or {}
            except TypeError:
                parser = ResumeParser(str(temp_file))
                parsed_data_pyresparser = parser.get_extracted_data() or {}
        else:
            parser = ResumeParser(str(temp_file))
            parsed_data_pyresparser = parser.get_extracted_data() or {}
    except Exception as e:
        logger.error(f"Error parsing resume '{filename}': {e}")
        parsed_data_pyresparser = {}
    finally:
        temp_file.unlink(missing_ok=True)

    full_text = text_extractor.extract_text_from_resume(bytes_, filename) or ""

    # --------------------------
    # Phone
    # --------------------------
    pyresparser_phone = parsed_data_pyresparser.get("mobile_number", "")
    custom_phone = extract_phone_number(full_text)
    final_phone_cleaned = normalize_phone(pyresparser_phone or custom_phone)

    # --------------------------
    # Email
    # --------------------------
    pyresparser_email = parsed_data_pyresparser.get("email", "")
    custom_email = extract_email_address(full_text)
    final_email_cleaned = re.sub(r'[^\w@.+-]', '', pyresparser_email or custom_email or "")

    # --------------------------
    # Name Extraction
    # --------------------------
    extracted_name = ""

    # Priority 1: spaCy top 10 lines
    if nlp:
        extracted_name = extract_name_with_spacy(full_text)

    # Priority 2: pyresparser, filtered
    if not extracted_name:
        name_pyresparser = parsed_data_pyresparser.get("name", "").strip()
        if name_pyresparser and len(name_pyresparser.split()) >= 2 and not any(char.isdigit() for char in name_pyresparser):
            extracted_name = name_pyresparser

    # Priority 3: email-based fallback
    if not extracted_name:
        extracted_name = extract_name_from_email(final_email_cleaned)

    final_name = extracted_name.replace('\n', ' ').strip() if extracted_name else ""

    return {
        "name": final_name,
        "email_cv": final_email_cleaned,
        "phone": final_phone_cleaned,
        "skills": parsed_data_pyresparser.get("skills", []),
        "education": parsed_data_pyresparser.get("education", []),
        "experience": parsed_data_pyresparser.get("experience", []),
        "total_experience": parsed_data_pyresparser.get("total_experience", 0.0),
        "full_text_content": full_text,
    }

