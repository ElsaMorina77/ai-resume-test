import docx2txt
from pdfminer.high_level import extract_text as extract_pdf_text
from pathlib import Path
import logging
import re # <-- New import for cleanup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_resume(bytes_: bytes, filename: str) -> str:
    """
    Extracts text from a resume file (PDF, DOC, DOCX).
    Handles temporary file creation and deletion for extraction and cleans up whitespace.
    """
    temp_file = Path("_tmp_text_extract" + Path(filename).suffix)
    temp_file.write_bytes(bytes_)
    full_text_content = ""

    try:
        ext = temp_file.suffix.lower()
        if ext == ".pdf":
            try:
                full_text_content = extract_pdf_text(str(temp_file))
            except Exception as pdf_e:
                logging.warning(f"Could not extract text from PDF '{filename}': {pdf_e}")
                full_text_content = ""
        elif ext == ".doc" or ext == ".docx":
            try:
                # docx2txt returns a clean string
                full_text_content = docx2txt.process(str(temp_file))
            except Exception as docx_e:
                logging.warning(f"Could not extract text from DOCX '{filename}': {docx_e}")
                full_text_content = ""
        else:
            logging.warning(f"Unsupported resume file type for text extraction: {filename}")
            full_text_content = ""
    finally:
        temp_file.unlink(missing_ok=True)

    # Aggressively normalize whitespace before returningg
    if full_text_content:
        # Replace multiple newlines with a standard double newline (paragraph break)
        full_text_content = re.sub(r'\n\s*\n', '\n\n', full_text_content)
        # Replace other multiple whitespaces (including tabs) with a single space
        full_text_content = re.sub(r'[ \t]+', ' ', full_text_content)
        # Strip leading/trailing whitespace
        full_text_content = full_text_content.strip()

    return full_text_content