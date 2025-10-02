# src/bot_logic/job_matcher.py

import logging
import re
from typing import Optional, List, Tuple

# Import the config from the top-level directory
from config import config
# Import the nlp object (spaCy model) directly from data_parser
from src.resume_parser.data_parser import nlp

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Keep at DEBUG to see detailed matching steps.

# --- Pre-processing for efficiency ---
# Pre-process job keywords into spaCy Doc objects for efficient semantic similarity calculation.
# We use the main job titles (keys) from JOB_KEYWORDS_MAPPING for semantic comparison.
JOB_TITLE_DOCS = {}
if nlp is not None and hasattr(nlp, 'vocab') and hasattr(nlp.vocab, 'vectors') and nlp.vocab.vectors.shape[0] > 0:
    for main_title in config.JOB_KEYWORDS_MAPPING.keys():
        JOB_TITLE_DOCS[main_title] = nlp(main_title.lower())
    logger.info("Pre-processed main job titles into spaCy Doc objects for semantic similarity.")
else:
    logger.warning("SpaCy model does not have vectors loaded or vectors object is not accessible. Semantic similarity for job titles will be disabled.")


def _find_best_match_in_text(text_content: str, source_name: str) -> Optional[str]:
    """
    Internal helper to find a job title using direct keyword/regex matching in a given text.
    Iterates through JOB_KEYWORDS_MAPPING, respecting its order for priority.
    """
    text_lower = text_content.lower()
    for main_job_title, associated_keywords in config.JOB_KEYWORDS_MAPPING.items():
        for keyword in associated_keywords:
            # --- IMPORTANT FIX START ---
            # If the keyword contains multiple words, re.escape will escape spaces as '\ '.
            # We want to match any whitespace for multi-word phrases.
            # Convert escaped spaces back to \s+ (one or more whitespace chars)
            # and ensure word boundaries are only around the *whole* phrase if it's multi-word.
            
            # Escape the entire keyword first
            escaped_keyword_lower = re.escape(keyword.lower())
            
            if ' ' in keyword: # It's a multi-word keyword
                # Replace escaped spaces with \s+ (one or more whitespace characters)
                regex_pattern_string = r'\b' + escaped_keyword_lower.replace(r'\ ', r'\s+') + r'\b'
            else: # It's a single-word keyword
                regex_pattern_string = r'\b' + escaped_keyword_lower + r'\b'

            # Define the patterns using the correctly formed regex_pattern_string
            patterns = [
                re.compile(regex_pattern_string), # Main exact or multi-word phrase match
                re.compile(r'(?:application|resume|interested)\s*(?:for|in)?\s*(?:a|an|the)?\s*' + regex_pattern_string + r'\s*(?:role|position|job)\b'),
                re.compile(regex_pattern_string + r'\s*application\b'),
            ]
            # --- IMPORTANT FIX END ---

            for pattern in patterns:
                if pattern.search(text_lower):
                    logger.info(f"Inferred job title '{main_job_title}' by direct match (keyword: '{keyword}') from {source_name}.")
                    return main_job_title
    return None


def _find_best_semantic_match(text_doc, source_name: str, similarity_threshold: float) -> Optional[Tuple[str, float]]:
    """
    Internal helper to find the best job title using semantic similarity in a given spaCy Doc.
    Returns (job_title, similarity_score) if a match is found above the threshold.
    """
    if nlp is None or not text_doc.has_vector:
        logger.debug(f"Semantic similarity skipped for {source_name}: Doc has no vector or NLP model not loaded.")
        return None

    current_best_candidate_title = None
    current_highest_candidate_similarity = 0.0

    for main_title, job_title_doc in JOB_TITLE_DOCS.items():
        if not job_title_doc.has_vector:
            logger.debug(f"Skipping semantic similarity for keyword '{main_title}': Doc has no vector.")
            continue

        similarity = text_doc.similarity(job_title_doc)
        logger.debug(f"  Semantic Sim. between '{source_name}' and '{main_title}': {similarity:.2f}")

        if similarity > current_highest_candidate_similarity and similarity >= similarity_threshold:
            current_highest_candidate_similarity = similarity
            current_best_candidate_title = main_title

    if current_best_candidate_title:
        logger.info(f"Inferred job title '{current_best_candidate_title}' by semantic similarity (score: {current_highest_candidate_similarity:.2f}) from {source_name}.")
        return (current_best_candidate_title, current_highest_candidate_similarity)
    
    return None


def infer_job_title(subject_line: str, email_body_content: str, full_cv_text_content: str) -> str:
    """
    Infers the job title applied for, using a tiered approach:
    1. Direct keyword/regex matching (prioritizing specific roles first based on JOB_KEYWORDS_MAPPING order).
    2. Semantic similarity matching (if direct match fails).
    """
    logger.debug(f"Attempting to infer job title for subject: '{subject_line[:100]}...'")

    # Limit CV text for efficiency and relevance
    cv_text_lower_snippet = full_cv_text_content[:2000].lower()

    # --- Phase 1: Direct Keyword/Regex Matching ---
    # Prioritized by order in JOB_KEYWORDS_MAPPING
    
    # 1. Check Subject Line
    inferred_title = _find_best_match_in_text(subject_line, "Email Subject (Direct)")
    if inferred_title:
        return inferred_title

    # 2. Check Email Body Content
    inferred_title = _find_best_match_in_text(email_body_content, "Email Body (Direct)")
    if inferred_title:
        return inferred_title

    # 3. Check CV Text Content Snippet
    inferred_title = _find_best_match_in_text(cv_text_lower_snippet, "CV Snippet (Direct)")
    if inferred_title:
        return inferred_title

    logger.debug("Direct keyword/regex match failed. Attempting semantic similarity.")

    # --- Phase 2: Semantic Similarity Matching (The "Smart" Part) ---
    # This phase only runs if Phase 1 didn't find a direct match.
    if nlp is None or not (hasattr(nlp, 'vocab') and hasattr(nlp.vocab, 'vectors') and nlp.vocab.vectors.shape[0] > 0):
        logger.warning("SpaCy model does not have vectors for semantic similarity. Returning 'Unspecified'.")
        return "Unspecified"

    SIMILARITY_THRESHOLD = 0.70 # Tune this threshold (e.g., 0.7 to 0.8) as needed

    email_subject_doc = nlp(subject_line) if nlp else None
    email_body_doc = nlp(email_body_content) if nlp else None
    cv_snippet_doc = nlp(full_cv_text_content[:2000]) if nlp else None

    best_semantic_match_info = (None, 0.0) # (job_title, similarity_score)

    # 1. Semantic Check Subject Line
    if email_subject_doc:
        match_info = _find_best_semantic_match(email_subject_doc, "Email Subject (Semantic)", SIMILARITY_THRESHOLD)
        if match_info and match_info[1] > best_semantic_match_info[1]:
            best_semantic_match_info = match_info
    
    # 2. Semantic Check Email Body Content
    if email_body_doc and (not best_semantic_match_info[0] or best_semantic_match_info[1] < SIMILARITY_THRESHOLD):
        match_info = _find_best_semantic_match(email_body_doc, "Email Body (Semantic)", SIMILARITY_THRESHOLD)
        if match_info and match_info[1] > best_semantic_match_info[1]:
            best_semantic_match_info = match_info

    # 3. Semantic Check CV Text Content Snippet
    if cv_snippet_doc and (not best_semantic_match_info[0] or best_semantic_match_info[1] < SIMILARITY_THRESHOLD):
        match_info = _find_best_semantic_match(cv_snippet_doc, "CV Snippet (Semantic)", SIMILARITY_THRESHOLD)
        if match_info and match_info[1] > best_semantic_match_info[1]:
            best_semantic_match_info = match_info

    if best_semantic_match_info[0]:
        return best_semantic_match_info[0]

    logger.info("Could not infer a specific job title using any method. Returning 'Unspecified'.")
    return "Unspecified"