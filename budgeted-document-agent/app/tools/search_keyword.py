from typing import List, Dict, Any
from app.tools.document_store import get_document_path
from app.pdf.parser import search_in_pdf


def search_keyword(doc_id: str, keyword: str) -> List[Dict[str, Any]]:
    """
    Search for a keyword across all pages of the document (case-insensitive).
    
    Args:
        doc_id: The unique document identifier (e.g., 'doc_a81f92c4').
        keyword: The word or phrase to locate.
        
    Returns:
        List of dicts containing 'page', 'page_number', 'match_count', and 'snippet'.
    """
    file_path = get_document_path(doc_id)
    return search_in_pdf(file_path, keyword)
