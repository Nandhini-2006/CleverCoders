from typing import List, Dict, Any
from app.tools.document_store import get_document_path
from app.pdf.parser import parse_headings


def list_headings(doc_id: str) -> List[Dict[str, Any]]:
    """
    List headings / table of contents for the specified document.
    
    Args:
        doc_id: The unique document identifier (e.g., 'doc_a81f92c4').
        
    Returns:
        List of dicts containing 'level', 'title', and 'page'.
    """
    file_path = get_document_path(doc_id)
    return parse_headings(file_path)
