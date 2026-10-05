from typing import List, Dict, Any
from app.tools.document_store import list_stored_documents, get_document_path
from app.pdf.parser import get_page_count


def list_documents() -> List[Dict[str, Any]]:
    """
    List all stored documents and their total page counts.
    Does not expose raw filesystem paths to prevent direct document text access.
    
    Returns:
        List of dicts: [{'doc_id': str, 'total_pages': int}]
    """
    stored = list_stored_documents()
    documents: List[Dict[str, Any]] = []

    for item in stored:
        doc_id = item["doc_id"]
        try:
            file_path = get_document_path(doc_id)
            pages = get_page_count(file_path)
        except Exception:
            pages = 0

        documents.append({
            "doc_id": doc_id,
            "total_pages": pages
        })

    return documents
