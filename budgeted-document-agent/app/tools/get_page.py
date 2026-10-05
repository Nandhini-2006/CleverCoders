from typing import Any
from app.tools.document_store import get_document_path
from app.pdf.parser import read_page


class PageResult(dict):
    """
    Structured representation of a retrieved page.
    Inherits from dict for serialization and compatibility, while allowing string operations.
    """
    def __init__(self, doc_id: str, page_number: int, total_pages: int, text: str):
        super().__init__(
            doc_id=doc_id,
            page=page_number,
            page_number=page_number,
            total_pages=total_pages,
            text=text
        )

    @property
    def text(self) -> str:
        return self["text"]

    @property
    def page_number(self) -> int:
        return self["page_number"]

    @property
    def doc_id(self) -> str:
        return self["doc_id"]

    @property
    def total_pages(self) -> int:
        return self["total_pages"]

    def __str__(self) -> str:
        return self["text"]

    def __contains__(self, item: Any) -> bool:
        if super().__contains__(item):
            return True
        if isinstance(item, str) and item in self.get("text", ""):
            return True
        return False


def get_page(doc_id: str, page_number: int) -> PageResult:
    """
    Retrieve the text content and metadata for a specific 1-indexed page.
    
    Args:
        doc_id: The unique document identifier (e.g., 'doc_a81f92c4').
        page_number: 1-indexed page number.
        
    Returns:
        PageResult: Dict-like object with 'doc_id', 'page_number', 'total_pages', and 'text'.
    """
    file_path = get_document_path(doc_id)
    info = read_page(file_path, page_number)
    return PageResult(
        doc_id=doc_id,
        page_number=info["page_number"],
        total_pages=info["total_pages"],
        text=info["text"]
    )
