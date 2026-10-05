from pathlib import Path
from typing import List, Dict, Any
import pymupdf


def get_page_count(file_path: Path) -> int:
    """
    Get the total number of pages in a PDF document.
    """
    with pymupdf.open(file_path) as doc:
        return len(doc)


def parse_headings(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract headings or table of contents from a PDF.
    Attempts to read bookmarks/TOC first; falls back to font-size heuristics.
    
    Returns:
        List of dicts with 'level', 'title', 'page', and 'page_number'.
    """
    headings: List[Dict[str, Any]] = []

    try:
        with pymupdf.open(file_path) as doc:
            # 1. Try extracting standard PDF Table of Contents (bookmarks)
            toc = doc.get_toc(simple=True)
            if toc:
                for item in toc:
                    lvl, title, page = item[0], str(item[1]).strip(), int(item[2])
                    if title:
                        headings.append({
                            "level": int(lvl),
                            "title": title,
                            "page": page,
                            "page_number": page
                        })
                if headings:
                    return headings

            # 2. Fallback heuristic: Detect headings by font-size disparity
            font_sizes: List[float] = []
            for page in doc:
                page_dict = page.get_text("dict")
                for block in page_dict.get("blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                font_sizes.append(round(float(span.get("size", 10.0)), 1))

            if not font_sizes:
                return []

            # Find predominant body font size (mode)
            body_size = max(set(font_sizes), key=font_sizes.count)

            for page_idx, page in enumerate(doc):
                page_number = page_idx + 1
                page_dict = page.get_text("dict")
                for block in page_dict.get("blocks", []):
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        if not spans:
                            continue
                        line_text = " ".join(s.get("text", "").strip() for s in spans if s.get("text", "").strip())
                        if not line_text:
                            continue
                        max_size = max(float(s.get("size", 10.0)) for s in spans)

                        # Heading criteria: significantly larger than body text and concise
                        if max_size >= body_size + 2.0 and 2 <= len(line_text) <= 120:
                            level = 1 if max_size >= body_size + 4.0 else 2
                            headings.append({
                                "level": level,
                                "title": line_text,
                                "page": page_number,
                                "page_number": page_number
                            })
    except Exception:
        return []

    return headings


def search_in_pdf(file_path: Path, keyword: str) -> List[Dict[str, Any]]:
    """
    Search for a keyword across all pages of a PDF document (case-insensitive).
    
    Returns:
        List of dicts with 'page', 'page_number', 'match_count', and 'snippet'.
    """
    if not keyword or not keyword.strip():
        return []

    keyword_clean = keyword.strip()
    keyword_lower = keyword_clean.lower()
    matches: List[Dict[str, Any]] = []

    try:
        with pymupdf.open(file_path) as doc:
            for page_idx, page in enumerate(doc):
                page_number = page_idx + 1
                page_text = page.get_text("text")
                page_lower = page_text.lower()

                count = page_lower.count(keyword_lower)
                if count > 0:
                    first_pos = page_lower.find(keyword_lower)
                    start = max(0, first_pos - 60)
                    end = min(len(page_text), first_pos + len(keyword_clean) + 60)
                    snippet = page_text[start:end].replace("\n", " ").strip()

                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(page_text):
                        snippet = snippet + "..."

                    matches.append({
                        "page": page_number,
                        "page_number": page_number,
                        "match_count": count,
                        "snippet": snippet
                    })
    except Exception:
        return []

    return matches



def read_page(file_path: Path, page_number: int) -> Dict[str, Any]:
    """
    Extract text from a specific 1-indexed page.
    
    Raises:
        ValueError: If page_number is out of bounds (< 1 or > total pages).
    """
    with pymupdf.open(file_path) as doc:
        total_pages = len(doc)
        if page_number < 1 or page_number > total_pages:
            raise ValueError(
                f"Page number {page_number} is out of range. Document has {total_pages} page(s)."
            )

        page = doc[page_number - 1]
        text = page.get_text("text").strip()

        return {
            "page_number": page_number,
            "page": page_number,
            "total_pages": total_pages,
            "text": text
        }
