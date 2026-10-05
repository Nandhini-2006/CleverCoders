from pathlib import Path
import uuid

from app.config import DOCUMENT_DIR


def save_document(uploaded_file):
    """
    Save an uploaded PDF using a generated document ID.

    Returns:
        doc_id: unique identifier
        file_path: saved PDF path
    """
    # Safe file handling: only accept PDF files
    if not uploaded_file.name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    # Generate unique document ID
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"

    # Always use our own filename.
    # Never trust the user's original filename as a filesystem path.
    file_path = DOCUMENT_DIR / f"{doc_id}.pdf"

    # Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return doc_id, file_path


def get_document_path(doc_id: str) -> Path:
    """
    Return the PDF path corresponding to a document ID.
    """
    # Prevent path traversal: ensure ID matches expected pattern
    if not doc_id.startswith("doc_") or not doc_id.replace("doc_", "").isalnum():
        raise ValueError("Invalid document ID")

    file_path = (DOCUMENT_DIR / f"{doc_id}.pdf").resolve()

    # Ensure path stays strictly within DOCUMENT_DIR
    if file_path.parent != DOCUMENT_DIR.resolve():
        raise ValueError("Invalid document path: path traversal detected")

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {doc_id}")

    return file_path


def list_stored_documents():
    """
    Return all stored PDF document IDs.
    """

    documents = []

    for file_path in DOCUMENT_DIR.glob("doc_*.pdf"):
        doc_id = file_path.stem

        documents.append({
            "doc_id": doc_id,
            "file_path": str(file_path)
        })

    return documents