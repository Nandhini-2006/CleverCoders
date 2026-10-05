from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Folder where uploaded PDFs are stored
DOCUMENT_DIR = BASE_DIR / "documents"

# Create folder if it doesn't exist
DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)

# Maximum number of tool calls allowed per question
MAX_TOOL_CALLS = 6 #cost function