from pathlib import Path
from typing import Optional

import fitz


def load_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text content as a single string
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    text_parts = []

    for page in doc:
        text_parts.append(page.get_text())

    doc.close()
    return "\n\n".join(text_parts)


def load_pdf_with_metadata(pdf_path: str) -> dict:
    """Extract text and metadata from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with keys: text, title, author, pages, metadata
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())

    metadata = doc.metadata
    result = {
        "text": "\n\n".join(text_parts),
        "title": metadata.get("title", path.stem),
        "author": metadata.get("author", ""),
        "pages": len(doc),
        "source": str(path),
        "metadata": metadata,
    }

    doc.close()
    return result


def load_pdf_by_pages(pdf_path: str) -> list[dict]:
    """Extract text from PDF, returning content per page.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts with page_num and content for each page
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        pages.append({"page_num": i + 1, "content": page.get_text()})

    doc.close()
    return pages
