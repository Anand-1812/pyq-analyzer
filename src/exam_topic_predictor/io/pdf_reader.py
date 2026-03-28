from pathlib import Path

import pdfplumber

from exam_topic_predictor.text_utils import normalize_text


def extract_text_from_pdf(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")

    text = normalize_text("\n".join(pages))
    if not text:
        raise ValueError(f"No readable text found in: {path}")

    return text
