from pathlib import Path

import pdfplumber

from exam_topic_predictor.text_utils import normalize_text


def extract_text_from_pdf(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    text = _extract_with_pdfplumber(path)
    if text:
        return text

    text = _extract_with_ocr(path)
    if text:
        return text

    raise ValueError(
        f"No readable text found in: {path}. This PDF may be image-only and OCR could not extract usable text."
    )


def _extract_with_pdfplumber(path: Path) -> str:
    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return normalize_text("\n".join(pages))


def _extract_with_ocr(path: Path) -> str:
    try:
        import numpy as np
        import pypdfium2 as pdfium
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return ""

    ocr = RapidOCR()
    pdf = pdfium.PdfDocument(str(path))
    pages: list[str] = []

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        bitmap = page.render(scale=2)
        image = bitmap.to_pil()
        result, _ = ocr(np.array(image))
        page_text = "\n".join(line[1] for line in result if len(line) > 1) if result else ""
        pages.append(page_text)
        bitmap.close()
        page.close()

    return normalize_text("\n".join(pages))
