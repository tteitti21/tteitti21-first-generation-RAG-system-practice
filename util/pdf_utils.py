from pypdf import PdfReader


def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)

    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            full_text += page_text + "\n"

    return full_text
