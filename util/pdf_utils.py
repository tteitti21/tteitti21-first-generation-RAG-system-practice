from pypdf import PdfReader

def load_pdf_pages(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            pages.append(
                {
                    "page": page_number,
                    "text": page_text
                }
            )

    return pages
