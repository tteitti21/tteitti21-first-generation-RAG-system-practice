def is_valid_document_cache(documents, max_chars):
    return all(
        isinstance(doc, dict)
        and isinstance(doc.get("page"), int)
        and isinstance(doc.get("text"), str)
        and len(doc["text"]) <= max_chars
        for doc in documents
    )


def get_document_text(document):
    return document["text"]


def format_document_for_context(document):
    return f"[Page {document['page']}]\n{document['text']}"
