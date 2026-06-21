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


def add_document_citation_metadata(document, chunk_index, score=None):
    cited_document = dict(document)
    cited_document["chunk_index"] = int(chunk_index)

    if score is not None:
        cited_document["score"] = float(score)

    return cited_document


def format_document_for_context(document):
    chunk_index = document.get("chunk_index", "unknown")

    return f"[page {document['page']}, chunk {chunk_index}]\n{document['text']}"
