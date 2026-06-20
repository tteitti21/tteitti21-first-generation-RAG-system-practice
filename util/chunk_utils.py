import re


def split_text_units(text):
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    if len(paragraphs) > 1:
        return paragraphs

    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def split_oversized_unit(text: str, max_chars: int):
    # Prefer sentence boundaries when a paragraph-like unit is too large.
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]

    # Some text, like a table of contents, may not contain sentence punctuation.
    # In that case, split by words so chunks do not start with half a word.
    if len(sentences) <= 1:
        return split_text_on_words(text, max_chars)

    return chunk_text(sentences, max_chars)


def split_text_on_words(text: str, max_chars: int):
    chunks = []
    current_chunk = ""

    for word in text.split():
        # This should be rare, but protects against a single huge token.
        if len(word) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            chunks.extend(
                word[start:start + max_chars]
                for start in range(0, len(word), max_chars)
            )
            continue

        # Try adding the next complete word to the current chunk.
        candidate = f"{current_chunk} {word}" if current_chunk else word

        if len(candidate) <= max_chars:
            current_chunk = candidate
        else:
            # If the word does not fit, finish the current chunk and start a new one.
            chunks.append(current_chunk)
            current_chunk = word

    # Store the final in-progress word chunk.
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_text(units: list[str], max_chars: int):
    chunks = []
    current_chunk = ""

    for unit in units:
        # A single unit can be larger than the chunk limit, so split it before
        # continuing with normal chunk packing.
        if len(unit) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            chunks.extend(split_oversized_unit(unit, max_chars))
            continue

        # Try adding the next complete unit to the current chunk.
        candidate = f"{current_chunk}\n\n{unit}" if current_chunk else unit

        if len(candidate) <= max_chars:
            current_chunk = candidate
        else:
            # If it no longer fits, store the finished chunk and start a new one.
            chunks.append(current_chunk)
            current_chunk = unit

    # Store the final in-progress chunk after all units have been handled.
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def create_page_chunks(pdf_pages, chunk_size):
    documents = []

    for page in pdf_pages:
        page_chunks = chunk_text(
            split_text_units(page["text"]),
            chunk_size,
        )

        for chunk in page_chunks:
            documents.append(
                {
                    "page": page["page"],
                    "text": chunk
                }
            )

    return documents
