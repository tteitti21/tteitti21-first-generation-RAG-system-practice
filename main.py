from openai import OpenAI
import numpy as np
import os
import re
from colorama import Fore, Style, init
from util.env_utils import get_env_path, load_env_file
from util.file_utils import json_file_has_content, load_json, save_json
from util.math_utils import cosine_similarity
from util.pdf_utils import load_pdf_text

init(autoreset=True)  # Automatically resets style after every print
client = OpenAI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
CHUNK_SIZE = 1000
TOP_K = 3


ENV_VALUES = load_env_file(ENV_PATH)

PDF_PATH = get_env_path(ENV_VALUES, "PDF_PATH", BASE_DIR)
CHUNKS_PATH = get_env_path(ENV_VALUES, "CHUNKS_PATH", BASE_DIR)
EMBEDDINGS_PATH = get_env_path(ENV_VALUES, "EMBEDDINGS_PATH", BASE_DIR)
RELEVANT_CHUNKS_PATH = get_env_path(ENV_VALUES, "RELEVANT_CHUNKS_PATH", BASE_DIR)


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


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

# Loads existing chunks or recreates them if they are too large 
def load_or_create_documents():
    if json_file_has_content(CHUNKS_PATH):
        print(f"Found stored chunks in {CHUNKS_PATH}...")
        documents = load_json(CHUNKS_PATH)

        if all(len(doc) <= CHUNK_SIZE for doc in documents):
            return documents, False

        print(f"{Fore.RED}Stored chunks are too large. Recreating chunks and embeddings...")

    print("Loading PDF...")
    pdf_text = load_pdf_text(PDF_PATH)

    print("Creating chunks...")
    documents = chunk_text(
        split_text_units(pdf_text),
        CHUNK_SIZE,
    )

    print(f"Storing chunks in {CHUNKS_PATH}...")
    save_json(CHUNKS_PATH, documents)

    return documents, True

# Loads existing embeddings if new chunks were created, or if their count doesn't match
def load_or_create_embeddings(documents, force_recreate=False):
    if json_file_has_content(EMBEDDINGS_PATH) and not force_recreate:
        print(f"Found stored embeddings in {EMBEDDINGS_PATH}...")
        doc_embeddings = load_json(EMBEDDINGS_PATH)

        if len(doc_embeddings) == len(documents):
            return doc_embeddings

        print(f"{Fore.RED}Stored embeddings do not match stored chunks. Recreating embeddings...")

    print("Creating embeddings...")
    doc_embeddings = [get_embedding(doc) for doc in documents]

    print(f"Storing embeddings in {EMBEDDINGS_PATH}...")
    save_json(EMBEDDINGS_PATH, doc_embeddings)

    return doc_embeddings

# Store current chunks, add them to history and return history
# for more performant future reference.
def store_relevant_chunks(question, documents, top_indices, scores):
    relevant_chunks = [documents[i] for i in top_indices]

    retrieval = {
        "question": question,
        "chunks": relevant_chunks,
        "matches": [
            {
                "chunk_index": int(idx),
                "score": float(scores[idx]),
                "chunk": documents[idx]
            }
            for idx in top_indices
        ]
    }

    history = []
    if json_file_has_content(RELEVANT_CHUNKS_PATH):
        history = load_json(RELEVANT_CHUNKS_PATH)

    history.append(retrieval)
    save_json(RELEVANT_CHUNKS_PATH, history)

    return relevant_chunks

# -------------------- Main program ----------------
def main():
    print(f"{Fore.CYAN}" + "_" * 50)

    documents, chunks_recreated = load_or_create_documents()
    print(f"Loaded {len(documents)} chunks")

    doc_embeddings = np.array(
        load_or_create_embeddings(
            documents,
            force_recreate=chunks_recreated
        )
    )

    print(f"{Fore.GREEN}Embeddings ready")

    while True:

        question = input(f"{Fore.LIGHTYELLOW_EX}\nQuestion: {Style.RESET_ALL}")
        if question.lower() in [ "exit","quit"]:
            break

        question_embedding = np.array(
            get_embedding(question)
        )

        scores = [
            cosine_similarity(
                question_embedding,
                doc_embedding
            )
            for doc_embedding in doc_embeddings
        ]

        top_indices = (np.argsort(scores)[-TOP_K:][::-1])

        relevant_chunks = store_relevant_chunks(
            question,
            documents,
            top_indices,
            scores
        )

        print(f"{Fore.LIGHTMAGENTA_EX}\nRetrieved chunks:")
        # Inspect retrieved chunks
        for rank, idx in enumerate(top_indices, start=1):
            print(f"\n--- Rank {rank} ---")
            print(f"Score: {scores[idx]:.4f}")
            print(documents[idx][:500])

        context = "\n\n".join(relevant_chunks)

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": """ You must answer only using the provided context.

                    If the answer cannot be found in the context, say so."""
                },
                {
                    "role": "user",
                    "content": f"""
            Context: 
            {context}

            Question:
            {question}
            """
                }
            ]
        )

        print(f"{Fore.GREEN}\nAnswer:\n")
        print(response.choices[0].message.content)
        print(f"{Fore.CYAN}" + "_" * 50)


if __name__ == "__main__":
    main()
