from openai import OpenAI
from pypdf import PdfReader
import numpy as np
import json
import os
import re
from colorama import Fore, Style, init

init(autoreset=True)  # Automatically resets style after every print
client = OpenAI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
CHUNK_SIZE = 1000
TOP_K = 3


def load_env_file(file_path):
    env_values = {}

    if not os.path.exists(file_path):
        return env_values

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            env_values[key.strip()] = value.strip().strip("\"'")

    return env_values


def get_env_path(env_values, key):
    value = os.environ.get(key) or env_values.get(key)

    if not value:
        raise ValueError(f"Missing required path in .env: {key}")

    if os.path.isabs(value):
        return value

    return os.path.join(BASE_DIR, value)


ENV_VALUES = load_env_file(ENV_PATH)

PDF_PATH = get_env_path(ENV_VALUES, "PDF_PATH")
CHUNKS_PATH = get_env_path(ENV_VALUES, "CHUNKS_PATH")
EMBEDDINGS_PATH = get_env_path(ENV_VALUES, "EMBEDDINGS_PATH")
RELEVANT_CHUNKS_PATH = get_env_path(ENV_VALUES, "RELEVANT_CHUNKS_PATH")


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

# IS FINE
def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)

    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            full_text += page_text + "\n"

    return full_text


def json_file_has_content(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(file_path, data):
    parent_dir = os.path.dirname(file_path)

    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def split_oversized_unit(text, max_chars):
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]

    if len(sentences) <= 1:
        return [
            text[start:start + max_chars].strip()
            for start in range(0, len(text), max_chars)
            if text[start:start + max_chars].strip()
        ]

    return chunk_text(sentences, max_chars)


def chunk_text(units, max_chars):
    chunks = []
    current_chunk = ""

    for unit in units:
        if len(unit) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            chunks.extend(split_oversized_unit(unit, max_chars))
            continue

        candidate = f"{current_chunk}\n\n{unit}" if current_chunk else unit

        if len(candidate) <= max_chars:
            current_chunk = candidate
        else:
            chunks.append(current_chunk)
            current_chunk = unit

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def load_or_create_documents():
    if json_file_has_content(CHUNKS_PATH):
        print(f"Found stored chunks in {CHUNKS_PATH}...")
        documents = load_json(CHUNKS_PATH)

        if all(len(doc) <= CHUNK_SIZE for doc in documents):
            return documents, False

        print("Stored chunks are too large. Recreating chunks and embeddings...")

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

def load_or_create_embeddings(documents, force_recreate=False):
    if json_file_has_content(EMBEDDINGS_PATH) and not force_recreate:
        print(f"Found stored embeddings in {EMBEDDINGS_PATH}...")
        doc_embeddings = load_json(EMBEDDINGS_PATH)

        if len(doc_embeddings) == len(documents):
            return doc_embeddings

        print("Stored embeddings do not match stored chunks. Recreating embeddings...")

    print("Creating embeddings...")
    doc_embeddings = [get_embedding(doc) for doc in documents]

    print(f"Storing embeddings in {EMBEDDINGS_PATH}...")
    save_json(EMBEDDINGS_PATH, doc_embeddings)

    return doc_embeddings


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

        question = input(f"{Fore.LIGHTYELLOW_EX}\nQuestion: ")
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
        # Diagnose incase of poor retrieval
        for idx in top_indices:
            print(f"Score: {scores[idx]:.4f}")

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
        print("AI", response.choices[0].message.content)
        print(f"{Fore.CYAN}" + "_" * 50)


if __name__ == "__main__":
    main()
