from openai import OpenAI
import numpy as np
import os
from colorama import Fore, Style, init
from util.chunk_utils import create_page_chunks
from util.document_utils import (
    format_document_for_context,
    get_document_text,
    is_valid_document_cache
)
from util.env_utils import get_env_path, load_env_file
from util.faiss_utils import load_or_create_faiss_index, normalize_query_embedding
from util.file_utils import json_file_has_content, load_json, save_json
from util.output_utils import print_search_results
from util.pdf_utils import load_pdf_pages

init(autoreset=True)  # Automatically resets style after every print
client = OpenAI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
CHUNK_SIZE = 1000
TOP_K = 3
MIN_SIMILARITY = 0.35
MAX_HISTORY = 100
RECENT_CHAT_TURNS = 3


ENV_VALUES = load_env_file(ENV_PATH)
FAISS_INDEX_TYPE = os.environ.get("FAISS_INDEX_TYPE") or ENV_VALUES.get(
    "FAISS_INDEX_TYPE",
    "flat"
)

PDF_PATH = get_env_path(ENV_VALUES, "PDF_PATH", BASE_DIR)
CHUNKS_PATH = get_env_path(ENV_VALUES, "CHUNKS_PATH", BASE_DIR)
EMBEDDINGS_PATH = get_env_path(ENV_VALUES, "EMBEDDINGS_PATH", BASE_DIR)
RELEVANT_CHUNKS_PATH = get_env_path(ENV_VALUES, "RELEVANT_CHUNKS_PATH", BASE_DIR)
FAISS_INDEX_PATH = get_env_path(ENV_VALUES, "FAISS_INDEX_PATH", BASE_DIR)

FAISS_INDEX_PATH = FAISS_INDEX_PATH.replace(
    ".index",
    f"_{FAISS_INDEX_TYPE}.index"
)

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# Loads existing chunks or recreates them if they are too large 
def load_or_create_documents():
    if json_file_has_content(CHUNKS_PATH):
        print(f"Found stored chunks in {CHUNKS_PATH}...")
        documents = load_json(CHUNKS_PATH)

        if is_valid_document_cache(documents, CHUNK_SIZE):
            return documents, False

        print(
            f"{Fore.RED}Stored chunks are old or too large. "
            "Recreating chunks and embeddings..."
        )

    print("Loading PDF...")
    pdf_pages = load_pdf_pages(PDF_PATH)

    print("Creating chunks...")
    documents = create_page_chunks(pdf_pages, CHUNK_SIZE)

    print(f"Storing chunks in {CHUNKS_PATH}...")
    save_json(CHUNKS_PATH, documents)

    return documents, True

# Loads existing embeddings if new chunks were created, or if their count doesn't match
def load_or_create_embeddings(documents, force_recreate=False):
    if json_file_has_content(EMBEDDINGS_PATH) and not force_recreate:
        print(f"Found stored embeddings in {EMBEDDINGS_PATH}...")
        doc_embeddings = load_json(EMBEDDINGS_PATH)

        if len(doc_embeddings) == len(documents):
            return doc_embeddings, False

        print(f"{Fore.RED}Stored embeddings do not match stored chunks. Recreating embeddings...")

    print("Creating embeddings...")
    doc_embeddings = [get_embedding(get_document_text(doc)) for doc in documents]

    print(f"Storing embeddings in {EMBEDDINGS_PATH}...")
    save_json(EMBEDDINGS_PATH, doc_embeddings)

    return doc_embeddings, True

# Store current chunks, add them to history and return history
# for more performant future reference.
def store_relevant_chunks(question, retrieval_query, documents, top_indices, top_scores):
    relevant_chunks = [documents[int(i)] for i in top_indices]

    retrieval = {
        "question": question,
        "retrieval_query": retrieval_query,
        "chunks": relevant_chunks,
        "matches": [
            {
                "chunk_index": int(idx),
                "score": float(score),
                "page": documents[int(idx)]["page"],
                "chunk": get_document_text(documents[int(idx)])
            }
            for idx, score in zip(top_indices, top_scores)
        ]
    }

    history = []
    if json_file_has_content(RELEVANT_CHUNKS_PATH):
        history = load_json(RELEVANT_CHUNKS_PATH)

    history.append(retrieval)
    history = history[-MAX_HISTORY:]
    save_json(RELEVANT_CHUNKS_PATH, history)

    return relevant_chunks


def build_retrieval_query(question, chat_history):
    recent_history = chat_history[-RECENT_CHAT_TURNS:]

    if not recent_history:
        return question

    history_text = "\n".join(
        f"User: {turn['question']}\nAssistant: {turn['answer']}"
        for turn in recent_history
    )

    return f"""
        Recent conversation:
        {history_text}

        Current question:
        {question}
        """.strip()

# -------------------- Main program ----------------
def main():
    print(f"{Fore.CYAN}" + "_" * 50)

    documents, chunks_recreated = load_or_create_documents()
    print(f"Loaded {len(documents)} chunks")

    doc_embeddings, embeddings_recreated = load_or_create_embeddings(
        documents,
        force_recreate=chunks_recreated
    )
    doc_embeddings = np.array(
        doc_embeddings,
        dtype="float32"
    )

    faiss_index = load_or_create_faiss_index(
        doc_embeddings,
        len(documents),
        FAISS_INDEX_PATH,
        EMBEDDINGS_PATH,
        index_type=FAISS_INDEX_TYPE,
        force_recreate=embeddings_recreated
    )
    # Ask for TOP_K results, unless there are fewer documents than TOP_K
    search_limit = min(TOP_K, len(documents))

    print(f"{Fore.GREEN}Embeddings ready")

    chat_history = []

    while True:

        question = input(f"{Fore.LIGHTYELLOW_EX}\nQuestion: {Style.RESET_ALL}")
        if question.lower() in [ "exit","quit"]:
            break

        retrieval_query = build_retrieval_query(question, chat_history)

        question_embedding = np.array(
            [get_embedding(retrieval_query)],
            dtype="float32"
        )

        question_embedding = normalize_query_embedding(question_embedding)

        result_scores, result_indices = faiss_index.search(
            question_embedding,
            search_limit
        )

        result_scores = result_scores[0]
        result_indices = result_indices[0]

        accepted_positions = np.where(result_scores >= MIN_SIMILARITY)[0]

        if len(accepted_positions) == 0:
            print(
                f"{Fore.YELLOW}\nNo chunks were similar enough. "
                "Could you elaborate or ask more specifically?"
            )
            print(f"{Fore.CYAN}" + "_" * 50)
            continue

        top_indices = result_indices[accepted_positions]
        top_scores = result_scores[accepted_positions]

        relevant_chunks = store_relevant_chunks(
            question,
            retrieval_query,
            documents,
            top_indices,
            top_scores
        )

        print(f"{Fore.LIGHTMAGENTA_EX}\nRetrieved chunks:")
        print_search_results(documents, top_indices, top_scores)

        context = "\n\n".join(
            format_document_for_context(chunk)
            for chunk in relevant_chunks
        )

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                "role": "system",
                "content": """
                    Answer using only the provided context.
                    You may make reasonable inferences from the context.
                    Do not introduce information that is not supported by the context.

                    If the answer cannot be determined from the context, say:
                    'I cannot find that information in the provided documents.'
                """
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

        answer = response.choices[0].message.content

        chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )
        chat_history = chat_history[-RECENT_CHAT_TURNS:]

        print(f"{Fore.GREEN}\nAnswer:\n")
        print(answer)
        print(f"{Fore.CYAN}" + "_" * 50)


if __name__ == "__main__":
    main()
