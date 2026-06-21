from openai import OpenAI
import numpy as np
import os
from colorama import Fore, Style, init
from util.bm25_utils import (
    build_bm25_index,
    combine_faiss_and_bm25_results,
    get_top_bm25_results
)
from util.chunk_utils import create_page_chunks
from util.document_utils import (
    format_document_for_context,
    get_document_text,
    is_valid_document_cache
)
from util.env_utils import get_env_bool, get_env_path, get_env_value, load_env_file
from util.faiss_utils import (
    build_comparison_indexes,
    compare_faiss_indexes,
    load_or_create_faiss_index,
    normalize_query_embedding,
    search_faiss_index
)
from util.file_utils import json_file_has_content, load_json, save_json
from util.output_utils import print_app_message, print_retrieval_debug
from util.pdf_utils import load_pdf_pages
from util.query_rewrite import rewrite_query_for_retrieval
from util.rerank_utils import rerank_candidate_chunks

init(autoreset=True)  # Automatically resets style after every print
client = OpenAI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
CHUNK_SIZE = 1000
TOP_K = 5
FETCH_K = 10
RERANK_K = 7
MIN_RETRIEVAL_SCORE = 0.45
MAX_HISTORY = 100
RECENT_CHAT_TURNS = 3


ENV_VALUES = load_env_file(ENV_PATH)
COMPARE_INDEXES = get_env_bool(ENV_VALUES, "COMPARE_INDEXES")
REVIEW_ALL_SCORES = get_env_bool(ENV_VALUES, "REVIEW_ALL_SCORES")
ENABLE_RERANKING = get_env_bool(ENV_VALUES, "ENABLE_RERANKING", True)
FAISS_INDEX_TYPE = get_env_value(ENV_VALUES, "FAISS_INDEX_TYPE", "flat")

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


# -------------------- Main program ----------------
def main():
    print_app_message("divider")

    documents, chunks_recreated = load_or_create_documents()
    print_app_message("chunks_loaded", len(documents))

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
    # Fetch more candidates than TOP_K, then keep the best accepted chunks.
    search_limit = min(FETCH_K, len(documents))
    comparison_indexes = build_comparison_indexes(doc_embeddings) if COMPARE_INDEXES else {}
    bm25_index = build_bm25_index(documents)

    print_app_message("embeddings_ready")

    chat_history = []

    while True:

        question = input(f"{Fore.LIGHTYELLOW_EX}\nQuestion: {Style.RESET_ALL}")
        if question.lower() in [ "exit","quit"]:
            break

        if not chat_history:
            retrieval_query = question
        else:
            retrieval_query = rewrite_query_for_retrieval(
                client,
                question,
                chat_history,
                RECENT_CHAT_TURNS
            )

        question_embedding = np.array(
            [get_embedding(retrieval_query)],
            dtype="float32"
        )

        question_embedding = normalize_query_embedding(question_embedding)

        result_scores, result_indices = search_faiss_index(
            faiss_index,
            question_embedding,
            search_limit
        )

        bm25_indices, bm25_scores = get_top_bm25_results(
            question,
            bm25_index,
            search_limit
        )

        combined_indices, combined_scores = combine_faiss_and_bm25_results(
            result_indices,
            result_scores,
            bm25_indices,
            bm25_scores
        )

        accepted_positions = np.where(combined_scores >= MIN_RETRIEVAL_SCORE)[0]

        if len(accepted_positions) == 0:
            print(
                f"{Fore.YELLOW}\nNo chunks were similar enough. "
                "Could you elaborate or ask more specifically?"
            )
            print_app_message("divider")
            continue

        accepted_positions = accepted_positions[
            np.argsort(combined_scores[accepted_positions])[::-1]
        ]
        candidate_indices = combined_indices[accepted_positions]
        candidate_scores = combined_scores[accepted_positions]
        rerank_indices = candidate_indices[:RERANK_K]
        rerank_scores = candidate_scores[:RERANK_K]

        if ENABLE_RERANKING:
            reranked_indices, reranked_scores = rerank_candidate_chunks(
                client,
                question,
                retrieval_query,
                documents,
                rerank_indices,
                rerank_scores
            )
        else:
            reranked_indices = rerank_indices
            reranked_scores = rerank_scores

        top_indices = np.array(reranked_indices[:TOP_K], dtype="int64")
        top_scores = np.array(reranked_scores[:TOP_K], dtype="float32")

        relevant_chunks = store_relevant_chunks(
            question,
            retrieval_query,
            documents,
            top_indices,
            top_scores
        )

        comparison = {}
        if COMPARE_INDEXES:
            comparison = compare_faiss_indexes(
                comparison_indexes,
                question_embedding,
                search_limit
            )

        print_retrieval_debug(
            documents,
            retrieval_query,
            COMPARE_INDEXES,
            REVIEW_ALL_SCORES,
            comparison,
            result_indices,
            result_scores,
            bm25_indices,
            bm25_scores,
            top_indices,
            top_scores
        )

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

        print_app_message("answer", answer)


if __name__ == "__main__":
    main()
