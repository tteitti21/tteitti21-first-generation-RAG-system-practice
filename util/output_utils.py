from colorama import Fore
from util.document_utils import get_document_text


def print_app_message(message_type, value=None):
    """Print a shared app status message.

    Args:
        message_type: Message key to print. Supported values are "divider",
            "chunks_loaded", "embeddings_ready", and "answer".
        value: Optional value used by message types that need content, such as
            the chunk count for "chunks_loaded" or answer text for "answer".
    """

    if message_type == "divider":
        print(f"{Fore.CYAN}" + "_" * 50)
    elif message_type == "chunks_loaded":
        print(f"Loaded {value} chunks")
    elif message_type == "embeddings_ready":
        print(f"{Fore.GREEN}Embeddings ready")
    elif message_type == "answer":
        print(f"{Fore.GREEN}\nAnswer:\n")
        print(value)
        print(f"{Fore.CYAN}" + "_" * 50)
    else:
        raise ValueError(f"Unknown output message type: {message_type}")


def print_search_results(documents, indices, scores, header=""):
    """Print ranked retrieval results with score, page, and chunk preview.

    Args:
        documents: Page-aware chunk objects in the same order used by FAISS/BM25.
        indices: Document indexes returned by retrieval.
        scores: Scores that correspond position-by-position with indices.
        header: Optional section heading printed before the ranked results.
    """

    if header:
        print(f"{Fore.CYAN}\n{header}")

    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        document = documents[int(idx)]

        print(f"\n--- Rank {rank} ---")
        print(f"Score: {score:.4f}")
        print(f"Page: {document['page']}")
        print(get_document_text(document)[:500])


def print_retrieval_debug(
    documents,
    retrieval_query,
    compare_indexes,
    review_all_scores,
    comparison,
    faiss_indices,
    faiss_scores,
    bm25_indices,
    bm25_scores,
    top_indices,
    top_scores
):
    """Print optional retrieval diagnostics and the final selected chunks.

    Args:
        documents: Page-aware chunk objects.
        retrieval_query: Query generated from the user question and recent chat
            history for embedding search.
        compare_indexes: When true, print flat and HNSW FAISS comparison output.
        review_all_scores: When true, print raw FAISS and BM25 candidate lists.
        comparison: Comparison results from compare_faiss_indexes().
        faiss_indices: Candidate indexes from the active FAISS search.
        faiss_scores: Scores matching faiss_indices.
        bm25_indices: Candidate indexes from BM25 keyword search.
        bm25_scores: Scores matching bm25_indices.
        top_indices: Final chunk indexes selected for answer context.
        top_scores: Final hybrid scores matching top_indices.
    """

    if compare_indexes:
        print_search_results(
            documents,
            comparison["flat"]["indices"],
            comparison["flat"]["scores"],
            "Flat search results:"
        )

        print_search_results(
            documents,
            comparison["hnsw"]["indices"],
            comparison["hnsw"]["scores"],
            "HNSW search results:"
        )

    if review_all_scores:
        print(f"{Fore.CYAN}\nRewritten retrieval query:")
        print(retrieval_query)

        print(f"{Fore.LIGHTMAGENTA_EX}\nFAISS candidates:")
        print_search_results(documents, faiss_indices, faiss_scores)

        print(f"{Fore.LIGHTMAGENTA_EX}\nBM25 candidates:")
        print_search_results(documents, bm25_indices, bm25_scores)

    print(f"{Fore.LIGHTMAGENTA_EX}\nTop chunks:")
    print_search_results(documents, top_indices, top_scores)
