import os

import faiss
import numpy as np


def build_faiss_index(doc_embeddings, index_type="flat"):
    # Convert embeddings into FAISS-friendly NumPy format.
    embeddings = np.asarray(doc_embeddings, dtype="float32").copy()
    faiss.normalize_L2(embeddings)

    # For document embeddings, the array is usually shaped like this:
    # (number_of_chunks, embedding_size). FAISS needs to know the latter.
    dimension = embeddings.shape[1]

    if index_type == "flat":
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "hnsw":
        hnsw_neighbors = 32
        index = faiss.IndexHNSWFlat(
            dimension,
            hnsw_neighbors,
            faiss.METRIC_INNER_PRODUCT
        )
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 64
    else:
        raise ValueError(f"Unknown FAISS index type: {index_type}")

    index.add(embeddings)

    return index


def load_or_create_faiss_index(
    doc_embeddings,
    document_count,
    faiss_index_path,
    embeddings_path,
    index_type="flat",
    force_recreate=False
):
    if os.path.exists(faiss_index_path) and not force_recreate:
        print(f"Found stored FAISS index in {faiss_index_path}...")

        index_is_newer_than_embeddings = (
            not os.path.exists(embeddings_path)
            or os.path.getmtime(faiss_index_path) >= os.path.getmtime(embeddings_path)
        )

        try:
            index = faiss.read_index(faiss_index_path)
        except RuntimeError:
            index = None
            print("Stored FAISS index could not be read. Recreating index...")

        # Number of vectors inside FAISS index should match number of document chunks.
        if index and index.ntotal == document_count and index_is_newer_than_embeddings:
            return index

        print("Stored FAISS index is stale or does not match document count. Recreating index...")

    print("Creating FAISS index...")
    index = build_faiss_index(doc_embeddings, index_type=index_type)

    print(f"Storing FAISS index in {faiss_index_path}...")
    parent_dir = os.path.dirname(faiss_index_path)

    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    faiss.write_index(index, faiss_index_path)

    return index


def normalize_query_embedding(query_embedding):
    query_embedding = np.asarray(query_embedding, dtype="float32")
    faiss.normalize_L2(query_embedding)

    return query_embedding


def search_faiss_index(index, query_embedding, top_k):
    scores, indices = index.search(query_embedding, top_k)

    return scores[0], indices[0]


def build_comparison_indexes(doc_embeddings):
    return {
        "flat": build_faiss_index(doc_embeddings, index_type="flat"),
        "hnsw": build_faiss_index(doc_embeddings, index_type="hnsw")
    }


def compare_faiss_indexes(comparison_indexes, query_embedding, top_k):
    comparison = {}

    for index_name, index in comparison_indexes.items():
        scores, indices = search_faiss_index(index, query_embedding, top_k)
        comparison[index_name] = {
            "scores": scores,
            "indices": indices
        }

    return comparison
