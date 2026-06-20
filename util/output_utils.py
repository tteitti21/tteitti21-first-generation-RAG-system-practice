from util.document_utils import get_document_text


def print_search_results(documents, indices, scores):
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        document = documents[int(idx)]

        print(f"\n--- Rank {rank} ---")
        print(f"Score: {score:.4f}")
        print(f"Page: {document['page']}")
        print(get_document_text(document)[:500])
