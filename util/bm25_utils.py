import math
import re
from collections import Counter

import numpy as np

from util.document_utils import get_document_text


BM25_K1 = 1.5
BM25_B = 0.75
TOKEN_STEM_LENGTH = 5


def tokenize_for_bm25(text):
    # BM25 is based on word overlap, so the text must first be split into
    # searchable tokens. The short prefix acts as a tiny Finnish-friendly stem:
    # taulukko, taulukkoa, and taulukot all become taulu.
    return [
        word[:TOKEN_STEM_LENGTH]
        for word in re.findall(r"\w+", text.lower())
        if len(word) >= TOKEN_STEM_LENGTH
    ]


def build_bm25_index(documents):
    # Precompute all document-side BM25 data once at startup. This keeps each
    # later question fast because query-time scoring can reuse these counters.
    tokenized_documents = [
        tokenize_for_bm25(get_document_text(document))
        for document in documents
    ]

    # Term frequency tells how many times each token appears in each document.
    # Example: Counter({"taulu": 8, "react": 2})
    term_frequencies = [
        Counter(tokens)
        for tokens in tokenized_documents
    ]

    # Document length is measured in tokens, not characters. BM25 uses this to
    # avoid always favoring long chunks just because they contain more words.
    document_lengths = [
        len(tokens)
        for tokens in tokenized_documents
    ]
    document_count = len(documents)
    average_document_length = (
        sum(document_lengths) / document_count
        if document_count
        else 0
    )

    document_frequencies = Counter()

    for tokens in tokenized_documents:
        # Document frequency counts in how many different chunks a token appears.
        # Common words get lower weight later; rare words get higher weight.
        document_frequencies.update(set(tokens))

    return {
        "term_frequencies": term_frequencies,
        "document_lengths": document_lengths,
        "document_frequencies": document_frequencies,
        "document_count": document_count,
        "average_document_length": average_document_length
    }


def get_bm25_scores(query, bm25_index):
    # Returns one BM25 keyword score for every document chunk.
    query_tokens = tokenize_for_bm25(query)
    scores = np.zeros(bm25_index["document_count"], dtype="float32")

    if not query_tokens or bm25_index["document_count"] == 0:
        return scores

    average_document_length = bm25_index["average_document_length"] or 1

    for token in query_tokens:
        document_frequency = bm25_index["document_frequencies"].get(token, 0)

        if document_frequency == 0:
            continue

        # Rare query terms get a larger value than terms appearing everywhere.
        # This is why a broad word like opinnäytetyö matters less than taulukko
        # if opinnäytetyö appears in many candidate chunks.
        inverse_document_frequency = math.log(
            1
            + (
                bm25_index["document_count"] - document_frequency + 0.5
            ) / (document_frequency + 0.5)
        )

        for document_index, term_frequency in enumerate(bm25_index["term_frequencies"]):
            token_count = term_frequency.get(token, 0)

            if token_count == 0:
                continue

            document_length = bm25_index["document_lengths"][document_index]
            # Longer chunks are normalized downward a bit, so they do not win
            # purely because they contain more total words.
            length_normalizer = 1 - BM25_B + BM25_B * (
                document_length / average_document_length
            )

            # Repeated matches help, but BM25 intentionally saturates the value.
            # The 10th occurrence of a word does not help as much as the 1st.
            term_score = (
                token_count * (BM25_K1 + 1)
            ) / (token_count + BM25_K1 * length_normalizer)

            scores[document_index] += inverse_document_frequency * term_score

    return scores


def get_top_bm25_results(query, bm25_index, top_k):
    # Keep only the strongest keyword matches so the hybrid merge stays small.
    scores = get_bm25_scores(query, bm25_index)

    if len(scores) == 0:
        return np.array([], dtype="int64"), scores

    top_positions = np.argsort(scores)[-top_k:][::-1]
    top_positions = top_positions[scores[top_positions] > 0]

    return top_positions, scores[top_positions]


def normalize_scores(scores):
    # BM25 scores are not naturally 0-1 like normalized FAISS cosine scores.
    # Scaling the best BM25 candidate to 1 makes keyword boosts comparable.
    if len(scores) == 0:
        return scores

    max_score = np.max(scores)

    if max_score <= 0:
        return np.zeros_like(scores, dtype="float32")

    return scores / max_score


def combine_faiss_and_bm25_results(
    faiss_indices,
    faiss_scores,
    bm25_indices,
    bm25_scores,
    keyword_weight=0.35
):
    combined_scores = {}

    for index, score in zip(faiss_indices, faiss_scores):
        # Keep the FAISS score as the base score. This avoids demoting a strong
        # semantic match just because it did not also match exact keywords.
        combined_scores[int(index)] = float(score)

    normalized_bm25_scores = normalize_scores(bm25_scores)

    for index, score in zip(bm25_indices, normalized_bm25_scores):
        index = int(index)
        # BM25 adds a boost on top of the semantic score. A BM25-only document
        # can still enter the result set, but strong FAISS matches are preserved.
        combined_scores[index] = (
            combined_scores.get(index, 0)
            + keyword_weight * float(score)
        )

    sorted_results = sorted(
        combined_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return (
        np.array([index for index, _ in sorted_results], dtype="int64"),
        np.array([score for _, score in sorted_results], dtype="float32")
    )
