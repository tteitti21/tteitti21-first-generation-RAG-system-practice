import re

import numpy as np

from util.document_utils import get_document_text


BOOST_TOKEN_PREFIX_LENGTH = 4
BOOST_MIN_TOKEN_LENGTH = 3
BOOST_COMPOUND_WORD_MIN_LENGTH = 8
SUBJECT_TERM_BOOST = 0.30
ACTION_TERM_BOOST = 0.08
MAX_ACTION_BOOST = 0.12
CONTEXT_TERM_BOOST = 0.5
MAX_CONTEXT_BOOST = 0.10


def tokenize_for_boost_matching(text, include_compound_parts=False):
    """Tokenize text for subject/context term matching.

    This tokenizer is intentionally separate from BM25 tokenization. BM25 uses
    five-character prefixes and ignores very short words, while subject matching
    should also catch shorter meaningful words like "ovi", "auto", or "lampun".
    Four-character prefixes are a small Finnish-friendly compromise:
    "autoissa" and "autossa" both become "auto", and "lampun" and "lamppu"
    both become "lamp".

    When building the document-side index, include_compound_parts adds internal
    three- and four-character pieces for longer words. That lets a query term
    like "työssä" match a compound word like "opinnäytetyössä" through "työs",
    and also lets "työ" match "opinnäytetyö" through "työ". This is only used
    for document chunks, not for query terms, so phrase terms do not become
    overly strict.
    """

    tokens = []

    for word in re.findall(r"\w+", text.lower()):
        if len(word) < BOOST_MIN_TOKEN_LENGTH:
            continue

        tokens.append(word[:BOOST_TOKEN_PREFIX_LENGTH])

        if include_compound_parts and len(word) >= BOOST_COMPOUND_WORD_MIN_LENGTH:
            for token_length in range(BOOST_MIN_TOKEN_LENGTH, BOOST_TOKEN_PREFIX_LENGTH + 1):
                for start in range(1, len(word) - token_length + 1):
                    tokens.append(word[start:start + token_length])

    return tokens


def normalize_boost_terms(terms):
    """Turn model-provided terms into comparable token groups.

    A term may be one word, such as "aloitusnäkymä", or a small phrase, such as
    "React Native". Phrases are represented as multiple tokens and only match a
    document when every token is present somewhere in that document chunk.
    """

    normalized_terms = []
    seen_terms = set()

    for term in terms:
        if not isinstance(term, str):
            continue

        term = term.strip()
        term_key = term.lower()

        if not term or term_key in seen_terms:
            continue

        tokens = sorted(set(tokenize_for_boost_matching(term)))

        if not tokens:
            continue

        normalized_terms.append(
            {
                "term": term,
                "tokens": tokens
            }
        )
        seen_terms.add(term_key)

    return normalized_terms


def build_boost_index(documents):
    """Precompute searchable token sets for every chunk.

    The boost logic checks candidate chunks on every question. Precomputing each
    chunk into a set keeps that check cheap and avoids repeatedly tokenizing the
    full document collection inside the question loop.
    """

    return [
        set(
            tokenize_for_boost_matching(
                get_document_text(document),
                include_compound_parts=True
            )
        )
        for document in documents
    ]


def get_matching_terms(term_specs, document_tokens):
    """Return each original term whose normalized tokens appear in a chunk."""

    matches = []

    for term_spec in term_specs:
        if all(token in document_tokens for token in term_spec["tokens"]):
            matches.append(term_spec["term"])

    return matches


def find_subject_candidate_indices(boost_index, subject_terms):
    """Find chunks that contain at least one analyzed subject term.

    This is the part that lets a subject word almost guarantee a chance to be
    reranked. Even if a chunk lands just outside FAISS/BM25 top results, a direct
    subject-term match adds it to the candidate pool before boosting.
    """

    subject_specs = normalize_boost_terms(subject_terms)

    if not subject_specs:
        return np.array([], dtype="int64")

    matching_indices = []

    for index, document_tokens in enumerate(boost_index):
        if get_matching_terms(subject_specs, document_tokens):
            matching_indices.append(index)

    return np.array(matching_indices, dtype="int64")


def apply_query_term_boosts(
    boost_index,
    combined_indices,
    combined_scores,
    subject_terms,
    action_terms,
    context_terms
):
    """Add subject, action, and context boosts to combined retrieval scores.

    The base score still comes from FAISS/BM25 hybrid retrieval. Subject terms
    then receive one strong boost per chunk if any subject term matches.
    Action terms add medium-small capped boosts for verb relationships like
    "vertaillaan", "verrataan", and "vertailu". Context terms add smaller capped
    boosts because they are useful, but often broader. Repeated occurrences
    inside the same chunk do not matter, which prevents keyword repetition from
    overpowering relevance.
    """

    subject_specs = normalize_boost_terms(subject_terms)
    action_specs = normalize_boost_terms(action_terms)
    context_specs = normalize_boost_terms(context_terms)
    score_by_index = {
        int(index): float(score)
        for index, score in zip(combined_indices, combined_scores)
    }

    for index in find_subject_candidate_indices(boost_index, subject_terms):
        score_by_index.setdefault(int(index), 0.0)

    score_details = {}

    for index, base_score in score_by_index.items():
        document_tokens = boost_index[index]
        matched_subject_terms = get_matching_terms(subject_specs, document_tokens)
        matched_action_terms = get_matching_terms(action_specs, document_tokens)
        matched_context_terms = get_matching_terms(context_specs, document_tokens)

        # Subject match is intentionally strong, but only once per chunk. The
        # reranker still decides whether the chunk deserves final answer context.
        subject_boost = SUBJECT_TERM_BOOST if matched_subject_terms else 0.0
        action_boost = min(
            len(matched_action_terms) * ACTION_TERM_BOOST,
            MAX_ACTION_BOOST
        )
        context_boost = min(
            len(matched_context_terms) * CONTEXT_TERM_BOOST,
            MAX_CONTEXT_BOOST
        )
        final_score = base_score + subject_boost + action_boost + context_boost

        score_details[index] = {
            "base_score": base_score,
            "score": final_score,
            "subject_boost": subject_boost,
            "action_boost": action_boost,
            "context_boost": context_boost,
            "matched_subject_terms": matched_subject_terms,
            "matched_action_terms": matched_action_terms,
            "matched_context_terms": matched_context_terms
        }

    sorted_results = sorted(
        score_details.items(),
        key=lambda item: item[1]["score"],
        reverse=True
    )

    return (
        np.array([index for index, _ in sorted_results], dtype="int64"),
        np.array([details["score"] for _, details in sorted_results], dtype="float32"),
        score_details
    )
