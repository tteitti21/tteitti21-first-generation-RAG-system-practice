import re

import numpy as np

from util.document_utils import get_document_text


BOOST_TOKEN_PREFIX_LENGTH = 4
BOOST_MIN_TOKEN_LENGTH = 3
SUBJECT_TERM_BOOST = 0.30
ACTION_TERM_BOOST = 0.08
MAX_ACTION_BOOST = 0.12
CONTEXT_TERM_BOOST = 0.05
MAX_CONTEXT_BOOST = 0.10
STRUCTURAL_TERM_BOOST = 0.80
STRUCTURAL_SYNONYM_GROUPS = [
    ["sisällysluettelo", "sisältö", "sisällys"],
    ["lähdeluettelo", "lähteet"],
    ["kuvaluettelo", "kuvat"],
    ["taulukkoluettelo", "taulukot"],
    ["liiteluettelo", "liitteet"],
    ["sanasto", "termistö"]
]


def add_unique_term(terms, seen_terms, term):
    """Append term once while preserving the original order."""

    normalized = term.lower()

    if normalized in seen_terms:
        return

    seen_terms.add(normalized)
    terms.append(term)


def term_matches_synonym_group(term, synonym_group):
    """Return True when a term belongs to a structural synonym group.

    The check is word-aware so that document-structure words do not match
    unrelated verbs. For example, "kuvat" should match the structural heading
    "Kuvat", but it should not make the verb "kuvata" structural.
    """

    normalized = term.strip().lower()

    if not normalized:
        return False

    words = re.findall(r"\w+", normalized)

    return any(
        word == synonym
        or (len(synonym) >= 6 and word.startswith(synonym))
        or (len(word) >= 6 and synonym.startswith(word))
        for word in words
        for synonym in synonym_group
    )


def is_structural_term(term):
    """Return True when a term belongs to the structural synonym vocabulary."""

    return any(
        term_matches_synonym_group(term, synonym_group)
        for synonym_group in STRUCTURAL_SYNONYM_GROUPS
    )


def expand_terms_with_synonyms(terms):
    """Add deterministic document-structure synonyms to analyzed terms.

    LLM query analysis may know that "sisällysluettelo" means table of contents,
    but the PDF heading may simply be "Sisältö". Expanding a small, predictable
    synonym list makes those document-structure terms easier to retrieve without
    relying on the model to choose the exact same word as the PDF.
    """

    expanded_terms = []
    seen_terms = set()

    for term in terms:
        if not isinstance(term, str) or not term.strip():
            continue

        term = term.strip()
        add_unique_term(expanded_terms, seen_terms, term)

        for synonym_group in STRUCTURAL_SYNONYM_GROUPS:
            if not term_matches_synonym_group(term, synonym_group):
                continue

            for synonym in synonym_group:
                add_unique_term(expanded_terms, seen_terms, synonym)

    return expanded_terms


def find_structural_terms_in_text(text):
    """Find document-structure terms directly from the user's question text."""

    structural_terms = []
    seen_terms = set()

    for synonym_group in STRUCTURAL_SYNONYM_GROUPS:
        if not term_matches_synonym_group(text, synonym_group):
            continue

        for synonym in synonym_group:
            add_unique_term(structural_terms, seen_terms, synonym)

    return structural_terms


def expand_query_analysis_terms(query_analysis, question):
    """Expand query analysis terms with deterministic structural synonyms.

    The LLM remains responsible for understanding the question, but this helper
    handles stable vocabulary differences like "sisällysluettelo" versus
    "Sisältö". Structural terms found directly in the question are added to
    subject_terms because they usually describe the concrete document part being
    asked about.
    """

    expanded_analysis = dict(query_analysis)
    question_structural_terms = find_structural_terms_in_text(question)

    expanded_analysis["subject_terms"] = expand_terms_with_synonyms(
        query_analysis.get("subject_terms", []) + question_structural_terms
    )
    expanded_analysis["action_terms"] = expand_terms_with_synonyms(
        query_analysis.get("action_terms", [])
    )
    expanded_analysis["context_terms"] = expand_terms_with_synonyms(
        query_analysis.get("context_terms", [])
    )

    return expanded_analysis


def tokenize_for_boost_matching(text):
    """Tokenize text for subject/context term matching.

    This tokenizer is intentionally separate from BM25 tokenization. BM25 uses
    five-character prefixes and ignores very short words, while subject matching
    should also catch shorter meaningful words like "ovi", "auto", or "lampun".
    Four-character prefixes are a small Finnish-friendly compromise:
    "autoissa" and "autossa" both become "auto", and "lampun" and "lamppu"
    both become "lamp".
    """

    tokens = []

    for word in re.findall(r"\w+", text.lower()):
        if len(word) < BOOST_MIN_TOKEN_LENGTH:
            continue

        tokens.append(word[:BOOST_TOKEN_PREFIX_LENGTH])

    return tokens


def get_words_for_matching(text):
    """Return full words for stricter structural-term matching.

    Normal boost terms intentionally use short prefixes, but document-structure
    terms are too easy to overmatch that way. For example, "sisällysluettelo",
    "sisältö", and unrelated words starting with "sisä" would all share the same
    prefix token. Full words avoid that noisy match.
    """

    return [
        word
        for word in re.findall(r"\w+", text.lower())
        if len(word) >= BOOST_MIN_TOKEN_LENGTH
    ]


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

        if is_structural_term(term):
            tokens = sorted(set(get_words_for_matching(term)))
            match_type = "structural"
        else:
            tokens = sorted(set(tokenize_for_boost_matching(term)))
            match_type = "prefix"

        if not tokens:
            continue

        normalized_terms.append(
            {
                "term": term,
                "tokens": tokens,
                "match_type": match_type
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
        {
            "tokens": set(tokenize_for_boost_matching(get_document_text(document))),
            "words": set(get_words_for_matching(get_document_text(document)))
        }
        for document in documents
    ]


def structural_tokens_match(term_tokens, document_words):
    """Match structural terms with full words instead of prefix tokens."""

    return all(
        any(
            document_word == term_token
            or (
                len(term_token) >= 6
                and document_word.startswith(term_token)
            )
            for document_word in document_words
        )
        for term_token in term_tokens
    )


def get_matching_terms(term_specs, document_index_entry):
    """Return each original term whose normalized tokens appear in a chunk."""

    matches = []
    document_tokens = document_index_entry["tokens"]
    document_words = document_index_entry["words"]

    for term_spec in term_specs:
        if term_spec["match_type"] == "structural":
            if structural_tokens_match(term_spec["tokens"], document_words):
                matches.append(term_spec["term"])

            continue

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
    boosts because they are useful, but often broader. Structural terms such as
    "sisällysluettelo" receive their own stronger boost because headings like
    "Sisältö" often have little surrounding vocabulary to help them rank.
    Repeated occurrences inside the same chunk do not matter, which prevents
    keyword repetition from overpowering relevance.
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
        matched_structural_terms = [
            term
            for term in (
                matched_subject_terms
                + matched_action_terms
                + matched_context_terms
            )
            if is_structural_term(term)
        ]
        matched_non_structural_subject_terms = [
            term
            for term in matched_subject_terms
            if not is_structural_term(term)
        ]

        # Subject match is intentionally strong, but only once per chunk. The
        # reranker still decides whether the chunk deserves final answer context.
        subject_boost = (
            SUBJECT_TERM_BOOST
            if matched_non_structural_subject_terms
            else 0.0
        )
        structural_boost = STRUCTURAL_TERM_BOOST if matched_structural_terms else 0.0
        action_boost = min(
            len(matched_action_terms) * ACTION_TERM_BOOST,
            MAX_ACTION_BOOST
        )
        context_boost = min(
            len(matched_context_terms) * CONTEXT_TERM_BOOST,
            MAX_CONTEXT_BOOST
        )
        final_score = (
            base_score
            + subject_boost
            + structural_boost
            + action_boost
            + context_boost
        )

        score_details[index] = {
            "base_score": base_score,
            "score": final_score,
            "subject_boost": subject_boost,
            "structural_boost": structural_boost,
            "action_boost": action_boost,
            "context_boost": context_boost,
            "matched_subject_terms": matched_subject_terms,
            "matched_structural_terms": matched_structural_terms,
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
