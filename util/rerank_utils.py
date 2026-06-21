import json

from util.document_utils import get_document_text


def build_rerank_prompt(question, retrieval_query, documents, candidate_indices):
    """Build the compact candidate list that the reranker model will inspect."""

    candidates = []

    for index in candidate_indices:
        document = documents[int(index)]
        candidates.append(
            {
                "chunk_index": int(index),
                "page": document["page"],
                "text": get_document_text(document)
            }
        )

    return f"""
Original question:
{question}

Retrieval query:
{retrieval_query}

Candidate chunks:
{json.dumps(candidates, ensure_ascii=False)}
""".strip()


def parse_reranked_indices(raw_response, allowed_indices):
    """Parse model output and keep only known candidate indexes.

    The reranker is asked to return JSON, but this guard keeps the application
    usable even if the model returns duplicates, strings, or unknown indexes.
    """

    allowed_indices = [int(index) for index in allowed_indices]
    allowed_set = set(allowed_indices)
    parsed = json.loads(raw_response)

    if isinstance(parsed, dict):
        parsed = parsed.get("chunk_indices", [])

    reranked_indices = []

    for index in parsed:
        index = int(index)

        if index in allowed_set and index not in reranked_indices:
            reranked_indices.append(index)

    # Add any missing candidates at the end in their original hybrid order.
    # This means a partial rerank response can improve the top results without
    # accidentally dropping valid candidates.
    for index in allowed_indices:
        if index not in reranked_indices:
            reranked_indices.append(index)

    return reranked_indices


def rerank_candidate_chunks(
    client,
    question,
    retrieval_query,
    documents,
    candidate_indices,
    candidate_scores
):
    """Rerank retrieved candidates using the full question and chunk text.

    FAISS and BM25 are good first-stage retrievers because they are fast. This
    second stage is slower, but it can compare the actual candidate text against
    the user's question and move the most answer-bearing chunks higher.
    """

    if len(candidate_indices) <= 1:
        return candidate_indices, candidate_scores

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
You rerank retrieved document chunks for a RAG system.

Return only JSON in this exact shape:
{"chunk_indices": [1, 2, 3]}

Rules:
- Use only the provided candidate chunks.
- Prefer chunks that directly contain information needed to answer the question.
- Keep supporting context lower than chunks with the actual answer.
- Do not invent chunk indexes.
"""
                },
                {
                    "role": "user",
                    "content": build_rerank_prompt(
                        question,
                        retrieval_query,
                        documents,
                        candidate_indices
                    )
                }
            ]
        )

        reranked_indices = parse_reranked_indices(
            response.choices[0].message.content,
            candidate_indices
        )
    except Exception as error:
        print(f"Reranking failed, using hybrid order instead: {error}")
        return candidate_indices, candidate_scores

    score_by_index = {
        int(index): float(score)
        for index, score in zip(candidate_indices, candidate_scores)
    }

    return (
        reranked_indices,
        [score_by_index[index] for index in reranked_indices]
    )
