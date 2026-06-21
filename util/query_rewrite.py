import json


def get_default_query_analysis(question):
    """Return the safest possible query analysis when the LLM step fails.

    The application can still retrieve with the original question even if JSON
    parsing fails, the model returns an unexpected shape, or the API call is
    unavailable during local testing.
    """

    return {
        "retrieval_query": question,
        "subject_terms": [],
        "action_terms": [],
        "context_terms": []
    }


def normalize_query_analysis(raw_analysis, question):
    """Validate and normalize the model-produced query analysis object.

    The rest of the retrieval pipeline expects a stable dictionary shape. This
    function keeps that contract stable even if the model returns a missing key,
    a single string where a list was expected, or empty retrieval text.
    """

    analysis = get_default_query_analysis(question)

    if not isinstance(raw_analysis, dict):
        return analysis

    retrieval_query = raw_analysis.get("retrieval_query")
    if isinstance(retrieval_query, str) and retrieval_query.strip():
        analysis["retrieval_query"] = retrieval_query.strip()

    for key in ["subject_terms", "action_terms", "context_terms"]:
        value = raw_analysis.get(key, [])

        if isinstance(value, str):
            value = [value]

        if not isinstance(value, list):
            continue

        # Keep order while removing duplicates and empty values. Order matters
        # for debugging because it mirrors what the model thought was important.
        seen_terms = set()
        clean_terms = []

        for term in value:
            if not isinstance(term, str):
                continue

            term = term.strip()
            normalized = term.lower()

            if not term or normalized in seen_terms:
                continue

            seen_terms.add(normalized)
            clean_terms.append(term)

        analysis[key] = clean_terms

    return analysis


def analyze_query_for_retrieval(client, question, chat_history, recent_chat_turns):
    """Create a structured retrieval analysis for the current user question.

    This is intentionally more specific than a plain query rewrite. The returned
    retrieval_query is used for semantic search, subject_terms are used for a
    strong score boost and direct subject-term candidate discovery, action_terms
    capture close verb/synonym relationships, and context_terms are used for a
    smaller boost plus better keyword retrieval.
    """

    recent_history = chat_history[-recent_chat_turns:]

    history_text = "\n".join(
        f"User: {turn['question']}\nAssistant: {turn['answer']}"
        for turn in recent_history
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
You analyze user questions for a RAG retrieval system.

Return only JSON in this exact shape:
{
  "retrieval_query": "standalone search query",
  "subject_terms": ["main concrete subjects"],
  "action_terms": ["important verbs and close verb synonyms"],
  "context_terms": ["supporting resolved terms"]
}

Definitions:
- retrieval_query: a standalone search query that resolves references like
  "it", "that", "niistä", "siinä", "näissä", or "sillä" using recent history.
- subject_terms: A subject term may be the thing being asked about, or the source/location
  scope that limits where the answer should be searched. Subject terms may come from the current question or recent
  history when the current question uses relative wording.
- action_terms: important verbs, actions, or close verbal synonyms that describe
  what is being asked about the subject. Examples: "vertaillaan", "vertailu",
  "verrataan", "muuttuu", "vaihtuu", "toteutetaan", "arvioidaan". Prefer close
  retrieval synonyms; avoid very broad loose verbs unless the question uses
  them directly.
- context_terms: broader entities, comparison targets, and descriptive terms
  that help retrieval, such as "Kotlin Multiplatform", "React Native",
  "helpompi", or "toteuttaminen".

Rules:
- Do not answer the question.
- Do not invent facts, page numbers, or expected results.
- Preserve Finnish terms when the question is Finnish.
- Prefer 1-3 subject_terms. Do not put every meaningful word there.
- Prefer 1-5 action_terms, including close synonyms when they help retrieval.
- Put the main reference target in subject_terms. Broad background terms usually
  belong in context_terms, unless they define the document scope being asked
  about, such as "tekstissä", "materiaalissa", "luvussa", or "vertailussa".
- In "Kummalla niistä aloitusnäkymän toteuttaminen oli helpompaa?",
  "aloitusnäkymä" is the subject term; "toteuttaminen" and "helpompaa" are
  action/context terms.
- In "Mitä teknologioita työssä vertaillaan yleisellä tasolla?",
  "teknologiat" is a subject term, and "vertaillaan", "verrataan", and
  "vertailu" are action terms.
"""
                },
                {
                    "role": "user",
                    "content": f"""
Recent conversation:
{history_text}

Current question:
{question}
"""
                }
            ]
        )

        raw_analysis = json.loads(response.choices[0].message.content)
        return normalize_query_analysis(raw_analysis, question)
    except Exception as error:
        print(f"Query analysis failed, using original question instead: {error}")
        return get_default_query_analysis(question)


def rewrite_query_for_retrieval(client, question, chat_history, recent_chat_turns):
    """Return only the rewritten retrieval query for older call sites."""

    return analyze_query_for_retrieval(
        client,
        question,
        chat_history,
        recent_chat_turns
    )["retrieval_query"]
