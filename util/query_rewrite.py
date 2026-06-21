def rewrite_query_for_retrieval(client, question, chat_history, recent_chat_turns):
    """Rewrite the current question into a retrieval-focused search query.

    The OpenAI client and history window are passed in from main.py so this
    utility does not depend on hidden globals from another file.
    """

    recent_history = chat_history[-recent_chat_turns:]

    history_text = "\n".join(
        f"User: {turn['question']}\nAssistant: {turn['answer']}"
        for turn in recent_history
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """
You rewrite user questions into better search queries for a RAG system.

Your goal is not to answer the question.
Your goal is to produce a search query that helps find the relevant document chunk.

Rules:
- Keep important technical terms.
- Add likely related terms when useful.
- Preserve Finnish terms if the question is in Finnish.
- Include relevant context from recent conversation if needed.
- Do not invent specific facts, page numbers, or results.
- Return only the rewritten search query.
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

    return response.choices[0].message.content.strip()
