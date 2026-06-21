RAG Practice Program

This is a small practice program for learning how retrieval-augmented generation
(RAG) works with a PDF document.

The program reads text from a PDF, splits the text into chunks, creates
embeddings for those chunks, and uses the most relevant chunks as context when
answering questions.

Retrieval uses two approaches together:

- FAISS semantic search finds chunks that are close in meaning to the question.
- BM25-style keyword search helps find chunks with important exact words.

Before retrieval, the app asks the model to analyze the question into a
standalone retrieval query, narrow subject terms, action terms, and broader
context terms. Subject terms receive a strong one-time score boost per chunk,
action terms receive a medium-small boost for close verb matches, and context
terms receive a smaller boost.

Some document-structure words are expanded with simple synonyms before
retrieval. For example, a question about "sisällysluettelo" can also match a PDF
heading named "Sisältö". These structural terms use stricter matching and a
stronger boost than ordinary context terms.

The FAISS, BM25, and boosted scores are combined before choosing the final
chunks that are sent to the model. This helps with questions where exact terms
matter, such as asking how many tables or figures the document contains.

After FAISS and BM25 have found candidate chunks, the app can also use an LLM
reranking step. Reranking looks at the actual candidate chunk text and moves the
chunks that best answer the question higher before the final context is built.
Only the strongest candidates are sent to reranking, which keeps token usage
smaller than reranking every retrieved chunk.

The program stores generated chunks, embeddings, retrieved chunks, and the
FAISS index so the same work does not need to be repeated every time.

Answers are generated from the selected context chunks and should cite the
supporting page and chunk number, for example [page 4, chunk 12].

The file paths are configured in the .env file.

Optional debug settings can also be configured in .env:

- COMPARE_INDEXES can compare flat and HNSW FAISS search results.
- REVIEW_ALL_SCORES can print the FAISS, BM25, and final selected chunks.
- ENABLE_RERANKING can turn the LLM reranking step on or off.

This project is meant for practice and experimentation, not as a production
application.
