RAG Practice Program

This is a small practice program for learning how retrieval-augmented generation
(RAG) works with a PDF document.

The program reads text from a PDF, splits the text into chunks, creates
embeddings for those chunks, and uses the most relevant chunks as context when
answering questions.

Retrieval uses two approaches together:

- FAISS semantic search finds chunks that are close in meaning to the question.
- BM25-style keyword search helps find chunks with important exact words.

The scores from these approaches are combined before choosing the final chunks
that are sent to the model. This helps with questions where exact terms matter,
such as asking how many tables or figures the document contains.

The program stores generated chunks, embeddings, retrieved chunks, and the
FAISS index so the same work does not need to be repeated every time.

The file paths are configured in the .env file.

Optional debug settings can also be configured in .env:

- COMPARE_INDEXES can compare flat and HNSW FAISS search results.
- REVIEW_ALL_SCORES can print the FAISS, BM25, and final selected chunks.

This project is meant for practice and experimentation, not as a production
application.
