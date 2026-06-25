# First-Generation RAG Practice

This is a small practice project for learning how a first-generation
Retrieval-Augmented Generation (RAG) workflow can answer questions about a PDF
document.

The app loads a PDF, splits it into chunks, creates embeddings, combines
semantic and keyword retrieval, optionally reranks candidate chunks with an LLM,
and answers questions from the selected document context in the terminal.

Before retrieval, the app asks the model to analyze the question into a
standalone retrieval query, narrow subject terms, action terms, and broader
context terms. Subject terms receive a strong one-time score boost per chunk,
action terms receive a medium-small boost for close verb matches, and context
terms receive a smaller boost.

## What This Project Does

- Loads project configuration from a local `.env` file.
- Reads a PDF from the `docs/` folder.
- Splits PDF pages into reusable text chunks.
- Creates OpenAI embeddings for each chunk.
- Builds and stores a FAISS index for semantic search.
- Builds a BM25-style keyword index for exact-term retrieval.
- Applies query-term boosts for subject, action, context, and structural terms.
- Optionally compares FAISS index types for debugging.
- Optionally reranks retrieved candidates before final context selection.
- Stores chunks, embeddings, the FAISS index, and retrieval history locally.
- Generates answers with citations such as `[page 4, chunk 12]`.

## Project Structure

```text
.
+-- docs/
|   +-- Teittinen_Tino.pdf
|   +-- .gitkeep
+-- util/
|   +-- bm25_utils.py
|   +-- chunk_utils.py
|   +-- document_utils.py
|   +-- env_utils.py
|   +-- faiss_utils.py
|   +-- file_utils.py
|   +-- output_utils.py
|   +-- pdf_utils.py
|   +-- query_rewrite.py
|   +-- rerank_utils.py
|   +-- retrieval_boost_utils.py
|   +-- __init__.py
+-- .env
+-- .gitignore
+-- main.py
+-- README.txt
+-- requirements.txt
```

## Environment Variables

`OPENAI_API_KEY`

Your OpenAI API key. This is required for embeddings, query analysis,
reranking, and final answer generation. Set it in your shell environment before
running the app.

`PDF_PATH`

Path to the PDF file that the app should read. Relative paths are resolved from
the project root.

`CHUNKS_PATH`

Path where generated document chunks are saved as JSON. Keeping this file lets
the app reuse chunks instead of recreating them every run.

`EMBEDDINGS_PATH`

Path where generated chunk embeddings are saved as JSON. Keeping this file lets
the app reuse embeddings instead of calling the embedding model every run.

`RELEVANT_CHUNKS_PATH`

Path where recent retrieval results are saved. This is useful for reviewing
which chunks were selected for previous questions.

`FAISS_INDEX_PATH`

Base path for the generated FAISS index file. The selected FAISS index type is
added to the filename automatically.

`FAISS_INDEX_TYPE`

FAISS index type to use. Supported values are `flat` and `hnsw`.

`COMPARE_INDEXES`

Set to `true` to compare flat and HNSW FAISS search results during retrieval.
Set to `false` for normal use.

`REVIEW_ALL_SCORES`

Set to `true` to print detailed retrieval scores, including FAISS, BM25, boost,
and final selected chunk scores. Set to `false` for quieter output.

`ENABLE_RERANKING`

Set to `true` to use the LLM reranking step after initial retrieval. Set to
`false` to use the combined retrieval scores directly.

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Set the OpenAI API key before running the app.

PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_api_key_here"
```

## Running The App

Start the terminal question-answer loop:

```bash
python main.py
```

Type a question about the configured PDF. Type `exit` or `quit` to stop the
program.

## Notes

This project is meant for practice and experimentation, not production use.
Generated files inside `docs/` are ignored by Git by default, while PDF files in
that folder can still be tracked.
