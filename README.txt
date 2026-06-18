RAG Practice Program

This is a small practice program for learning how retrieval-augmented generation
(RAG) works with a PDF document.

The program reads text from a PDF, splits the text into chunks, creates
embeddings for those chunks, and uses the most relevant chunks as context when
answering questions. It also stores chunks, embeddings, and retrieved chunks as
JSON files so the same work does not need to be repeated every time.

The file paths are configured in the .env file.

This project is meant for practice and experimentation, not as a production
application.

Create docs/ folder for the PDF and stored chunks/embeddings.