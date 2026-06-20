from colorama import Fore
from util.document_utils import get_document_text


def print_app_message(message_type, value=None):
    if message_type == "divider":
        print(f"{Fore.CYAN}" + "_" * 50)
    elif message_type == "chunks_loaded":
        print(f"Loaded {value} chunks")
    elif message_type == "embeddings_ready":
        print(f"{Fore.GREEN}Embeddings ready")
    elif message_type == "answer":
        print(f"{Fore.GREEN}\nAnswer:\n")
        print(value)
        print(f"{Fore.CYAN}" + "_" * 50)
    else:
        raise ValueError(f"Unknown output message type: {message_type}")


def print_search_results(documents, indices, scores, header=""):
    if header:
        print(f"{Fore.CYAN}\n{header}")

    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        document = documents[int(idx)]

        print(f"\n--- Rank {rank} ---")
        print(f"Score: {score:.4f}")
        print(f"Page: {document['page']}")
        print(get_document_text(document)[:500])
