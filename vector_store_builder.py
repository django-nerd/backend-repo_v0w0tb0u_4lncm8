# ==============================================================================
# Step 4: Build the Vector Store
# ==============================================================================
#
# Description:
# This script reads the final, enriched corpus from `rag_corpus.jsonl`.
# It generates vector embeddings for each chunk and stores them along with
# their metadata in a persistent ChromaDB vector store. This is the final step
# in preparing the knowledge base.
#
# Output:
# A local ChromaDB database directory containing the indexed knowledge base,
# ready to be queried by the RAG system's AI service.
# ==============================================================================

import os
import json
from typing import List, Dict, Any

# --- CONFIGURATION ---
CORPUS_FILE = './backend/app/services/RAG/docs/rag_corpus.jsonl'
CHROMA_DB_PATH = './backend/app/services/chroma_db'
COLLECTION_NAME = 'n8n_universal_knowledge'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def load_corpus(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus file '{path}' not found. Please run the full pipeline (1-3) first.")
    chunks: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


essential_import_error_msg = (
    "Required packages for vector store build are missing. Install: "
    "'chromadb' and 'sentence-transformers'."
)


def build_vector_store(corpus_file: str = CORPUS_FILE,
                       chroma_path: str = CHROMA_DB_PATH,
                       collection_name: str = COLLECTION_NAME,
                       embedding_model: str = EMBEDDING_MODEL,
                       batch_size: int = 500) -> None:
    # Lazy imports to avoid impacting backend startup
    try:
        import chromadb  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(f"{essential_import_error_msg} Details: {e}")

    print("=============================================")
    print("        n8n Universal Vector Store Build       ")
    print("=============================================")

    print(f"Loading corpus from '{corpus_file}'...")
    chunks = load_corpus(corpus_file)
    if not chunks:
        print("Corpus is empty. Aborting.")
        return

    print(f"Loaded {len(chunks)} chunks to be indexed.")

    # Initialize embedding model
    print(f"Loading embedding model: '{embedding_model}'...")
    model = SentenceTransformer(embedding_model)

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name=collection_name)
    print(f"ChromaDB collection '{collection_name}' is ready.")

    # Prepare data
    ids = [str(chunk['metadata']['chunk_id']) for chunk in chunks]
    documents = [str(chunk['chunk_text']) for chunk in chunks]
    metadatas = [chunk.get('metadata', {}) for chunk in chunks]

    # Generate embeddings
    print("Generating embeddings (this may take a while for the full corpus)...")
    embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)

    # Add in batches
    total = len(ids)
    print(f"Adding {total} documents to ChromaDB in batches of {batch_size}...")
    for i in range(0, total, batch_size):
        end_index = min(i + batch_size, total)
        print(f"  -> Adding batch {i//batch_size + 1} ({i+1}-{end_index}/{total})")
        collection.add(
            ids=ids[i:end_index],
            embeddings=embeddings[i:end_index].tolist(),
            documents=documents[i:end_index],
            metadatas=metadatas[i:end_index]
        )

    print("\n=============================================")
    print(" Vector Store build complete!")
    print(f" Database is stored at: '{chroma_path}'")
    print("=============================================")


if __name__ == "__main__":
    try:
        build_vector_store()
    except FileNotFoundError as e:
        print(f"FATAL: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
