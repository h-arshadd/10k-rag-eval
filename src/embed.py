import chromadb
from sentence_transformers import SentenceTransformer

from ingest import load_pages, chunk_pages

DB_DIR = "data/chroma"
COLLECTION = "tenk_baseline"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 64


def clean_metadata(m):
    # Chroma only accepts str/int/float/bool, so keep just the fields we need
    return {
        "company": m["company"],
        "filename": m["filename"],
        "page": int(m["page"]) + 1,  # PyMuPDF is 0-indexed; store the real PDF page
        "chunk_id": m["chunk_id"],
        "chunk_type": m["chunk_type"],
    }


def build_index():
    chunks = chunk_pages(load_pages())
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)  # rebuild from scratch, no duplicates
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c.page_content for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        col.add(
            ids=[str(c.metadata["chunk_id"]) for c in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[clean_metadata(c.metadata) for c in batch],
        )
    print(f"Indexed {col.count()} chunks into '{COLLECTION}'")


def search(query, k=5):
    model = SentenceTransformer(MODEL_NAME)
    col = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    emb = model.encode([QUERY_PREFIX + query], normalize_embeddings=True).tolist()
    return col.query(query_embeddings=emb, n_results=k)


if __name__ == "__main__":
    build_index()
    res = search("What was Microsoft's total revenue in fiscal 2024?")
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        print(meta["filename"], "p.", meta["page"])
        print(doc[:200].replace("\n", " "), "\n")