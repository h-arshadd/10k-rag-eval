from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ingest import load_pages, chunk_pages

DB_DIR = "chroma_db"

# the embedding model (turns text into numbers)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def get_vector_store():
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


def build_index():
    chunks = chunk_pages(load_pages())

    get_vector_store().delete_collection()  # start fresh so re-running doesn't add duplicates

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    print(len(chunks), "chunks stored in", DB_DIR)


if __name__ == "__main__":
    build_index()