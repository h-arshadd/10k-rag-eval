from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_DIR = Path("data/raw")


def load_pages():
    docs = []
    for pdf in sorted(RAW_DIR.rglob("*.pdf")):
        pages = PyMuPDFLoader(str(pdf)).load()
        for p in pages:
            p.metadata["company"] = pdf.parent.name
            p.metadata["filename"] = pdf.name
        docs.extend(pages)
    return docs


def chunk_pages(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        c.metadata["chunk_type"] = "text"
    return chunks


if __name__ == "__main__":
    docs = load_pages()
    chunks = chunk_pages(docs)
    print(len(docs), "pages ->", len(chunks), "chunks")