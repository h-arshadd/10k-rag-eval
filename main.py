from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

docs = []
for pdf in sorted(Path("docs").rglob("*.pdf")):
    pages = PyMuPDFLoader(str(pdf)).load()
    for p in pages:
        p.metadata["company"] = pdf.parent.name   # "microsoft" or "nike"
        p.metadata["filename"] = pdf.name
    docs.extend(pages)

print(len(docs))