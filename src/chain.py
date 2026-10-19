from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from embed import get_vector_store

load_dotenv()  # reads GROQ_API_KEY from .env

vector_store = get_vector_store()
llm = ChatGroq(model="openai/gpt-oss-120b")

prompt = PromptTemplate.from_template("""
    You are a helpful assistant. Answer the question using only the context.
    Mention the filename and page number you used.
    If the answer is not in the context, say 'I dont know.'
    Context: {context}
    Question: {question}
""")


# Chain: get_context | prompt | llm
def get_context(query: str):
    data = vector_store.similarity_search(query=query, k=5)
    context = ""
    for doc in data:
        page = doc.metadata["page"] + 1  # PyMuPDF counts pages from 0
        context += f"[{doc.metadata['filename']}, page {page}]\n{doc.page_content}\n\n"
    return {"context": context, "question": query}


rag_chain = get_context | prompt | llm

if __name__ == "__main__":
    res = rag_chain.invoke("What was Microsoft's total revenue in fiscal 2024?")
    print(res.content)