from RAG.retrieves.faiss import get_faiss_retriever
from RAG.retrieves.utils import merge_docs

def run_retriever(user_input: str, classification_result: str):
    if classification_result == "general":
        # do not run retriever
        return [], ""
    
    retriever = get_faiss_retriever()
    docs = retriever.invoke(user_input)
    context = merge_docs(docs=docs)
    return docs, context