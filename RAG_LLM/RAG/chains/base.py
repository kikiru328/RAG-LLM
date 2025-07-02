from RAG.retrieves.faiss import get_faiss_retriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def build_qa_chain(prompt, llm):
    """
    QA Chain Functions
    context, question → prompt → llm → output parser
    """
    retriever = get_faiss_retriever()

    if retriever is None:
        raise RuntimeError("Cannot Load Retriever")

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
