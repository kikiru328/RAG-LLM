from langchain_core.prompts import ChatPromptTemplate


def load_ollama_prompt():
    """load llama3 prompt template"""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    ...
                ),
            ),
            ("user", "#Question : \n{question}"),
        ]
    )
