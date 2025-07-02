from dataclasses import dataclass

@dataclass
class PromptRequest:
    classification_result: str
    user_input: str
    context: str
    tokenizer: object
    llm: object
    