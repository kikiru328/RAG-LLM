import json
from utils.paths import PROMPT_TEMPLATE_PATH

def load_prompt_templates() -> dict:
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    templates = {}
    for key, value in data.items():
        if isinstance(value, list):
            templates[key] = "\n".join(value)
        else:
            templates[key] = value

    return templates

def get_system_prompt(classification_result: str, context: str = "") -> str:
    # context is came from RAG retriever
    # classification_result is came from classifer (user_input classification)
    templates = load_prompt_templates()
    base_prompt = templates.get(classification_result, templates["general"])
    return f"{base_prompt}\n\n[context]\n{context}"
    
def make_messages(classification_result: str, user_input: str, context: str) -> list[dict]:
    system_content = get_system_prompt(classification_result=classification_result,
                                       context=context,
                                       )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input},
    ]
    
def generate_prompt(classification_result: str, user_input: str, context: str, tokenizer) -> str:
    """
    Generate prompt using user_input, system prompt, context from RAG retriever
    """
    messages = make_messages(classification_result=classification_result, user_input=user_input, context=context)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )