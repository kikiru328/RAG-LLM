import re

def format_response(text: str) -> str:
    """
    Response Formatter
    """
    #Emphasis
    emphasis_keywords = [
    ]
    for kw in emphasis_keywords:
        text = re.sub(rf"(?<!\*)\b({re.escape(kw)})\b(?!\*)", r"**\1**", text)

    # Line change by "X: ~"
    text = re.sub(
        r"\n*관련\s+\*\*?X\*\*?:\s*",  # 문장 + X
        r"\n\n**관련 X**\n\n",
        text
    )

    return text.strip()
