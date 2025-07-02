import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# RAG/faiss
EMBED_MODEL_PATH: str = os.getenv("EMBED_MODEL_PATH")
VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH")

# APP: Legal_Chat 
VLLM_MODEL_PATH: str = os.getenv("VLLM_MODEL_PATH")
RULE_FILTER_PATH: str = os.getenv("RULE_FILTER_PATH")
PROMPT_TEMPLATE_PATH: str = os.getenv("PROMPT_TEMPLATE_PATH")
SEMANTIC_CLF_EX_PATH: str = os.getenv("SEMANTIC_CLF_EX_PATH")

# Analysis
WORDCLOUD_FONT_PATH: str = os.getenv("WORDCLOUD_FONT_PATH")
WORDCLOUD_STOPWORD_PATH: str = os.getenv("WORDCLOUD_STOPWORD_PATH")
