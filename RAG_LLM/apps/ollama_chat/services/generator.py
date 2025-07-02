import streamlit as st
from apps.ollama_chat.models.ollama_models import load_bllossom_latest
from apps.ollama_chat.prompts.ollama_prompts import load_ollama_prompt
from RAG.chains.base import build_qa_chain
from utils.logger import logger
def create_chain():
    """
    create LCEL QA Chain
    """
    try:
        prompt = load_ollama_prompt()
        model = load_bllossom_latest()
        return build_qa_chain(prompt, model)
    except Exception as e:
        logger.error(f"Error in Crate Retriever: {e}", exc_info=True)
        st.error("An error occurred while creating the chain.")
        return None