import streamlit as st
import json
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.paths import EMBED_MODEL_PATH, SEMANTIC_CLF_EX_PATH
from utils.logger import logger

@st.cache_resource
def load_similarity_classifier() -> tuple:
    """
    load embed model (Korean) and example data
    """
    model = SentenceTransformer(EMBED_MODEL_PATH)   
     
    with open(SEMANTIC_CLF_EX_PATH, "r", encoding="UTF-8") as f:
        examples = json.load(f)
    
    example_embeddings = {
        label: model.encode(sentences) for label, sentences in examples.items()
    }

    return model, example_embeddings

def classify_by_similarity(
    user_input: str, 
    model: SentenceTransformer, 
    example_embeddings: Dict[str, np.ndarray]) -> str:
    """
    주어진 질문을 임베딩 후 각 카테고리 예시들과 평균 유사도로 분류
    
    return: "domain_specific" or "domain_general" or "general"
    """
    input_embedding = model.encode([user_input])
    similarities = {
        label: np.mean(cosine_similarity(input_embedding, embs)) for label, embs in example_embeddings.items()
    }
    
    # Debug
    logger.debug(f"user input: {user_input}")
    for label, score in similarities.items():
        logger.debug(f"DEBUG {label} 유사도: {score:.4f}")

    best_label = max(similarities, key=similarities.get)
    return best_label


def classifier_system_prompt(user_input: str) -> str:
    
    """
    classifier prompt type by keywords
    """
    
    domain_general_keywords = [...]
    domain_specific_keywords = [...]

    contains_domain_general = any(k in user_input for k in domain_general_keywords)
    contains_domain_specific = any(k in user_input for k in domain_specific_keywords)

    if contains_domain_general and contains_domain_specific:
        return "domain_specific"
    elif contains_domain_specific:
        return "domain_specific"
    elif contains_domain_general:
        return "domain_general"
    model, example_embeddings = load_similarity_classifier()
    return classify_by_similarity(user_input, model, example_embeddings)
