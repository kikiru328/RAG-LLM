from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils.paths import EMBED_MODEL_PATH, VECTOR_DB_PATH

def load_faiss_vector_store():
    """
    Load vector faiss file from path
    vector faiss: convert pdf files to vector by huggingface models
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_PATH,
        model_kwargs={"device": "cuda:0"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    return FAISS.load_local(
        folder_path=VECTOR_DB_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    
def get_faiss_retriever():
    """Return FAISS retriever"""
    try:
        vector_store = load_faiss_vector_store()
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3},
        )
    except Exception as e:
        print(f"Error creation retriever: {e}")
        return None