# build_excel_vector_store.py

import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

# 1. 엑셀 로드 및 LangChain 문서 변환
df = pd.read_excel("data.xlsx")

def dataframe_to_langchain(df: pd.DataFrame) -> list[Document]:
    langchain_output = []
    for _, row in df.iterrows():
        metadata = {
            "source": "#"+str(row["title"]),
            "subject": "##"+str(row["subject"]),
            "content_title": "###"+str(row["content_title"]),
        }
        doc = Document(page_content=str(row["content"]), metadata=metadata)
        langchain_output.append(doc)
    return langchain_output

documents = dataframe_to_langchain(df)

# 2. 임베딩 모델 로드
embed_model = HuggingFaceEmbeddings(
    model_name='../models/jhgan/ko-sroberta-multitask-strans',
    model_kwargs={"device": "cuda:0"},
    encode_kwargs={"normalize_embeddings": True},
)

# 3. 벡터 DB 생성 및 저장
vector_store = FAISS.from_documents(
    documents,
    embedding=embed_model,
    distance_strategy=DistanceStrategy.COSINE
)
vector_store.save_local('../vectorstore/')
