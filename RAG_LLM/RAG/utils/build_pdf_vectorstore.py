## code from index.ipynb (archive)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import pymupdf4llm
from langchain.schema import Document
from markdown_langchain import llamaindex_to_langchain

# 1. PDF → Markdown → LangChain 문서 변환
pdf_path = "../resources/data.pdf"
loader = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, image_format=True)
documents = llamaindex_to_langchain(loader)

# 2. 텍스트 쪼개기
def get_text_splitter(chunk_size=500, chunk_overlap=200):
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

splitter = get_text_splitter()
split_documents = splitter.split_documents(documents)

# 3. 임베딩 모델 로드
embed_model = HuggingFaceEmbeddings(
    model_name='../models/jhgan/ko-sroberta-multitask-strans',
    model_kwargs={"device": "cuda:0"},
    encode_kwargs={"normalize_embeddings": True},
)

# 4. 벡터 DB 생성 및 저장
vector_store = FAISS.from_documents(
    split_documents,
    embedding=embed_model,
    distance_strategy=DistanceStrategy.COSINE
)
vector_store.save_local('../vectorstore/chunck_size_500_200')

# 5. 로드 후 검색 테스트 (선택적)
vector_store = FAISS.load_local(
    '../vectorstore/chunck_size_500_200',
    embed_model,
    allow_dangerous_deserialization=True
)