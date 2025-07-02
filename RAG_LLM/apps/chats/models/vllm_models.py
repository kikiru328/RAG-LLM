import streamlit as st
import asyncio
from transformers import AutoTokenizer
from vllm import LLM
from utils.paths import VLLM_MODEL_PATH

# 비동기처리 event 확인
try:
    asyncio.get_running_loop() # 현재 실행중인 event loop 확인
except RuntimeError: # event loop 가 없다면
    asyncio.set_event_loop(asyncio.new_event_loop())
    
@st.cache_resource # caching the loaded model
def load_qna_model():

    tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    llm = LLM(
        model=VLLM_MODEL_PATH,
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
    )
    
    return tokenizer, llm
