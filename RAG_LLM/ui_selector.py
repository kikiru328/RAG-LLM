import streamlit as st
import importlib
from utils.sessions import initialize_session  # 공통 세션 초기화

# Streamlit 앱 이름
st.set_page_config(page_title="ChatBot", layout="wide")
st.title("ChatBot: 모드 선택")

# 선택 UI
mode_key = st.sidebar.selectbox("해당 채팅 모드를 선택하세요", [
    "chatbot",
    "debug",
    "ollama-chat (previous)"
])

# 모드에 따른 main() 동적 실행 함수
@st.cache_resource(show_spinner=False)
def get_interface_module(mode_key: str):
    if mode_key == "chatbot":
        return importlib.import_module("apps.chats.interfaces.chatbot")
    elif mode_key == "debug":
        return importlib.import_module("apps.chats.interfaces.debug")
    elif mode_key == "ollama-chat":
        return importlib.import_module("apps.ollama_chat.interfaces.ollama_chat")
    return None

# 모드 문자열 정리
# mode_key = mode.split()[1]  # e.g.,

# 인터페이스 가져오기
interface_module = get_interface_module(mode_key)

# 초기화
initialize_session()

# 실행
if interface_module:
    interface_module.main()