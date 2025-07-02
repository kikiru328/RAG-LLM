### Test
import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

import streamlit as st
from apps.ollama_chat.services.generator import create_chain
from utils.sessions import initialize_session, show_messages, add_message

def main():
    st.title("Ollama bllossom_latest ChatBot")
    initialize_session()
    show_messages()
    
    user_input = st.chat_input("궁금한 내용을 물어보세요.")
    
    if user_input:
        st.chat_message("user").write(user_input)
        chain = create_chain()

        if not chain:
            return
        
        with st.chat_message("assistant"):
            container = st.empty()
            answer = ""
            
            for token in chain.stream(user_input):
                answer += token
                container.markdown(answer)
        
        add_message("user", user_input)
        add_message("assistant", answer)
        
if __name__ == "__main__":
    main()
