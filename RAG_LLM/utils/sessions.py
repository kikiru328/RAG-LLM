import streamlit as st
from langchain_core.messages.chat import ChatMessage
import base64
import io

def initialize_session():
    """initialize all conversation session"""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def show_messages():
    """Print previous chat history"""
    for message in st.session_state["messages"]:
        role = message.role
        content = message.content
        
        if role == "assistant_image":
            # image decoding
            image_buffer = io.BytesIO(base64.b64decode(content))
            st.chat_message("assistant").image(image_buffer)
        else:
            st.chat_message(role).write(content)


def add_message(role, message):
    """add message into session status"""
    st.session_state["messages"].append(
        ChatMessage(
            role=role,
            content=message,
        )
    )