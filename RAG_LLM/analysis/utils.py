import streamlit as st
from typing import Optional

def get_message_by_role(role: Optional[str] = None) -> list:
    """ 
    collect all messages by role to anlysis
    if not role, collect all messages (user & assistant)
    """
    all_messages = st.session_state["messages"]
    
    if role not in ("user", "assistant", None):
        raise ValueError("role must in [user, assistant, None]")
    
    result = []
    for message in all_messages:
        if (role is None) or (message.role == role):
            result.append(message.content)
            
    return result

def get_message_text(role: Optional[str] = None) -> str:
    """ 
    collect all messages by role as string to make Wordcloud
    if not role, collect all messages (user & assistant)
    """
    return " ".join(get_message_by_role(role=role))
    