import streamlit as st
from typing import Optional
from analysis.wordcloud import WordCloudGenerator
from analysis.utils import get_message_text
from utils.sessions import add_message

def is_wordcloud_request(user_input: str) -> bool:
    
    """Check User_input requests for wordcloud?"""
    
    if not user_input: # 없을 땐 False
        return False
    
    keywords: list[str] = ["그려줘", "그려", "만들어줘", "생성해줘", "작성해줘"]
    normalized_input: str = user_input.strip().lower()
    contains_wordcloud_request: bool = "워드클라우드" in normalized_input
    contains_keywords: bool = any(
        keyword in normalized_input for keyword in keywords
    )
    
    return contains_wordcloud_request and contains_keywords    
    
def handle_wordcloud_request(role: Optional[str]) -> None:
    """ Create Wordlcoud image & Handle UI """
    
    text = get_message_text(role=role)
    if not text.strip():
        notice = "이전에 제가 대답한 내용이 없어서 워드클라우드를 생성할 수 없습니다."
        st.chat_message("assistant").write(notice)
        add_message("assistant", notice)
        return
    
    wc_generator = WordCloudGenerator()
    image_buffer, image_base64 = wc_generator.generate_wordcloud(text=text)
    assistant_notice = "이전에 제가 대답한 내용을 기반으로 워드클라우드를 생성합니다."
    with st.chat_message("assistant"):
        st.write(assistant_notice)
        st.image(image_buffer, caption="워드클라우드 생성 결과")
        
    add_message("assistant", assistant_notice)
    add_message("assistant_image", image_base64)
        
    