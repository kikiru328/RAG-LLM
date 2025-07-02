import streamlit as st
from apps.chats.models.vllm_models import load_qna_model
from apps.chats.prompts.classifier import classifier_system_prompt
from apps.chats.types.prompt import PromptRequest

from apps.chats.services.filter_rules import filtering_by_rules
from apps.chats.services.RAG import run_retriever
from apps.chats.services.llm_generator import generate_llm_response
from apps.chats.services.wordcloud_service import is_wordcloud_request, handle_wordcloud_request

from utils.sessions import initialize_session, show_messages, add_message
from utils.logger import logger

def main():
    st.title("Chat-bot")
    
    initialize_session()
    show_messages()
    
    if "tokenizer" not in st.session_state or "llm" not in st.session_state:
        with st.empty():
            st.markdown("🚀 **모델을 준비 중입니다.**")
            with st.spinner("LLM 모델을 초기 로딩 중입니다."):
                tokenizer, llm = load_qna_model()
                st.session_state.tokenizer = tokenizer
                st.session_state.llm = llm
            st.success("LLM 모델 초기화 완료")
    
    user_input = st.chat_input("궁금한 내용을 물어보세요.")
    
    if not user_input:
        return
    
    st.chat_message("user").write(user_input)
    add_message("user", user_input)
    
    try:
        # wordcloud service
        if is_wordcloud_request(user_input=user_input):
            with st.spinner("워드클라우드 생성 중"):
                handle_wordcloud_request(role="assistant")
            st.success("워드클라우드 생성 완료")
            return
        
        # filtering by rules (passed, reason, response, rule_type)
        filtered_result = filtering_by_rules(user_input=user_input)

        if not filtered_result.passed: # filtered: No RAG, No LLM
            if filtered_result.response: # 답변이 있을 시
                st.chat_message("assistant").write(response)
                add_message("assistant", response)
            st.warning(f"[오류] {reason}") # 차단 내용 설명
            
            return # 답변 끝
        
        # classifier
        classification_result = classifier_system_prompt(user_input=user_input)
        
        # RAG
        docs, context = run_retriever(user_input=user_input,
                                      classification_result=classification_result)
        
        # LLM
        tokenizer, llm = load_qna_model() # load model and save in cache
        
        request = PromptRequest(
            classification_result=classification_result,
            user_input=user_input,
            context=context,
            tokenizer=tokenizer,
            llm=llm
        )
        
        prompt, answer = generate_llm_response(request=request, docs=docs)
        st.chat_message("assistant").markdown(answer) # answer format in markdown
        add_message("assistant", answer)
        
    except Exception as e:
        logger.error(f"Error during debug response: {e}", exc_info=True)
        st.error("오류가 발생했습니다.")
        
if __name__ == "__main__":
    main()        
            