import re
from vllm import SamplingParams
from apps.legal_chat.prompts.builder import generate_prompt
from apps.legal_chat.types.prompt import PromptRequest
from apps.legal_chat.services.formatter import format_response

def detail(request: PromptRequest):
    generated_prompt = generate_prompt(
        classification_result=request.classification_result,
        user_input=request.user_input,
        context=request.context,
        tokenizer=request.tokenizer
    )
    sampling = SamplingParams(
        stop_token_ids=[request.tokenizer.eos_token_id],
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
    )
    outputs = request.llm.generate(generated_prompt, sampling)
    answer = "".join([o.outputs[0].text for o in outputs])
    return generated_prompt, answer

def general(request: PromptRequest):
    generated_prompt = generate_prompt(
        classification_result=request.classification_result,
        user_input=request.user_input,
        context=request.context,
        tokenizer=request.tokenizer
    )
    sampling = SamplingParams(
        stop_token_ids=[request.tokenizer.eos_token_id],
        temperature=0.3,
        top_p=0.7,
        max_tokens=64,
    )
    outputs = request.llm.generate(generated_prompt, sampling)
    answer = "".join([o.outputs[0].text for o in outputs])
    
    section_pattern = r"관련 X:.*"
    answer = re.sub(section_pattern, "", answer).strip()
    
    if len(answer) < 5:
        answer = "질문과 관련된 정보를 찾을 수 없습니다. 다른 질문을 해주세요."
        
    return generated_prompt, answer

def generate_llm_response(request: PromptRequest, docs: list = None):
    if request.classification_result == "general":
        return general(request)
    
    prompt, answer = detail(request)
        
    if request.classification_result in ["domain_specific", "domain_general"]:
        answer = format_response(answer)
        
    return prompt, answer