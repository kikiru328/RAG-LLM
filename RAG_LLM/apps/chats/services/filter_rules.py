import json
from apps.legal_chat.types.filter import FilterResult
from utils.paths import RULE_FILTER_PATH

def load_filters() -> dict:
    """
    Load json for find keywords to filtering by rules
    """
    with open(RULE_FILTER_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
    
      
def filtering_by_rules(user_input: str) -> tuple[bool, str | None, str | None]:
    """
    filtering by rules
    
    user_input에 대해서 RAG 및 LLM 모델을 통해서 답변을 제공하지 않고,
    지정된 답변을 하게 끔 합니다.  
    
    e.g.) 가령 금지된 단어를 user_input에서 넘겼을 경우
    user_input: "blocked keywords"
    > RAG X
    > VLLM X
    > reason: "금지된 내용입니다." (log 혹은 debug 용)
    > response: "해당 질문은 처리할 수 없습니다." (client에 반응)
    
    따라서 해당 코드 혹은 json 파일을 수정하여
    관련되지 않은 질문을 1차적으로 선별 후 지정된 답변을 제공할 수 있습니다.
    
    차후 금지 목록이 아닌 다른 것들도 확장할 수 있습니다.
    """
    rules = load_filters()
    lowered_input = user_input.lower()
    
    for rule in rules:
        rule_type = rule.get("rule_type")
        
        for keyword in rule.get("keywords", []):
            if keyword in lowered_input:
                reason = rule.get("reason", "")
                response = rule.get("response", None)
                return FilterResult(
                    passed=False,
                    reason=reason,
                    response=response,
                    rule_type=rule_type
                )
    return FilterResult(passed=True)