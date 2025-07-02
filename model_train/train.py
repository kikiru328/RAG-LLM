# =========================== #
# 1. 라이브러리 및 GPU 설정
# =========================== #

import torch
print(torch.cuda.is_available())  # GPU 사용 가능 여부를 확인 (True/False 출력)
import os
from datasets import load_dataset  # 데이터셋 로드 기능 제공

# Hugging Face 및 관련 프레임워크 라이브러리
from transformers import (
    AutoModelForCausalLM,  # 사전 학습된 언어 모델 로드
    AutoTokenizer,         # 모델과 호환되는 토크나이저 로드
    TrainingArguments,     # 학습 매개변수 설정
)
from peft import (
    LoraConfig,            # LoRA 설정 클래스
    PeftModel,             # PEFT 모델 불러오기
    get_peft_model,        # 모델에 LoRA를 적용하는 함수
    prepare_model_for_kbit_training  # k-bit 학습 준비 함수
)
from trl import SFTTrainer  # Hugging Face 모델 미세 조정 도구


# =========================== #
# 2. 데이터셋 프롬프트 함수 설정
# =========================== #
# 프롬프트 형식 지정: 사용자 입력과 어시스턴트 출력을 학습에 맞게 포맷
def prompts(example):
    """
    학습용 데이터셋의 'instruction'과 'output', 'reference' 필드를 사용하여
    언어 모델이 학습할 수 있는 프롬프트 형식을 생성합니다.
    """
    prompt_list = []
    for i in range(len(example['instruction'])):
        # instruction(질문)과 output(답변)을 연결하여 완성된 프롬프트 생성
        #FIXME
        text = f"..."
        prompt_list.append(
f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>{example['instruction'][i]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{example['output'][i]}{text}<|eot_id|>"""
        )
    return prompt_list

# =========================== #
# 3. 데이터셋 및 경로 설정
# =========================== #
BASE_MODEL = "MLP-KTLim/llama-3-Korean-Bllossom-8B"  # 학습에 사용할 사전 훈련된 모델
DATASET = ""  # Q&A 데이터셋 경로
new_model = ""  # 최종 학습된 모델의 저장 이름

# 데이터셋 불러오기 (train split만 사용)
dataset = load_dataset(DATASET, split="train")

# =========================== #
# 4. GPU 및 데이터 타입 설정
# =========================== #
# GPU 사양에 따라 데이터 타입 설정
if torch.cuda.get_device_capability()[0] >= 8:
    # 최신 GPU에서 최적화된 flash attention 사용
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16  # bfloat16은 성능이 뛰어남
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16  # 구형 GPU 호환성

# =========================== #
# 5. LoRA 설정 (학습용)
# =========================== #
# LoRA 기법을 이용해 모델을 효율적으로 미세조정
lora_config = LoraConfig(
    r=8,  # LoRA 랭크 (모델 파라미터 수를 줄이는 역할)
    lora_alpha=32,  # LoRA scaling factor (스케일링 비율)
    lora_dropout=0.1,  # 드롭아웃 비율 (오버피팅 방지)
    target_modules=[  # LoRA를 적용할 특정 모듈 지정
        "q_proj", "o_proj", "k_proj", "v_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",  # 바이어스 학습 비활성화
    task_type="CAUSAL_LM",  # 작업 유형: 언어 모델 (Causal Language Modeling)
)

# =========================== #
# 6. 모델 및 토크나이저 설정
# =========================== #
# 사전 학습된 모델 불러오기
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")

# LoRA 적용 전 k-bit 학습 준비
model.gradient_checkpointing_enable()  # 메모리 최적화
model = prepare_model_for_kbit_training(model)  # k-bit 학습 준비
model = get_peft_model(model, lora_config)  # LoRA 적용

# 학습 가능한 파라미터 확인
model.print_trainable_parameters()

# 토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰 설정
tokenizer.padding_side = "right"  # 패딩 방향 설정

# =========================== #
# 7. 학습 설정 및 시작
# =========================== #
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="outputs",  # 학습 결과 저장 경로
        #num_train_epochs=1,  # 학습 에폭 수
        max_steps=300,  # 최대 학습 스텝 수 // check: 300 250318
        per_device_train_batch_size=16,  # 배치 크기 설정
        gradient_accumulation_steps=4,  # 그래디언트 누적 단계 수
        optim="paged_adamw_8bit",  # 8bit 최적화 알고리즘
        warmup_steps=0,  # 워밍업 스텝 비활성화
        learning_rate=2e-4,  # 학습률 설정
        fp16=True,  # 16비트 연산 활성화
        logging_steps=100,  # 로그 출력 간격
        push_to_hub=False,  # Hugging Face Hub 푸시 비활성화
        report_to='none',  # 로그 리포팅 비활성화
    ),
    peft_config=lora_config,  # LoRA 설정 전달
    formatting_func=prompts,  # 프롬프트 포맷 함수 적용
)

trainer.train()  # 모델 학습 시작

# =========================== #
# 8. 모델 저장 및 병합
# =========================== #
# 학습된 LoRA 어댑터 저장
ADAPTER_MODEL = "lora_adapter"
trainer.model.save_pretrained(ADAPTER_MODEL)

# 어댑터 모델과 기존 모델 병합
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto')
model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto')
model = model.merge_and_unload()  # LoRA 병합 후 기존 모델 반환
model.save_pretrained(new_model)  # 병합된 모델 저장
print(f"LoRA 학습이 완료되었습니다. 모델이 '{new_model}' 폴더에 저장되었습니다.")

# 원본 모델의 토크나이저 다시 불러오기
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# LoRA 병합된 모델 폴더에 저장
tokenizer.save_pretrained(new_model)

print(f"토크나이저가 {new_model} 경로에 저장되었습니다.")
