# 🧠 모델 학습 스크립트 (`train.py`)

이 디렉토리는 GPT 기반 질의응답 모델을 학습하기 위한 스크립트와 데이터를 포함합니다.  
`train.py`는 **LoRA 기반 파인튜닝을 수행하는 스크립트**로, Q&A 데이터를 활용해 학습을 진행합니다.

## 📂 구성

- `train.py`: 모델 학습 전체 파이프라인 정의
  - Hugging Face Transformers 및 PEFT(LoRA) 기반
  - Flash Attention2, k-bit, gradient checkpointing 적용
- `dataset/`: 학습에 사용되는 JSONL 형식의 Q&A 데이터
- `outputs/`: 학습된 모델 출력 결과 저장 디렉토리

## 📝 학습 데이터 형식

```json
{
  "instruction": "질문 내용",
  "output": "GPT가 생성한 응답",
  "reference": "참고자료"
}
```

# 경로 설정
경로(BASE_MODEL, DATASET, new_model 등)는 실제 사용 환경에 맞게 수정해야 합니다.  
예를 들어, Docker 컨테이너, 서버, 또는 로컬 학습 환경 등 상황에 따라 절대경로 또는 상대경로를 지정하세요.
