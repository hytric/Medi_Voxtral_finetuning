# Multimodel - Voxtral 멀티태스크 학습 및 추론

Voxtral 모델을 사용하여 ASR(음성인식), SQA(음성 질의응답), TQA(텍스트 질의응답) 세 가지 태스크를 동시에 학습하는 멀티모달 프로젝트입니다.

## 📁 파일 구조

```
Multimodel/
├── train_multitask_voxtral.py        # 멀티태스크 학습 메인 스크립트
├── train_multitask_voxtral_debug.py  # 디버그 버전 학습 스크립트
├── infer_voxtral_mt.py               # 멀티태스크 추론 및 평가
└── README.md                         # 이 파일
```

## 🎯 주요 기능

### 1. 멀티태스크 학습 (train_multitask_voxtral.py)

**핵심 기능:**
- 3가지 태스크 동시 학습: ASR, SQA, TQA
- 태스크별 가중 샘플링으로 균형 잡힌 학습
- Whisper 인코더 고정, 프로젝터 + LLM LoRA 파인튜닝
- 실시간 성능 모니터링 및 메트릭 계산

**멀티태스크 아키텍처:**
```
[Audio] → Whisper Encoder (Frozen)
         ↓
       Projector (LoRA)
         ↓
       LLM Layers (LoRA) → [Text Output]
```

**주요 처리 과정:**
1. **태스크 감지:** 데이터 형식에 따른 자동 태스크 분류
2. **템플릿 적용:** ASR용 전사 템플릿 vs 일반 Chat-QA 템플릿
3. **가중 샘플링:** 태스크별 데이터 불균형 해결
4. **통합 학습:** 단일 모델로 모든 태스크 처리
5. **성능 평가:** 태스크별 최적 메트릭 적용

**실행 방법:**
```bash
python train_multitask_voxtral.py \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --train_jsonl data/ASR.jsonl,data/SQA.jsonl,data/TQA.jsonl \
  --val_jsonl data/ASR.dev.jsonl,data/SQA.dev.jsonl,data/TQA.dev.jsonl \
  --output_dir ckpt_voxtral_mt \
  --per_device_bsz 2 \
  --grad_accum 16 \
  --epochs 3 \
  --max_len 1280
```

**고급 설정:**
```bash
python train_multitask_voxtral.py \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --train_jsonl data/ASR.jsonl,data/SQA.jsonl,data/TQA.jsonl \
  --val_jsonl data/ASR.dev.jsonl,data/SQA.dev.jsonl,data/TQA.dev.jsonl \
  --output_dir ckpt_voxtral_mt \
  --per_device_bsz 2 \
  --grad_accum 16 \
  --max_len 1280 \
  --eval_max_new_tokens 256 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --task_weights 1.0,1.5,1.2 \
  --bf16 \
  --tf32 \
  --grad_ckpt \
  --freeze_whisper
```

### 2. 멀티태스크 추론 (infer_voxtral_mt.py)

**핵심 기능:**
- 학습된 모델로 3가지 태스크 통합 평가
- 태스크별 최적 메트릭으로 성능 측정
- 배치 처리 및 결과 분석

**평가 메트릭:**
- **ASR:** CER (Character Error Rate)
- **SQA/TQA:** ROUGE-L F1 Score
- **전체:** 태스크별 가중 평균 성능

**주요 처리 과정:**
1. **모델 로딩:** 베이스 모델 + LoRA 어댑터 결합
2. **태스크 분류:** 입력 데이터의 태스크 자동 감지
3. **템플릿 적용:** 태스크별 적절한 프롬프트 템플릿 사용
4. **배치 추론:** 효율적인 배치 처리로 대량 평가
5. **성능 집계:** 태스크별 세부 결과 및 전체 요약

**실행 방법:**
```bash
python infer_voxtral_mt.py \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --lora_dir ckpt_voxtral_mt \
  --val_jsonl data/ASR.dev.jsonl,data/SQA.dev.jsonl,data/TQA.dev.jsonl \
  --per_device_bsz 4 \
  --max_len 1280 \
  --gen_max_new_tokens 256
```

## 🔧 기술적 특징

### 멀티태스크 학습 전략

#### 1. 태스크 인식 시스템
```python
def detect_task(data):
    if 'audio_path' and 'transcript': return 'asr'
    elif 'audio_path' and 'question': return 'sqa' 
    elif 'instruction' and 'output': return 'tqa'
```

#### 2. 템플릿 시스템
- **ASR 템플릿:** 전사 전용 최적화된 프롬프트
- **Chat-QA 템플릿:** 질의응답용 범용 대화 형식
- **동적 전환:** 태스크에 따른 자동 템플릿 선택

#### 3. 가중 샘플링
```python
# 태스크별 데이터 불균형 해결
task_weights = {'asr': 1.0, 'sqa': 1.5, 'tqa': 1.2}
weighted_sampler = WeightedRandomSampler(weights, num_samples)
```

### 성능 최적화

#### 메모리 효율성
- **선택적 고정:** Whisper 인코더만 고정하여 메모리 절약
- **LoRA 적용:** 프로젝터와 LLM에만 어댑터 적용
- **Mixed Precision:** bfloat16으로 메모리 및 속도 최적화
- **Gradient Checkpointing:** 메모리 vs 계산 트레이드오프

#### 학습 안정성
- **Gradient Clipping:** 그라디언트 폭발 방지
- **Label Smoothing:** 오버피팅 완화
- **Learning Rate Scheduling:** 코사인 스케줄링으로 안정적 수렴
- **Early Stopping:** 검증 성능 기반 자동 중단

#### 실시간 모니터링
- **TensorBoard 통합:** 태스크별 세부 메트릭 시각화
- **GPU 모니터링:** 메모리, 사용률, 전력 소비 추적
- **Preview Generation:** 학습 중 품질 실시간 확인
- **태스크별 성능:** ASR CER, QA ROUGE-L 분리 추적

## 📊 성능 지표

### 태스크별 평가 메트릭

#### ASR (음성인식)
- **CER (Character Error Rate):** 문자 단위 오류율
- **WER (Word Error Rate):** 단어 단위 오류율 (선택적)
- **편집 거리:** Levenshtein distance 기반

#### SQA/TQA (질의응답)
- **ROUGE-L F1:** 최장 공통 부분수열 기반 품질
- **Exact Match:** 정확한 일치 비율 (선택적)
- **BLEU Score:** N-gram 기반 품질 측정 (선택적)

#### 통합 성능
- **Macro Average:** 태스크별 평균 성능
- **Weighted Average:** 데이터셋 크기 고려 가중 평균
- **Task Balance:** 태스크간 성능 균형 지표

### 하드웨어 효율성
- **Throughput:** 초당 처리 샘플 수
- **Memory Usage:** 피크 GPU 메모리 사용량
- **Power Consumption:** GPU 전력 소비량
- **Training Speed:** 에포크당 소요 시간

## 🚀 완전한 멀티태스크 파이프라인

### 1단계: 데이터 준비
```bash
# 3가지 태스크 데이터 준비
# ASR: {audio_path, transcript, language}
# SQA: {audio_path, question, answer}
# TQA: {instruction, input, output}

# 데이터 형식 검증
python -c "
import json
for file in ['ASR.jsonl', 'SQA.jsonl', 'TQA.jsonl']:
    with open(f'data/{file}') as f:
        sample = json.loads(next(f))
        print(f'{file}: {list(sample.keys())}')"
```

### 2단계: 멀티태스크 학습
```bash
python train_multitask_voxtral.py \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --train_jsonl data/ASR.jsonl,data/SQA.jsonl,data/TQA.jsonl \
  --val_jsonl data/ASR.dev.jsonl,data/SQA.dev.jsonl,data/TQA.dev.jsonl \
  --output_dir ckpt_voxtral_mt \
  --per_device_bsz 2 \
  --grad_accum 16 \
  --max_len 1280 \
  --eval_max_new_tokens 256 \
  --lora_r 16 \
  --lora_alpha 32 \
  --task_weights 1.0,1.5,1.2 \
  --bf16 \
  --freeze_whisper \
  --epochs 5
```

### 3단계: 통합 평가
```bash
python infer_voxtral_mt.py \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --lora_dir ckpt_voxtral_mt \
  --val_jsonl data/ASR.dev.jsonl,data/SQA.dev.jsonl,data/TQA.dev.jsonl \
  --per_device_bsz 4 \
  --max_len 1280 \
  --gen_max_new_tokens 256 \
  --bf16
```

## 🎯 사용 사례

### 의료 도메인 멀티모달 AI
1. **음성 전사:** 의료진-환자 대화 자동 기록
2. **음성 질의응답:** 환자 질문에 대한 음성 기반 답변
3. **텍스트 질의응답:** 의료 지식 기반 텍스트 상담

### 교육 분야 응용
1. **강의 전사:** 실시간 강의 내용 텍스트 변환
2. **학습 지원:** 음성/텍스트 질문에 대한 자동 답변
3. **다국어 지원:** 한국어 교육 콘텐츠 특화

### 고객 서비스
1. **콜센터 자동화:** 음성 문의 자동 처리
2. **FAQ 시스템:** 음성/텍스트 기반 FAQ 자동 응답
3. **상담 기록:** 고객 상담 내용 자동 정리

## 💡 고급 활용법

### 커스텀 태스크 가중치
```python
# 태스크별 중요도에 따른 가중치 조정
task_weights = {
    'asr': 1.0,    # 기본 중요도
    'sqa': 1.5,    # 음성 QA 강화
    'tqa': 1.2     # 텍스트 QA 약간 강화
}
```

### 동적 배치 구성
```python
# 태스크별 최적 배치 크기
batch_config = {
    'asr': 4,      # 오디오 처리로 메모리 많이 사용
    'sqa': 2,      # 오디오 + 텍스트 처리
    'tqa': 8       # 텍스트만 처리하므로 큰 배치 가능
}
```

### 성능 모니터링 대시보드
TensorBoard를 통한 실시간 모니터링:
- 태스크별 Loss 곡선
- 태스크별 성능 메트릭 (CER, ROUGE-L)
- GPU 메모리 및 사용률
- 학습 속도 및 처리량
- 생성 품질 샘플 (태스크별)

### 점진적 학습 전략
1. **1단계:** TQA만으로 언어 이해 능력 확보
2. **2단계:** ASR 추가로 음성-텍스트 매핑 학습
3. **3단계:** SQA 추가로 멀티모달 추론 완성

## 🔬 실험 및 분석

### A/B 테스트 프레임워크
- 단일 태스크 vs 멀티태스크 성능 비교
- 태스크별 데이터 비율 최적화
- LoRA 설정별 성능 분석

### 에러 분석
- 태스크별 실패 케이스 분석
- 크로스 태스크 간섭 효과 측정
- 도메인별 성능 차이 분석

### 확장성 테스트
- 새로운 태스크 추가 실험
- 다국어 확장 가능성
- 도메인 적응 효과 측정
