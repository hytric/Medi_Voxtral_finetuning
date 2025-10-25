# SLLM - Speech-Language Large Model

Voxtral 기반 의료 도메인 멀티모달 AI 시스템. 음성인식(ASR), 음성 질의응답(SQA), 텍스트 질의응답(TQA) 지원.

## ✨ 주요 특징

- **3가지 태스크**: ASR, SQA, TQA 멀티태스크 학습
- **의료 특화**: 327K+ 의료 도메인 데이터셋
- **효율적 학습**: 4bit 양자화 + LoRA 어댑터
- **단계별 파이프라인**: LLM → ASR → 멀티태스크

## 📁 프로젝트 구조

```
SLLM/
├── LLM_only/          # Stage 1: 텍스트 사전학습
├── Adapter_only/      # Stage 2: ASR 어댑터 학습
├── Multimodel/        # Stage 3: 멀티태스크 학습
└── dataset/           # 데이터셋 (ASR 141K, SQA 16K, TQA 168K)
```

## 🔧 설치

```bash
# 필수 패키지 설치
pip install torch transformers>=4.40.0 peft>=0.10.0 bitsandbytes>=0.43.0
pip install accelerate tensorboard librosa soundfile
```

**요구사항**: NVIDIA GPU (24GB+ 권장), CUDA 11.8+, Python 3.8+

## 🚀 빠른 시작

### 1. 데이터 준비

```bash
# 데이터셋이 이미 준비되어 있다면 생략 가능
cd SLLM/dataset

# 데이터 품질 검증
python med_qa_dedup_normalize.py

# 데이터 통계 확인
python ../LLM_only/analyze_lengths.py --input TQA.jsonl
```

### 2. Stage 1: 텍스트 전용 사전학습

```bash
cd SLLM/LLM_only

# 환경 점검 및 자동 실행
python preflight_voxtral.py \
  --nproc 2 \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --train_jsonl ../dataset/TQA.jsonl \
  --val_jsonl ../dataset/val_TQA.jsonl \
  --out_dir ckpt_stage1_text \
  --autorun

# 또는 수동 실행 (2-GPU)
torchrun --standalone --nproc_per_node=2 \
  train_stage1_text_ddp.py \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --train_jsonl ../dataset/TQA.jsonl \
  --val_jsonl ../dataset/val_TQA.jsonl \
  --output_dir ckpt_stage1_text \
  --per_device_bsz 6 \
  --grad_accum 8 \
  --epochs 3
```

### 3. Stage 2: ASR 어댑터 학습

```bash
cd SLLM/Adapter_only

# 오디오 features 캐싱 (학습 속도 향상)
python build_cache.py

# ASR 어댑터 학습
python train_stage2_asr_adapter.py \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --train_jsonl ../dataset/ASR.jsonl \
  --val_jsonl ../dataset/val_ASR.jsonl \
  --output_dir ckpt_stage2_asr \
  --epochs 2 \
  --lr 1e-4 \
  --per_device_bsz 8 \
  --grad_accum 4

# ASR 평가
python inference_asr.py \
  --base_model mistralai/Voxtral-Mini-3B-2507 \
  --adapter_path ckpt_stage2_asr/lora_stage2_adapter \
  --test_jsonl ../dataset/val_ASR.jsonl \
  --output_path results.jsonl
```

### 4. Stage 3: 멀티태스크 통합 학습

```bash
cd SLLM/Multimodel

# 3개 태스크 동시 학습
python train_multitask_voxtral.py \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --train_jsonl ../dataset/ASR.jsonl,../dataset/SQA.jsonl,../dataset/TQA.jsonl \
  --val_jsonl ../dataset/val_ASR.jsonl,../dataset/val_SQA.jsonl,../dataset/val_TQA.jsonl \
  --output_dir ckpt_voxtral_mt \
  --per_device_bsz 2 \
  --grad_accum 16 \
  --epochs 3

# 멀티태스크 평가
python infer_voxtral_mt.py \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --lora_dir ckpt_voxtral_mt \
  --val_jsonl ../dataset/val_ASR.jsonl,../dataset/val_SQA.jsonl,../dataset/val_TQA.jsonl \
  --per_device_bsz 4
```

## 📚 학습 파이프라인

SLLM은 3단계 학습 파이프라인을 제공합니다:

### 파이프라인 개요

```
Stage 1 (LLM_only)
    ↓ 텍스트로 언어 이해 능력 확보
Stage 2 (Adapter_only)  
    ↓ 음성-텍스트 매핑 학습
Stage 3 (Multimodel)
    ↓ 멀티태스크 통합 학습
최종 모델
```

### 각 단계 상세

| 단계 | 목적 | 학습 대상 | 데이터 | 소요 시간 |
|------|------|----------|--------|----------|
| **Stage 1** | 도메인 언어 이해 | LLM LoRA | TQA (168K) | 6-8시간 (2 GPU) |
| **Stage 2** | 음성 인식 능력 | Projector LoRA | ASR (141K) | 3-4시간 (1 GPU) |
| **Stage 3** | 통합 멀티태스크 | Projector+LLM LoRA | ASR+SQA+TQA | 8-12시간 (2 GPU) |

### 추천 학습 전략

1. **빠른 프로토타이핑**: Stage 3만 실행 (데이터 적을 경우)
2. **최고 성능**: Stage 1 → 2 → 3 순차 실행 (충분한 데이터)
3. **ASR 전용**: Stage 2만 실행
4. **QA 전용**: Stage 1만 실행

## 📊 데이터셋

### 데이터 통계

| 태스크 | 학습 데이터 | 검증 데이터 | 도메인 | 출처 |
|--------|------------|------------|--------|------|
| **ASR** | 141,820개 | 2,145개 | 의료 상담 대화 | AI-Hub 비대면 진료 |
| **SQA** | 16,403개 | 1,823개 | 의료 질의응답 | 초거대 AI + TTS |
| **TQA** | 168,903개 | 18,767개 | 의료 백과사전 | 초거대 AI, 아산병원 |

### 데이터 형식

#### ASR 데이터
```json
{
  "audio_path": "path/to/audio.wav",
  "transcript": "가족 중에 암으로 돌아가신 분이 있나요?",
  "language": "ko"
}
```

#### SQA 데이터
```json
{
  "instruction": "나의 난청이 왜 생기는지 알고 싶어요.",
  "input": "",
  "audio_path": "path/to/question.wav",
  "output": "난청의 원인에는 전음성 난청, 감각신경성 난청..."
}
```

#### TQA 데이터
```json
{
  "instruction": "성병의 다양한 치료 방법에 대해 설명해주세요.",
  "input": "",
  "output": "성병은 감염의 원인에 따라 다양한 치료 방법을 사용합니다..."
}
```

### 데이터 생성 도구

프로젝트는 자체 데이터 생성 파이프라인을 포함합니다:

- **TTS 합성**: CosyVoice 기반 고품질 한국어 음성 생성
- **QA 생성**: LLM 기반 자동 질의응답 쌍 생성
- **데이터 정제**: 중복 제거, 정규화, 품질 필터링

자세한 내용은 [`SLLM/dataset/README.md`](SLLM/dataset/README.md)를 참조하세요.

## 🎯 성능 및 결과

### 평가 메트릭

| 태스크 | 메트릭 | 설명 |
|--------|--------|------|
| ASR | CER (Character Error Rate) | 문자 단위 오류율 (낮을수록 좋음) |
| ASR | WER (Word Error Rate) | 단어 단위 오류율 (낮을수록 좋음) |
| SQA/TQA | ROUGE-L F1 | 최장 공통 부분수열 기반 품질 (높을수록 좋음) |
| SQA/TQA | Distinct-N | N-gram 다양성 지표 (높을수록 좋음) |

### 예상 성능

실제 성능은 데이터 품질과 학습 설정에 따라 다를 수 있습니다:

- **ASR CER**: 5-10% (의료 도메인 한국어)
- **SQA ROUGE-L**: 0.35-0.45
- **TQA ROUGE-L**: 0.40-0.55

## 🛠️ 기술 스택

### 핵심 기술

- **베이스 모델**: Voxtral-Mini-3B-2507 (Mistral AI)
- **음성 인코더**: Whisper (OpenAI)
- **파인튜닝**: LoRA (Low-Rank Adaptation)
- **양자화**: 4bit bitsandbytes
- **분산 학습**: PyTorch DDP (Distributed Data Parallel)

### 최적화 기법

| 기법 | 설명 | 효과 |
|------|------|------|
| **4bit 양자화** | BitsAndBytesConfig로 모델 압축 | GPU 메모리 75% 절감 |
| **LoRA** | r=8~16, α=16~32 어댑터 | 파라미터 효율적 학습 |
| **Flash Attention** | 메모리 효율적 어텐션 | 학습 속도 30% 향상 |
| **Gradient Checkpointing** | 중간 활성화 재계산 | 메모리 추가 절약 |
| **Mixed Precision (bfloat16)** | 16bit 연산 | 속도 향상, 메모리 절감 |
| **캐시 시스템** | 오디오 features 사전 계산 | I/O 병목 제거 |

### 주요 라이브러리

```
transformers>=4.40.0      # Voxtral 모델 지원
peft>=0.10.0              # LoRA 구현
bitsandbytes>=0.43.0      # 4bit 양자화
torch>=2.0.0              # PyTorch
accelerate>=0.29.0        # 분산 학습
tensorboard               # 실시간 모니터링
```

## 📖 상세 문서

각 모듈의 상세 문서는 해당 디렉토리의 README를 참조하세요:

- **[LLM_only](SLLM/LLM_only/README.md)**: Stage 1 텍스트 전용 학습
- **[Adapter_only](SLLM/Adapter_only/README.md)**: Stage 2 ASR 어댑터 학습  
- **[Multimodel](SLLM/Multimodel/README.md)**: Stage 3 멀티태스크 학습
- **[Dataset](SLLM/dataset/README.md)**: 데이터셋 및 생성 도구

## 🐛 문제 해결

### 자주 발생하는 문제

#### 1. GPU 메모리 부족 (OOM)

```bash
# 배치 크기 감소
--per_device_bsz 2 --grad_accum 16

# Gradient Checkpointing 활성화
--grad_ckpt

# 더 작은 LoRA rank 사용
--lora_r 8 --lora_alpha 16
```

#### 2. 학습 속도 느림

```bash
# 캐시 시스템 사용 (ASR)
python build_cache.py

# DataLoader 워커 증가
--dataloader_num_workers 4

# Mixed Precision 활성화
--bf16 --tf32
```

#### 3. 호환성 문제

```bash
# 환경 점검 도구 실행
python SLLM/LLM_only/preflight_voxtral.py

# Transformers 버전 확인
pip install transformers>=4.40.0 --upgrade
```

## 🤝 기여 가이드

프로젝트 개선에 기여하고 싶다면:

1. **이슈 등록**: 버그 리포트나 기능 제안
2. **코드 기여**: Pull Request 환영
3. **데이터 기여**: 새로운 의료 데이터셋 추가
4. **문서 개선**: 오타 수정, 설명 추가

## 📝 라이선스

본 프로젝트는 연구 및 교육 목적으로 사용됩니다.

### 데이터 라이선스

- **AI-Hub 데이터**: AI-Hub 이용 약관 준수
- **초거대 AI 데이터**: 공개 데이터셋 라이선스 준수
- **아산병원 데이터**: 공개 건강정보 활용

### 의료 면책 조항

⚠️ **중요**: 본 모델은 연구 및 참고 목적으로만 사용되어야 합니다. 실제 의료 진단이나 치료에는 반드시 의료 전문가와 상담하시기 바랍니다.

## 📧 문의

- **개발팀 문의**: 프로젝트 관리자에게 연락
- **버그 리포트**: GitHub Issues 활용
- **기술 지원**: 각 모듈의 README 참조

## 🌟 인용

본 프로젝트를 연구에 사용하신다면 다음과 같이 인용해주세요:

```bibtex
@misc{sllm2024,
  title={SLLM: Speech-Language Large Model for Medical Domain},
  author={SLLM Team},
  year={2024},
  howpublished={\url{https://github.com/yourusername/SLLM}},
}
```

