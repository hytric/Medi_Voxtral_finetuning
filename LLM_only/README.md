# LLM_only - Voxtral 텍스트 전용 학습 및 추론

Voxtral 모델의 LLM 부분만을 텍스트 데이터로 학습하는 1단계 프리트레이닝 프로젝트입니다. 의료 QA 데이터를 사용하여 도메인 특화 언어 이해 능력을 향상시킵니다.

## 📁 파일 구조

```
LLM_only/
├── train_stage1_text_ddp.py       # DDP 분산 학습 메인 스크립트
├── infer_from_ckpt.py             # 체크포인트 추론 및 평가
├── preflight_voxtral.py           # 학습 전 환경 점검 및 자동 실행
├── analyze_lengths.py             # 텍스트 길이 분석
├── audit_voxtral_data.py          # 데이터 품질 검증
├── find_long_samples.py           # 긴 샘플 탐지
├── split_and_clean.py             # 데이터 분할 및 정제
└── README.md                      # 이 파일
```

## 🎯 주요 기능

### 1. 분산 학습 (train_stage1_text_ddp.py)

**핵심 기능:**
- 다중 GPU 분산 학습 (DDP) 지원
- LoRA 어댑터를 사용한 효율적 파인튜닝
- 실시간 성능 모니터링 및 메트릭 계산
- 동적 샘플링 및 길이 기반 배치 그룹핑

**주요 처리 과정:**
1. **모델 초기화:** Voxtral-Mini-3B-2507을 4bit 양자화로 로드
2. **LoRA 설정:** LLM 레이어에 r=16, α=32 어댑터 적용
3. **데이터 인코딩:** 미리 토큰화하여 학습 속도 향상
4. **분산 학습:** 다중 GPU 환경에서 안정적 학습
5. **성능 평가:** ROUGE-L, Distinct-N 등 다양한 지표 계산

**실행 방법:**
```bash
# 2-GPU 분산 학습
torchrun --standalone --nproc_per_node=2 \
  train_stage1_text_ddp.py \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --train_jsonl data/text_medqa.train.jsonl \
  --val_jsonl data/text_medqa.val.jsonl \
  --output_dir ckpt_stage1_text \
  --per_device_bsz 6 \
  --grad_accum 8 \
  --epochs 3
```

**고급 기능:**
- **Gradient Checkpointing:** 메모리 효율성 향상
- **Preview Generation:** 학습 중 샘플 생성으로 진행상황 확인
- **Dynamic Preview:** 온도, top-p 등 생성 파라미터 실시간 조정
- **TensorBoard 로깅:** 상세한 학습 메트릭 시각화

### 2. 체크포인트 추론 (infer_from_ckpt.py)

**핵심 기능:**
- 학습된 체크포인트 자동 탐지 및 로드
- 배치 추론 및 성능 평가
- 다양한 생성 파라미터 실험

**주요 처리 과정:**
1. **체크포인트 탐지:** 최신 checkpoint-N 또는 lora_stage1 폴더 자동 찾기
2. **모델 복원:** 베이스 모델 + LoRA 어댑터 결합
3. **배치 추론:** 효율적인 배치 처리로 대량 데이터 평가
4. **품질 평가:** Distinct-N, ROUGE 등 텍스트 품질 지표

**실행 방법:**
```bash
python infer_from_ckpt.py \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --output_dir ckpt_stage1_text \
  --jsonl data/text_medqa.val.jsonl \
  --max_new_tokens 256 \
  --temperature 0.7
```

### 3. 환경 점검 도구 (preflight_voxtral.py)

**핵심 기능:**
- 학습 전 시스템 환경 및 데이터 검증
- 자동 보정 및 실행 커맨드 생성
- 호환성 문제 사전 탐지 및 해결

**검증 항목:**
1. **시스템 검사:** GPU 메모리, CUDA 버전, 라이브러리 호환성
2. **데이터 검증:** 파일 존재성, 스키마 검사, 길이 분포 분석
3. **설정 최적화:** 배치 크기, 워커 수 등 자동 튜닝
4. **명령어 생성:** 검증된 파라미터로 안전한 실행 스크립트 제공

**실행 방법:**
```bash
python preflight_voxtral.py \
  --nproc 2 \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --train_jsonl data/text_medqa.train.jsonl \
  --val_jsonl data/text_medqa.val.jsonl \
  --out_dir ckpt_stage1_text \
  --autorun  # 검증 후 자동 학습 실행
```

### 4. 데이터 분석 도구들

#### analyze_lengths.py
- **목적:** 텍스트 길이 분포 분석 및 시각화
- **기능:** 토큰 길이 히스토그램, 통계 요약, 이상치 탐지

#### audit_voxtral_data.py
- **목적:** 데이터 품질 검증 및 오류 탐지
- **기능:** JSON 스키마 검증, 중복 제거, 손상된 레코드 식별

#### find_long_samples.py
- **목적:** 최대 길이를 초과하는 샘플 탐지
- **기능:** 긴 텍스트 샘플 리스트업 및 처리 방안 제시

#### split_and_clean.py
- **목적:** 데이터셋 분할 및 정제
- **기능:** 훈련/검증 분할, 품질 필터링, 형식 정규화

## 🔧 기술적 특징

### 분산 학습 최적화
- **DDP (DistributedDataParallel):** 다중 GPU 효율적 활용
- **Mixed Precision:** bfloat16으로 메모리 및 속도 최적화
- **Gradient Accumulation:** 효과적인 배치 크기 확장
- **Dynamic Batching:** 길이 기반 배치 그룹핑으로 패딩 최소화

### 메모리 효율성
- **4bit 양자화:** 메모리 사용량 대폭 감소
- **LoRA 어댑터:** 전체 파라미터의 일부만 업데이트
- **Gradient Checkpointing:** 메모리 vs 계산 트레이드오프
- **캐시된 인코딩:** 미리 토큰화로 실시간 처리 부하 제거

### 실시간 모니터링
- **TensorBoard 통합:** 상세한 학습 메트릭 시각화
- **GPU 사용률 추적:** 메모리 및 전력 소비 모니터링
- **Preview Generation:** 학습 중 품질 실시간 확인
- **Early Stopping:** 성능 수렴 자동 탐지

## 📊 성능 지표

### 텍스트 품질 평가
- **ROUGE-L:** 최장 공통 부분수열 기반 품질 측정
- **Distinct-N:** N-gram 다양성 지표
- **PPL (Perplexity):** 언어 모델 품질 측정
- **BLEU Score:** 기계 번역 품질 평가 (선택적)

### 학습 효율성 지표
- **Tokens/Second:** 처리 속도 측정
- **GPU Utilization:** 하드웨어 활용률
- **Memory Usage:** 피크 메모리 사용량
- **Convergence Speed:** 수렴 속도 분석

## 🚀 완전한 학습 파이프라인

### 1단계: 데이터 준비
```bash
# 데이터 분석
python analyze_lengths.py --input data/raw_medqa.jsonl

# 데이터 정제
python split_and_clean.py \
  --input data/raw_medqa.jsonl \
  --train_output data/text_medqa.train.jsonl \
  --val_output data/text_medqa.val.jsonl

# 품질 검증
python audit_voxtral_data.py --input data/text_medqa.train.jsonl
```

### 2단계: 환경 점검
```bash
python preflight_voxtral.py \
  --nproc 2 \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --train_jsonl data/text_medqa.train.jsonl \
  --val_jsonl data/text_medqa.val.jsonl \
  --out_dir ckpt_stage1_text
```

### 3단계: 분산 학습
```bash
torchrun --standalone --nproc_per_node=2 \
  train_stage1_text_ddp.py \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --train_jsonl data/text_medqa.train.jsonl \
  --val_jsonl data/text_medqa.val.jsonl \
  --output_dir ckpt_stage1_text \
  --per_device_bsz 6 \
  --grad_accum 8 \
  --epochs 3 \
  --lr 2e-4 \
  --warmup_steps 1500 \
  --bf16 \
  --tf32 \
  --grad_ckpt
```

### 4단계: 모델 평가
```bash
python infer_from_ckpt.py \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --output_dir ckpt_stage1_text \
  --jsonl data/text_medqa.val.jsonl \
  --batch_size 8 \
  --preview_rows 20
```

## 🎯 사용 사례

이 프로젝트는 다음과 같은 경우에 특히 유용합니다:

1. **도메인 적응:** 의료, 법률 등 전문 도메인 언어 모델 구축
2. **다국어 지원:** 한국어 등 특정 언어 성능 향상
3. **리소스 최적화:** 제한된 GPU 환경에서의 효율적 학습
4. **실험 환경:** 빠른 프로토타이핑 및 파라미터 튜닝

## 💡 고급 활용법

### 커스텀 데이터 형식
```json
{
  "instruction": "다음 증상에 대해 설명해주세요.",
  "input": "환자가 기침과 발열을 호소합니다.",
  "output": "기침과 발열은 호흡기 감염의 일반적인 증상입니다..."
}
```

### 하이퍼파라미터 튜닝
- **LoRA 설정:** r(8-64), α(16-128), dropout(0.05-0.1)
- **학습률:** 1e-5 ~ 5e-4, 코사인 스케줄링 권장
- **배치 크기:** GPU 메모리에 따라 4-32 조정
- **시퀀스 길이:** 1024-2048, 데이터에 따라 조정

### 모니터링 대시보드
TensorBoard를 통해 실시간으로 다음을 모니터링:
- Loss 곡선 (train/validation)
- GPU 메모리 사용률
- 생성 품질 샘플
- 학습 속도 및 처리량
