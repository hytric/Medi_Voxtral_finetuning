# Adapter_only - Voxtral ASR 어댑터 학습 및 추론

Voxtral 모델의 멀티모달 프로젝터 부분만 LoRA 어댑터로 학습하여 ASR(음성인식) 성능을 향상시키는 프로젝트입니다.

## 📁 파일 구조

```
Adapter_only/
├── train_stage2_asr_adapter.py    # ASR 어댑터 학습 메인 스크립트
├── inference_asr.py               # ASR 추론 및 평가 스크립트
├── prepare_dataset.ipynb          # 데이터셋 전처리 노트북
├── cut_audio.py                   # 오디오 길이 필터링 스크립트
├── build_cache.py                 # 학습 데이터 캐시 생성
├── manifest.ipynb                 # 매니페스트 파일 생성 노트북
└── README.md                      # 이 파일
```

## 🎯 주요 기능

### 1. ASR 어댑터 학습 (train_stage2_asr_adapter.py)

**핵심 기능:**
- Voxtral의 multi_modal_projector 레이어만 LoRA로 파인튜닝
- 4bit 양자화와 Flash Attention 최적화
- 캐시된 입력 features 사용으로 학습 속도 향상

**주요 처리 과정:**
1. **모델 준비:** Voxtral-Mini-3B-2507 모델을 4bit 양자화로 로드
2. **타겟 모듈 탐지:** multi_modal_projector 내 Linear 레이어 자동 탐지
3. **LoRA 구성:** r=8, alpha=16으로 어댑터 설정
4. **데이터 로딩:** 캐시된 input_features와 텍스트 레이블 사용
5. **학습 최적화:** FastTrainer로 커스텀 DataLoader 설정

**실행 방법:**
```bash
python train_stage2_asr_adapter.py \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --train_jsonl dataset/train_data.cached.jsonl \
  --output_dir ckpt_stage2_asr \
  --epochs 2 \
  --lr 1e-4 \
  --per_device_bsz 8 \
  --grad_accum 4
```

### 2. ASR 추론 및 평가 (inference_asr.py)

**핵심 기능:**
- 학습된 LoRA 어댑터를 사용한 음성 인식
- CER(Character Error Rate) 및 WER(Word Error Rate) 자동 계산
- 단일 파일 추론 및 배치 평가 지원

**주요 처리 과정:**
1. **모델 로딩:** 베이스 모델 + LoRA 어댑터 결합
2. **음성 전처리:** Voxtral processor로 오디오 인코딩
3. **텍스트 생성:** 빔 서치 없이 deterministic 디코딩
4. **성능 평가:** 편집 거리 기반 CER/WER 계산

**실행 방법:**
```bash
# 단일 파일 추론
python inference_asr.py \
  --base_model mistralai/Voxtral-Mini-3B-2507 \
  --adapter_path ckpt_stage2_asr/lora_stage2_adapter \
  --audio_path sample.wav

# 테스트 데이터셋 평가
python inference_asr.py \
  --base_model mistralai/Voxtral-Mini-3B-2507 \
  --adapter_path ckpt_stage2_asr/lora_stage2_adapter \
  --test_jsonl test_data.jsonl \
  --output_path results.jsonl
```

### 3. 데이터 전처리 (prepare_dataset.ipynb, cut_audio.py)

**데이터 전처리 파이프라인:**
1. **텍스트 정규화:** 괄호, 특수문자 필터링
2. **매니페스트 변환:** SCP 파일을 JSONL 형식으로 변환
3. **오디오 길이 분석:** 길이 분포 시각화 및 통계 계산
4. **길이 필터링:** 3-25초 범위의 오디오만 선별

**cut_audio.py 실행:**
```bash
python cut_audio.py  # 3-25초 범위로 필터링
```

### 4. 캐시 생성 (build_cache.py)

**캐시 시스템:**
- 학습 속도 향상을 위해 오디오 features를 미리 계산
- Voxtral processor로 input_features(80, T) 추출 후 저장
- 메모리 효율적인 배치 학습 지원

**실행 방법:**
```bash
python build_cache.py
# dataset/train_data.cached.jsonl 생성
# dataset/cache_stage2/ 폴더에 .pt 파일들 생성
```

## 🔧 기술적 특징

### 최적화 기법
- **4bit 양자화:** BitsAndBytesConfig로 메모리 사용량 대폭 감소
- **Flash Attention:** GPU 메모리 효율적인 어텐션 구현
- **LoRA 어댑터:** 파라미터 효율적 파인튜닝 (r=8, α=16)
- **Mixed Precision:** bfloat16으로 학습 안정성과 속도 향상

### 데이터 효율성
- **캐시 시스템:** 오디오 전처리 결과를 디스크에 저장
- **길이 기반 필터링:** 너무 짧거나 긴 오디오 제거
- **텍스트 정규화:** 의료 도메인 특화 전처리

### 평가 지표
- **CER (Character Error Rate):** 문자 단위 오류율
- **WER (Word Error Rate):** 단어 단위 오류율
- **편집 거리:** Levenshtein distance 기반 정확한 계산

## 📊 성능 최적화

### 메모리 최적화
- 4bit 양자화로 GPU 메모리 사용량 75% 감소
- Gradient checkpointing으로 추가 메모리 절약 가능
- 캐시된 features로 실시간 오디오 처리 부하 제거

### 학습 효율성
- FastTrainer로 커스텀 DataLoader 파라미터 최적화
- num_workers, prefetch_factor 튜닝으로 I/O 병목 해결
- TensorBoard 연동으로 실시간 학습 모니터링

## 🎯 사용 사례

이 프로젝트는 특히 다음과 같은 경우에 적합합니다:

1. **의료 도메인 ASR:** 의료진-환자 대화 전사
2. **한국어 특화:** 한국어 음성 인식 성능 향상
3. **리소스 제약:** 제한된 GPU 메모리 환경에서의 효율적 학습
4. **빠른 프로토타이핑:** 어댑터만 학습하여 빠른 실험 가능

## 🚀 시작하기

1. **환경 설정:**
   ```bash
   pip install transformers torch torchaudio peft bitsandbytes
   ```

2. **데이터 준비:**
   ```bash
   jupyter notebook prepare_dataset.ipynb  # 데이터 전처리
   python build_cache.py  # 캐시 생성
   ```

3. **모델 학습:**
   ```bash
   python train_stage2_asr_adapter.py
   ```

4. **모델 평가:**
   ```bash
   python inference_asr.py --adapter_path ckpt_stage2_asr/lora_stage2_adapter
   ```