# Simple Ultravox 사용법 예시

## 1. 데이터셋 준비

### 개별 태스크 데이터셋 생성
```bash
# 음성 전사 데이터셋
python create_dataset.py \
  --input_scp /path/to/train.scp \
  --output_json ./data/train_transcription.json \
  --task transcription

# 음성 QA 데이터셋
python create_dataset.py \
  --input_scp /path/to/qa.scp \
  --output_json ./data/train_qa.json \
  --task qa

# 음성 정보 QA 데이터셋
python create_dataset.py \
  --input_scp /path/to/audio_info_qa.scp \
  --output_json ./data/train_audio_info_qa.json \
  --task audio_info_qa
```

### 멀티태스크 데이터셋 생성 (권장)
```bash
# 여러 태스크의 데이터를 하나의 파일로 합치기
cat ./data/train_transcription.json ./data/train_qa.json ./data/train_audio_info_qa.json > ./data/train_multitask.json
```

### CSV 파일에서 데이터셋 생성
```bash
# CSV에 WAV 경로가 있는 경우
python create_dataset.py \
  --input_csv /path/to/data.csv \
  --output_json ./data/train_dataset.json \
  --task transcription

# CSV에 WAV 경로가 없고 별도 디렉토리에 있는 경우
python create_dataset.py \
  --input_csv /path/to/data.csv \
  --wav_dir /path/to/wav_files \
  --output_json ./data/train_dataset.json \
  --task qa
```

## 2. 설정 파일 생성

### 기본 설정 파일 생성 (멀티태스크 학습)
```bash
python train.py --create_config configs/my_config.yaml \
  --text_model "meta-llama/Llama-3.1-8B-Instruct" \
  --audio_model "openai/whisper-medium" \
  --train_dataset "./data/train_multitask.json" \
  --eval_dataset "./data/eval_multitask.json" \
  --task null \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --max_steps 1000
```

### 설정 파일 수정 (멀티태스크 학습)
```yaml
# configs/my_config.yaml
model:
  text_model_name: "meta-llama/Llama-3.1-8B-Instruct"
  audio_model_name: "openai/whisper-medium"
  use_lora: true
  lora_r: 16
  lora_alpha: 32

training:
  batch_size: 4
  learning_rate: 1e-4
  max_steps: 1000
  output_dir: "./checkpoints"

data:
  train_dataset_path: "./data/train_multitask.json"
  eval_dataset_path: "./data/eval_multitask.json"
  task: null  # null이면 멀티태스크 학습
```

## 3. 모델 훈련

### 설정 파일로 훈련
```bash
python train.py --config configs/my_config.yaml
```

### 명령행 인자로 훈련 (멀티태스크)
```bash
python train.py \
  --text_model "meta-llama/Llama-3.1-8B-Instruct" \
  --audio_model "openai/whisper-medium" \
  --train_dataset "./data/train_multitask.json" \
  --eval_dataset "./data/eval_multitask.json" \
  --task null \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --max_steps 1000 \
  --output_dir "./checkpoints"
```

## 4. 추론

### 단일 파일 추론
```bash
python inference.py \
  --model_path "./checkpoints/best" \
  --audio_file "test.wav" \
  --task "transcription" \
  --text_input "다음 음성을 텍스트로 전사해주세요:" \
  --max_new_tokens 512 \
  --temperature 0.7
```

### QA 태스크 추론
```bash
python inference.py \
  --model_path "./checkpoints/best" \
  --audio_file "question.wav" \
  --task "qa" \
  --text_input "다음 질문에 답해주세요:" \
  --system_prompt "당신은 도움이 되는 AI 어시스턴트입니다."
```

### 음성 정보 QA 태스크 추론
```bash
python inference.py \
  --model_path "./checkpoints/best" \
  --audio_file "speech.wav" \
  --task "audio_info_qa" \
  --text_input "이 음성에서 언급된 시간은 언제인가요?" \
  --system_prompt "당신은 음성 정보를 분석하는 전문가입니다."
```

## 5. 웹 인터페이스

### Streamlit 앱 실행
```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하여 웹 인터페이스를 사용할 수 있습니다.

## 6. 데이터셋 형식 예시

### 멀티태스크 데이터셋 (권장)
```json
{"audio_filepath": "/path/to/audio1.wav", "text": "", "output": "안녕하세요, 오늘 날씨가 정말 좋네요.", "task": "transcription", "duration": 3.2}
{"audio_filepath": "/path/to/question1.wav", "text": "환자의 혈압이 높을 때 어떤 조치를 취해야 하나요?", "output": "고혈압 환자의 경우 먼저 혈압을 재측정하고...", "task": "qa", "duration": 4.1}
{"audio_filepath": "/path/to/speech1.wav", "text": "이 음성에서 언급된 시간은 언제인가요?", "output": "오후 2시에 회의가 있다고 언급했습니다.", "task": "audio_info_qa", "duration": 2.9}
{"audio_filepath": "/path/to/medical1.wav", "text": "다음 의료 관련 질문에 답해주세요.", "output": "고혈압은 혈압이 140/90mmHg 이상일 때를 말합니다...", "task": "medical_qa", "duration": 3.5}
```

### 개별 태스크 데이터셋
```json
# 전사 데이터셋
{"audio_filepath": "/path/to/audio1.wav", "text": "", "output": "안녕하세요, 오늘 날씨가 정말 좋네요.", "task": "transcription", "duration": 3.2}

# QA 데이터셋
{"audio_filepath": "/path/to/question1.wav", "text": "환자의 혈압이 높을 때 어떤 조치를 취해야 하나요?", "output": "고혈압 환자의 경우 먼저 혈압을 재측정하고...", "task": "qa", "duration": 4.1}

# 음성 정보 QA 데이터셋
{"audio_filepath": "/path/to/speech1.wav", "text": "이 음성에서 언급된 시간은 언제인가요?", "output": "오후 2시에 회의가 있다고 언급했습니다.", "task": "audio_info_qa", "duration": 2.9}
```

## 7. 고급 설정

### LoRA 설정 조정
```yaml
model:
  use_lora: true
  lora_r: 32          # 더 높은 랭크 = 더 많은 파라미터
  lora_alpha: 64      # 스케일링 팩터
  lora_dropout: 0.1   # 드롭아웃
  lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### 프로젝터 설정 조정
```yaml
model:
  projector_hidden_size: 2048    # 더 큰 히든 크기
  projector_output_size: 4096    # LLM 임베딩 크기와 맞춤
  projector_dropout: 0.2         # 더 높은 드롭아웃
```

### 훈련 설정 조정
```yaml
training:
  batch_size: 8                  # 더 큰 배치 크기
  gradient_accumulation_steps: 2 # 그래디언트 누적
  learning_rate: 5e-5            # 더 낮은 학습률
  warmup_steps: 200              # 더 긴 워밍업
  max_steps: 5000                # 더 많은 스텝
```

## 8. 멀티태스크 학습의 장점

### 일반적인 LLM 훈련 방식
- **단일 태스크 학습**: 각 태스크별로 별도 모델 훈련
- **멀티태스크 학습**: 모든 태스크를 하나의 모델로 훈련 (권장)

### 멀티태스크 학습의 장점
1. **더 나은 일반화**: 여러 태스크를 동시에 학습하여 더 강건한 모델
2. **효율적인 학습**: 하나의 모델로 여러 태스크 수행 가능
3. **태스크 간 지식 공유**: 한 태스크에서 학습한 지식이 다른 태스크에 도움
4. **실용성**: 배포 시 하나의 모델로 모든 태스크 처리

### 데이터셋 구성 권장사항
- 각 태스크별로 균형잡힌 데이터 비율 유지
- 예: transcription 40%, qa 30%, audio_info_qa 20%, medical_qa 10%
- 데이터 품질이 양보다 중요

## 9. 문제 해결

### 메모리 부족
- 배치 크기를 줄이세요: `batch_size: 2`
- 그래디언트 누적을 사용하세요: `gradient_accumulation_steps: 4`
- 더 작은 모델을 사용하세요: `text_model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"`

### 훈련이 느림
- 더 많은 워커를 사용하세요: `num_workers: 8`
- FP16을 사용하세요: `fp16: true`
- 더 큰 배치 크기를 사용하세요

### 오디오 파일 오류
- 오디오 파일 경로가 올바른지 확인하세요
- 오디오 파일이 손상되지 않았는지 확인하세요
- 지원되는 형식인지 확인하세요: WAV, MP3, FLAC, M4A
