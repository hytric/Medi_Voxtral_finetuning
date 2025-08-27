# Simple Ultravox - 음성-언어 멀티모달 모델

Ultravox의 복잡한 구조를 단순화하여 개인 데이터셋 사용을 쉽게 만든 음성-언어 멀티모달 모델입니다.

## 주요 특징

- **단순한 데이터셋 형식**: `audio`, `text`, `output` 필드로 구성
- **템플릿 기반 학습**: 다양한 태스크를 템플릿으로 정의
- **LLM 미세조정 지원**: LoRA를 통한 효율적인 LLM 학습
- **어댑터 학습**: 오디오-텍스트 연결을 위한 어댑터 학습
- **개인 데이터셋 쉬운 사용**: JSON Lines 형식으로 간단한 데이터셋 등록

## 설치

```bash
pip install -r requirements.txt
```

## 데이터셋 형식

### 기본 형식 (JSON Lines)
```json
{"audio_filepath": "/path/to/audio.wav", "text": "입력 텍스트", "output": "출력 텍스트", "task": "transcription"}
{"audio_filepath": "/path/to/audio.wav", "text": "질문", "output": "답변", "task": "qa"}
{"audio_filepath": "/path/to/audio.wav", "text": "음성 정보 질문", "output": "답변", "task": "audio_info_qa"}
```

### 지원하는 태스크
- `transcription`: 음성 → 텍스트 전사
- `qa`: 음성 질문 → 텍스트 답변
- `audio_info_qa`: 음성 정보 + 텍스트 질문 → 답변
- `medical_qa`: 의료 관련 음성 질문 → 답변

## 사용법

### 1. 데이터셋 준비
```bash
python create_dataset.py --input_csv your_data.csv --output_json your_dataset.json
```

### 2. 훈련 설정
```yaml
# config.yaml
model:
  text_model: "meta-llama/Llama-3.1-8B-Instruct"
  audio_model: "openai/whisper-medium"
  
training:
  batch_size: 8
  learning_rate: 1e-4
  max_steps: 1000
  
data:
  train_dataset: "your_dataset.json"
  task: null  # null이면 멀티태스크 학습, 특정 태스크명을 지정하면 단일 태스크 학습
```

### 3. 훈련 실행
```bash
python train.py --config config.yaml
```

### 4. 추론
```bash
python inference.py --model_path ./checkpoints/best --audio_file test.wav
```

## 구조

```
SpeechLLM/
├── data/
│   ├── dataset.py          # 데이터셋 로딩
│   └── templates.py        # 템플릿 정의
├── model/
│   ├── ultravox.py         # 메인 모델
│   └── config.py           # 모델 설정
├── training/
│   ├── trainer.py          # 훈련 로직
│   └── config.py           # 훈련 설정
├── utils/
│   ├── audio.py            # 오디오 처리
│   └── text.py             # 텍스트 처리
├── create_dataset.py       # 데이터셋 생성
├── train.py               # 훈련 스크립트
├── inference.py           # 추론 스크립트
└── app.py                 # 웹 인터페이스
```

## 예시

### 의료 음성 QA 데이터셋
```json
{"audio_filepath": "medical_001.wav", "text": "환자의 혈압이 높을 때 어떤 조치를 취해야 하나요?", "output": "고혈압 환자의 경우 먼저 혈압을 재측정하고...", "task": "qa"}
```

### 음성 전사 데이터셋
```json
{"audio_filepath": "speech_001.wav", "text": "", "output": "안녕하세요, 오늘 날씨가 정말 좋네요.", "task": "transcription"}
```

### 음성 정보 QA 데이터셋
```json
{"audio_filepath": "speech_001.wav", "text": "이 음성에서 언급된 시간은 언제인가요?", "output": "오후 2시에 회의가 있다고 언급했습니다.", "task": "audio_info_qa"}
```

## 라이선스

MIT License
