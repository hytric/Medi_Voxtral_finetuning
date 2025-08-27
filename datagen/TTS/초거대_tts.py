import sys
sys.path.append('/shared/home/milab/users/kskim/CosyVoice/third_party/Matcha-TTS')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import pandas as pd
from tqdm import tqdm

# 모델 로드
cosyvoice = CosyVoice2(
    '/shared/home/milab/users/kskim/CosyVoice/pretrained_models/CosyVoice2-0.5B',
    load_jit=False, load_trt=False, load_vllm=False, fp16=False
)

# 입력 파일 경로
input_path = '/shared/home/milab/users/kskim/CosyVoice/dataset/AI_Hub_medQA.json'
kson_csv_path = "/shared/home/milab/users/kskim/CosyVoice/results/tts_data.csv"
output_dir = "/shared/home/milab/users/kskim/CosyVoice/results/초거대/wav"
output_csv_path = "/shared/home/milab/users/kskim/CosyVoice/results/초거대/manifest.csv"

# JSON 파일 로드
import json

# JSON 파일 열기
with open(input_path, 'r', encoding='utf-8') as file:
    # JSON 파일 읽고 파이썬 리스트로 변환
    data = json.load(file)

# instruction 리스트 추출 (JSON 구조에 맞게 수정)
instruction_list = []
original_data = []  # 원본 데이터 보존

for item in data:
    # disease_category가 '귀코목질환'인 경우만 필터링
    if item.get('disease_category', '') == '귀코목질환':
        if 'generated_text' in item:
            instruction_list.append(item['generated_text'])
            original_data.append(item)
        else:
            print(f"Warning: 'generated_text' 필드가 없는 항목 발견: {item}")

# 프롬프트 오디오 + 텍스트
kson_df = pd.read_csv(kson_csv_path, header=None)
prompt_audio_list = kson_df[0].tolist()
prompt_text_list = kson_df[1].tolist()

# 길이 맞추기
min_len = min(len(instruction_list), len(prompt_text_list), len(prompt_audio_list))
instruction_list = instruction_list[:min_len]
prompt_text_list = prompt_text_list[:min_len]
prompt_audio_list = prompt_audio_list[:min_len]

# 결과 저장 리스트
results = []

# TTS 변환 루프
for idx, (instruction, prompt_text, prompt_audio_path) in enumerate(tqdm(zip(instruction_list, prompt_text_list, prompt_audio_list))):
    # 참조 음성 로드
    prompt_audio_path = prompt_audio_path.replace('/home/nas4/DB/OLKAVS/preprocessed/', '/shared/home/milab/users/kskim/dataset/OLKAVS/')

    prompt_speech_16k = load_wav(prompt_audio_path, 16000)

    # zero-shot TTS
    for j in cosyvoice.inference_zero_shot(instruction, prompt_text, prompt_speech_16k,
                                           text_frontend=False, stream=False):
        # 파일명 생성 (인덱스 기반)
        output_filename = f"초거대_{idx}.wav"
        output_wav_path = os.path.join(output_dir, output_filename)
        
        # wav 저장
        torchaudio.save(output_wav_path, j['tts_speech'], cosyvoice.sample_rate)

        # 결과 저장 (원본 데이터 + 음성 파일 경로)
        original_item = original_data[idx]
        result_row = {
            'path': original_item.get('path', ''),
            'generated_text': original_item.get('generated_text', ''),
            'answer': original_item.get('answer', ''),
            'department': original_item.get('department', ''),
            'disease_name': original_item.get('disease_name', ''),
            'disease_category': original_item.get('disease_category', ''),
            'audio_path': output_wav_path
        }
        results.append(result_row)

        break  # stream=False이므로 첫 결과만 사용

    print(f"{idx} / {min_len} -> {idx/min_len*100}")

# 결과 CSV로 저장 (manifest)
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False, encoding='utf-8')

print("✅ 모든 TTS 변환 및 저장 완료. Manifest CSV 저장됨:", output_csv_path)
print(f"총 {len(results)}개의 음성 파일이 생성되었습니다.")
