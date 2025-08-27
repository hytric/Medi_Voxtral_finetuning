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
# cosyvoice = CosyVoice2(
#     '/shared/home/milab/users/kskim/CosyVoice/pretrained_models/CosyVoice2-0.5B',
#     load_jit=False, load_trt=False, load_vllm=False, fp16=False
# )

# 입력 파일 경로
input_csv_path = '/shared/home/milab/users/kskim/CosyVoice/dataset/Asan-AMC-Healthinfo.csv'
kson_csv_path = "/shared/home/milab/users/kskim/CosyVoice/results/tts_data.csv"
output_dir = "/shared/home/milab/users/kskim/CosyVoice/results/Asan/wav"
output_csv_path = "/shared/home/milab/users/kskim/CosyVoice/results/Asan/manifest.csv"

# 입력 CSV 로드 (instruction 컬럼 포함)
input_df = pd.read_csv(input_csv_path)
instruction_list = input_df['instruction'].tolist()


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
start = 5205

# TTS 변환 루프
for idx, (instruction, prompt_text, prompt_audio_path) in enumerate(tqdm(zip(instruction_list, prompt_text_list, prompt_audio_list))):
    if start < idx:
        break
    # 참조 음성 로드
    prompt_audio_path = prompt_audio_path.replace('/home/nas4/DB/OLKAVS/preprocessed/', '/shared/home/milab/users/kskim/dataset/OLKAVS/')

    prompt_speech_16k = load_wav(prompt_audio_path, 16000)

    # # zero-shot TTS
    # for j in cosyvoice.inference_zero_shot(instruction, prompt_text, prompt_speech_16k,
    #                                        text_frontend=False, stream=False):
        # # 파일명 생성 (인덱스 기반)
        # output_filename = f"asan_{idx}.wav"
        # output_wav_path = os.path.join(output_dir, output_filename)

        # break  # stream=False이므로 첫 결과만 사용
        # wav 저장
        # torchaudio.save(output_wav_path, j['tts_speech'], cosyvoice.sample_rate)

            # 파일명 생성 (인덱스 기반)
    output_filename = f"asan_{idx}.wav"
    output_wav_path = os.path.join(output_dir, output_filename)
    
    # 결과 저장 (원본 데이터 + 음성 파일 경로)
    result_row = input_df.iloc[idx].to_dict()
    result_row['audio_path'] = output_wav_path
    results.append(result_row)

    # print(f"{idx} / {min_len} -> {idx/min_len*100}")

# 결과 CSV로 저장 (manifest)
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False, encoding='utf-8')

print("✅ 모든 TTS 변환 및 저장 완료. Manifest CSV 저장됨:", output_csv_path)
print(f"총 {len(results)}개의 음성 파일이 생성되었습니다.")
