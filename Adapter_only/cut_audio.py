import json
import os
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

def filter_and_save_by_length(input_jsonl, output_jsonl, min_sec=3.0, max_sec=25.0):
    """오디오 길이가 min_sec 이상, max_sec 이하인 데이터만 필터링하여 저장"""
    valid_lengths = []
    with open(input_jsonl, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for line in tqdm(lines, desc=f"Filtering {os.path.basename(input_jsonl)}"):
            data = json.loads(line)
            audio_path = data.get("audio_path")
            if audio_path and os.path.exists(audio_path):
                info = sf.info(audio_path)
                duration = info.frames / info.samplerate
                if min_sec <= duration <= max_sec:
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                    valid_lengths.append(duration)
    return valid_lengths

os.makedirs("./dataset", exist_ok=True)

train_lengths = filter_and_save_by_length(
    "dataset/medical_manifest.jsonl", "./dataset/train_data.jsonl", min_sec=3.0, max_sec=25.0
)
# valid_lengths = filter_and_save_by_length(
#     "validation_data.jsonl", "./dataset/validation_data.jsonl", min_sec=3.0, max_sec=25.0
# )

train_mean = sum(train_lengths) / len(train_lengths) if train_lengths else 0
# valid_mean = sum(valid_lengths) / len(valid_lengths) if valid_lengths else 0

plt.figure(figsize=(12, 6))
plt.hist(train_lengths, bins=50, alpha=0.7, label=f'Train (Mean: {train_mean:.2f}s)')
# plt.hist(valid_lengths, bins=50, alpha=0.7, label=f'Validation (Mean: {valid_mean:.2f}s)')
plt.axvline(train_mean, color='blue', linestyle='dashed', linewidth=2)
# plt.axvline(valid_mean, color='orange', linestyle='dashed', linewidth=2)
plt.xlabel('Audio Length (seconds)')
plt.ylabel('Count')
plt.title('Audio Length Distribution (Filtered)')
plt.legend()
plt.grid(True)
plt.show()