"""
전체 QA 생성 데모 스크립트
- 모든 SPK에 대해 처리
- tqdm으로 프로그래스바 표시
- 로그 기록 및 개별 저장
- 최적화: 모델 한 번만 로드, GPU 활용
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qa_dataset_generator import QADatasetGenerator
import json
import time
import re
from tqdm import tqdm
import logging
import os

# 로그 설정
logging.basicConfig(
    filename='qa_generation.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def generate_all_spk():
    print("=== 전체 SPK QA 생성 데모 ===")

    model_id = "openai/gpt-oss-20b"
    save_path = "/shared/home/milab/users/kskim/gpt-oss-20b"
    
    os.makedirs('./preQA', exist_ok=True)

    # CUDA 디바이스 확인
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 모델 로드 (1회만)
    print("[1단계] 모델 로드 중...")
    start_model_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        save_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    end_model_load = time.perf_counter()
    print(f"[1단계 완료] 모델 로드 완료 ({end_model_load - start_model_load:.2f} sec)")

    # QA 생성기 초기화
    print("\nQA 생성기 초기화...")
    generator = QADatasetGenerator(model, tokenizer)

    # 데이터 로드
    with open("manifest_news.json", "r") as f:
        json_data = json.load(f)

    mid_start = False

    # 전체 SPK 반복 처리
    for entry in tqdm(json_data, desc="QA 생성 진행"):
        if entry['spk_id'] == 'SPK022YTNSO185':
            mid_start = True
        if mid_start == False:
            continue
        spk_id = entry['spk_id']
        merged_text = entry['merged_text']

        start_time = time.perf_counter()
        logging.info(f"SPK {spk_id}: QA 생성 시작")

        try:
            # QA 생성
            output = generator.generate_qa_pairs(merged_text, num_pairs=3, max_retries=2)

            # 대괄호 안 JSON 객체만 추출
            match = re.search(r'\[.*?\]', output, re.DOTALL)
            if match:
                json_only = match.group(0)
                qa_list = json.loads(json_only)
            else:
                qa_list = output  # JSON 못 찾으면 원본 그대로
                logging.warning(f"SPK {spk_id}: JSON 배열을 찾을 수 없음")

            # 개별 파일 저장
            save_file = f'./preQA/{spk_id}.json'
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(qa_list, f, indent=2, ensure_ascii=False)

            end_time = time.perf_counter()
            elapsed = end_time - start_time
            logging.info(f"SPK {spk_id}: QA 생성 완료, 소요시간 {elapsed:.2f} sec")

        except Exception as e:
            logging.error(f"SPK {spk_id}: QA 생성 실패 - {e}")

    print("=== 전체 QA 생성 완료 ===")

if __name__ == "__main__":
    generate_all_spk()
