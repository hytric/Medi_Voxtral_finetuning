# 필요한 라이브러리를 설치합니다.
# pip install aiohttp transformers tqdm
# 비대면 진료

import aiohttp
import asyncio
import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer

# -------------------------------------------------------------------------
# 1. 요청 준비
# -------------------------------------------------------------------------

model_name = "Meta-Llama-3-8B-Instruct" # <--- 이렇게 수정!
tokenizer = AutoTokenizer.from_pretrained(model_name)

def make_llama_prompt(question_text):
    """
    LLaMA용 대화 형식에 맞는 리스트를 생성합니다.
    (한국어, 한 문장 생성에 최적화)
    """
    messages = [
        {
            "role": "system", 
            "content": "당신은 병원 진료 상황을 시뮬레이션하는 AI입니다. 주어진 대화에 이어질 의사 또는 환자의 다음 발화를 **한 문장의 간결한 한국어**로 생성하세요. 영어는 절대 사용하지 마세요."
        },
        {
            "role": "user", 
            "content": question_text
        }
    ]
    return messages
    
# 병합할 파일 목록
filenames = ['train.scp', 'valid.scp']

# 모든 데이터를 저장할 리스트
all_data = []

# 각 파일을 순서대로 읽기
for filename in filenames:
    print(f"'{filename}' 파일을 읽는 중...")
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned_line = line.strip()
            if cleaned_line:
                parts = cleaned_line.split('|', 1)
                if len(parts) == 2:
                    file_path, text = parts
                    
                    messages = make_llama_prompt(text)
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    # 딕셔너리 형태로 데이터 추가
                    all_data.append({'path': file_path, 'text': text, 'prompt': prompt})


# -------------------------------------------------------------------------
# 2. 비동기 요청 함수
# -------------------------------------------------------------------------

async def generate_question(session, item):
    api_url = "http://localhost:8000/v1/completions"
    payload = {
        "model": model_name,
        "prompt": item["prompt"],
        "max_tokens": 1024,
        "temperature": 0.6,
        "top_p": 0.9,
        "stop": ["<|eot_id|>"]
    }
    headers = {"Content-Type": "application/json"}
    
    async with session.post(api_url, json=payload, headers=headers) as response:
        result = await response.json()
        generated_question = result['choices'][0]['text'].strip()
        # 원본 정보 + 생성된 질문 반환
        return {
            **item,
            "generated_question": generated_question
        }

# -------------------------------------------------------------------------
# 3. 배치 실행 함수 (프로그래스바 적용)
# -------------------------------------------------------------------------

async def process_in_batches(session, data, batch_size=2):
    results = []
    total_batches = (len(data) + batch_size - 1) // batch_size
    with tqdm(total=len(data), desc="질문 생성 진행", unit="건") as pbar:
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            tasks = [generate_question(session, item) for item in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            pbar.update(len(batch_results))
    return results

# -------------------------------------------------------------------------
# 4. 메인 실행부
# -------------------------------------------------------------------------

async def main(batch_size=2, output_path="generated_results.json"):
    async with aiohttp.ClientSession() as session:
        results = await process_in_batches(session, all_data, batch_size=batch_size)
        
        print("\n" + "="*60)
        print("✅ 모든 질문 생성 완료!")
        print("="*60 + "\n")

        # 결과를 json 파일로 저장
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"결과가 '{output_path}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    # batch 크기를 직접 정할 수 있음
    asyncio.run(main(batch_size=4))
