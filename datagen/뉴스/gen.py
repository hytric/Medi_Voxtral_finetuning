# 필요한 라이브러리를 설치합니다.
# pip install aiohttp transformers tqdm

import aiohttp
import asyncio
import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer

# -------------------------------------------------------------------------
# 1. 요청 준비
# -------------------------------------------------------------------------

model_name = "/shared/home/milab/users/kskim/Meta-Llama-3-8B-Instruct" # <--- 이렇게 수정!
tokenizer = AutoTokenizer.from_pretrained(model_name)

data_path = "/shared/home/milab/users/kskim/dataset/138.뉴스_대본_및_앵커_음성_데이터/01-1.정식개방데이터/Validation/02.라벨링데이터"


# 모든 json 파일 수집
json_file_list = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".json"):
            json_file_list.append(os.path.join(root, file))

news_qas = []
for idx, json_path in enumerate(tqdm(json_file_list, desc="QA 변환 작업")):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # JSON 구조에 맞게 text 필드 추출
    script_data = data.get("script", {})
    text = script_data.get("text", None)  # 뉴스 기사 본문

    if text is None:
        print(f"{json_path}에서 script.text 필드를 찾지 못했습니다.")
        continue

    messages = [
        {
            "role": "system",
            "content": """당신은 뉴스 기사를 읽고 Q&A 데이터를 생성하는 AI입니다. 
주어진 기사 본문을 읽고 독자가 물어볼 법한 **자연스러운 질문 하나**와, 
그에 대한 **정확한 답변**을 생성하세요.

규칙:
1. 질문은 반드시 한국어로, 실제 독자가 묻는 것처럼 대화체 문장으로 하나만 생성합니다.
2. 답변은 반드시 기사 본문에 근거하여 작성합니다.
3. 출력은 JSON 형식으로만 하고, {"question": "...", "answer": "..."} 구조여야 합니다.
"""
        },
        {
            "role": "user",
            "content": f"""다음은 뉴스 기사 본문입니다. 이를 기반으로 질문과 답변을 생성해 주세요.

[본문]
{text}

[출력 형식]
{{"question": "...", "answer": "..."}}"""
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    news_qas.append({
        "text": text,
        "prompt": prompt,
    })



# -------------------------------------------------------------------------
# 2. 비동기 요청 함수
# -------------------------------------------------------------------------

async def generate_question(session, item):
    api_url = "http://localhost:8000/v1/completions"
    payload = {
        "model": model_name,
        "prompt": item["prompt"],
        "max_tokens": 2048,
        "temperature": 0.6,
        "top_p": 0.9,
        "stop": ["<|eot_id|>"]
    }
    headers = {"Content-Type": "application/json"}

    try:
        async with session.post(api_url, json=payload, headers=headers) as response:
            response.raise_for_status()
            result = await response.json()
            # 생성된 텍스트만 정확히 추출
            generated_text = result['choices'][0]['text'].strip()
            return {**item, "generated_text": generated_text}
            
    except aiohttp.ClientError as e:
        error_message = f"ERROR: ClientError - {e}"
        print(f"❌ ClientError for item '{item['text'][:20]}...': {e}")
        return {**item, "generated_text": error_message}
        
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        error_message = f"ERROR: DataError - {e}"
        print(f"❌ DataError for item '{item['text'][:20]}...': {e}")
        return {**item, "generated_text": error_message}

# -------------------------------------------------------------------------
# 3. 배치 실행 함수
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
    # news_qas 변수명으로 통일
    async with aiohttp.ClientSession() as session:
        results = await process_in_batches(session, news_qas, batch_size=batch_size)
        
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