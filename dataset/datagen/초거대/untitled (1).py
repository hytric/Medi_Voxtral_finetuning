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

model_name = "Meta-Llama-3-8B-Instruct" # <--- 이렇게 수정!
tokenizer = AutoTokenizer.from_pretrained(model_name)

data_path = "1.데이터"


# 모든 json 파일 수집
json_file_list = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".json"):
            json_file_list.append(os.path.join(root, file))

medical_answers = []
for idx, json_path in enumerate(tqdm(json_file_list, desc="QA 변환 작업")):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    answer = data.get("answer", None)
    department = data.get("department", None)
    disease_name = data.get("disease_name", None)
    disease_category = data.get("disease_category", None)

    if answer is None:
        print(f"{json_path}에서 answer 필드를 찾지 못했습니다.")
        continue

    # answer가 dict일 경우 intro/body/conclusion을 합침
    if isinstance(answer, dict):
        answer_text = ""
        for key in ["intro", "body", "conclusion"]:
            if key in answer and answer[key]:
                answer_text += answer[key].strip() + "\n"
        answer_text = answer_text.strip()
    else:
        answer_text = str(answer).strip()

    messages = [
        {
            "role": "system",
            "content": """당신은 주어진 의료 답변을 보고, 그 답변을 이끌어냈을 환자의 원래 질문을 역으로 추론하는 AI입니다.
다음 규칙을 반드시 준수하세요:
1. **반드시 한국어로만, 그리고 오직 하나의 질문만** 생성하세요.
2. 질문은 실제 환자가 말하는 것처럼 **자연스러운 대화체**여야 합니다.
3. 번호, 목록, 부연 설명 등 **질문 외의 다른 텍스트는 절대 포함하지 마세요.**"""
        },
        {
            "role": "user",
            "content": f"""다음은 '{department}' 진료과목의 '{disease_name}'에 대한 답변입니다. 이 답변이 나오게 된 원래 질문을 생성해 주세요.

[답변]
{answer_text}

[생성할 질문]"""
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    medical_answers.append({
        "answer": answer_text,
        "department": department,
        "disease_name": disease_name,
        "disease_category": disease_category,
        "prompt": prompt,
    })


# -------------------------------------------------------------------------
# 2. 비동기 요청 함수
# -------------------------------------------------------------------------

async def generate_question(session, item):
    api_url = "http://localhost:8001/v1/completions"
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
    # all_data를 medical_answers로 사용하도록 변수명 통일
    async with aiohttp.ClientSession() as session:
        results = await process_in_batches(session, medical_answers, batch_size=batch_size)
        
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
