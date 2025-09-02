# build_cache.py
import os, json, torch
from tqdm import tqdm
from transformers import AutoProcessor

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
PROC = AutoProcessor.from_pretrained(MODEL_ID)
TOK = PROC.tokenizer
if TOK.pad_token is None: TOK.pad_token = TOK.eos_token

IN = "dataset/train_data.jsonl"
OUT_JSONL = "dataset/train_data.cached.jsonl"
OUT_DIR = "dataset/cache_stage2"
os.makedirs(OUT_DIR, exist_ok=True)

with open(IN, "r", encoding="utf-8") as fin, open(OUT_JSONL, "w", encoding="utf-8") as fout:
    for i, line in enumerate(tqdm(fin, desc="precompute")):
        ex = json.loads(line)
        a = (ex.get("audio") or ex.get("audio_path") or "").strip()
        t = (ex.get("text")  or ex.get("transcript") or "").strip()
        if not (a and t): 
            continue
        enc = PROC.apply_transcription_request(
            language=["ko"], audio=[a], model_id=MODEL_ID, return_tensors="pt"
        )
        prompt_ids = enc["input_ids"][0]          # 1D
        input_features = enc["input_features"][0] # (80, T)

        # 저장
        stem = f"ex{i:08d}"
        torch.save({"input_features": input_features}, f"{OUT_DIR}/{stem}.pt")
        rec = {
            "cache_path": f"{OUT_DIR}/{stem}.pt",
            "prompt_ids": prompt_ids.tolist(),
            "ref_text": t
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("done:", OUT_JSONL)