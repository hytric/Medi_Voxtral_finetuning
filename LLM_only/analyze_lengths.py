# analyze_lengths.py
import json, argparse, math, statistics
from typing import Dict, List
from collections import Counter
import torch
from transformers import AutoProcessor

def build_messages(ex: Dict[str, str]) -> List[Dict[str, str]]:
    ins = (ex.get("instruction") or "").strip()
    cond = (ex.get("input") or "").strip()
    out  = (ex.get("output") or "").strip()
    user = ins if ins else ""
    if cond:
        user = f"{user}\n\n조건: {cond}"
    return [{"role":"user","content":user}, {"role":"assistant","content":out}]

def seq_len(tok, messages, max_len=4096):
    prompt = tok.apply_chat_template(messages[:-1], tokenize=True, return_tensors="pt",
                                     truncation=True, max_length=max_len)
    full   = tok.apply_chat_template(messages,     tokenize=True, return_tensors="pt",
                                     truncation=True, max_length=max_len,
                                     continue_final_message=True)
    return prompt.view(-1).shape[0], full.view(-1).shape[0]

def percentile(arr, p):
    if not arr: return 0
    k = (len(arr)-1) * p/100.0
    f = math.floor(k); c = math.ceil(k)
    if f == c: return arr[int(k)]
    return arr[f] + (arr[c]-arr[f])*(k-f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=os.getenv("MODEL_PATH", "/shared/models/Voxtral-Mini-3B-2507"))
    ap.add_argument("--jsonl", default='data/text_medqa.jsonl')
    ap.add_argument("--max_rows", type=int, default=50000)  # 전수면 크게 올리기
    ap.add_argument("--max_len", type=int, default=4096)
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model_id)
    tok  = proc.tokenizer
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    prompt_lens, full_lens = [], []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_rows and i >= args.max_rows: break
            ex = json.loads(line)
            msgs = build_messages(ex)
            p, t = seq_len(tok, msgs, args.max_len)
            prompt_lens.append(p); full_lens.append(t)

    prompt_lens.sort(); full_lens.sort()

    def stats(name, arr):
        print(f"\n[{name}] n={len(arr)}")
        print(f"  mean={statistics.mean(arr):.1f}  median={percentile(arr,50)}")
        for q in [75,90,95,99]:
            print(f"  p{q}={percentile(arr,q)}")
        print(f"  max={max(arr)}")

    stats("prompt_len", prompt_lens)
    stats("full_len", full_lens)

    # 간단 히스토그램 (bin=512단위)
    def hist(arr, bin_size=512):
        buckets = Counter((x // bin_size) * bin_size for x in arr)
        for b in sorted(buckets):
            print(f"{b:>4}-{b+bin_size-1:<4}: {buckets[b]}")
    print("\n[Histogram full_len, bin=512]")
    hist(full_lens, 512)

if __name__ == "__main__":
    main()