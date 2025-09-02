# find_long_samples.py
import json, argparse
from transformers import AutoProcessor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="mistralai/Voxtral-Mini-3B-2507")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--max_len", type=int, default=1024)
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model_id)
    tok = proc.tokenizer
    long_idx = []
    with open(args.jsonl,"r",encoding="utf-8") as f:
        for i,l in enumerate(f):
            ex = json.loads(l)
            msg = [{"role":"user","content":(ex.get("instruction") or "") + ("\n\n조건: "+(ex.get("input") or "") if ex.get("input") else "")},
                   {"role":"assistant","content":ex.get("output") or ""}]
            ids = tok.apply_chat_template(msg, tokenize=True, return_tensors="pt", truncation=False, continue_final_message=True)
            if ids.numel() > args.max_len:
                long_idx.append((i, ids.numel()))
    print(f"over {args.max_len} tokens: {len(long_idx)} samples")
    print(long_idx[:20])

if __name__ == "__main__":
    main()