'''
python infer_from_ckpt.py \
  --model_id ${MODEL_PATH:-/shared/models/Voxtral-Mini-3B-2507} \
  --output_dir ckpt_stage1_text \
  --jsonl data/text_medqa.val.jsonl 
'''

import os, re, json, argparse, math, torch
from glob import glob
from typing import List, Dict, Optional

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    VoxtralForConditionalGeneration,
)
from peft import PeftModel
from torch.utils.data import DataLoader

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    cks = [d for d in glob(os.path.join(output_dir, "checkpoint-*")) if os.path.isdir(d)]
    if not cks:
        # 학습이 마지막에만 저장했다면 lora_stage1 폴더가 있을 수도 있음
        lora_dir = os.path.join(output_dir, "lora_stage1")
        return lora_dir if os.path.exists(lora_dir) else None
    def step(d):
        m = re.search(r"checkpoint-(\d+)", d)
        return int(m.group(1)) if m else -1
    return sorted(cks, key=step)[-1]

def build_messages(ex: Dict[str, str]) -> List[Dict[str, str]]:
    ins = (ex.get("instruction") or "").strip()
    cond = (ex.get("input") or "").strip()
    out  = (ex.get("output") or "").strip()
    user = ins if ins else ""
    if cond:
        user = f"{user}\n\n조건: {cond}"
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": out},
    ]

def distinct_n_ratio(text, n=2):
    toks = text.split()
    if len(toks) < n: return 1.0
    ngrams = set(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))
    return len(ngrams) / (len(toks)-n+1)

# ---- preview_generate 시그니처 및 본문 교체 ----
@torch.no_grad()
def preview_generate(
    model, tokenizer, rows: List[Dict],
    max_new_tokens: int = 128, device: str = "cuda",
    do_sample: bool = True, temperature: float = 0.7,
    top_p: float = 0.9, top_k: int = 0, typical_p: float = 1.0,
    repetition_penalty: float = 1.12, no_repeat_ngram: int = 3,
):
    model.eval()
    prev_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True

    # 샘플링 가드
    if do_sample and (temperature is None or temperature <= 0.0):
        print("[warn] temperature<=0 with do_sample=True → set to 1e-3")
        temperature = 1e-3

    for i, ex in enumerate(rows):
        user = (ex.get("instruction") or "").strip()
        if ex.get("input"):
            user += "\n\n조건: " + ex["input"].strip()
        messages = [{"role": "user", "content": user}]
        enc = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to(device)

        gen_kwargs = dict(
            input_ids=enc,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(dict(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram,
            ))
            if top_k and top_k > 0:
                gen_kwargs["top_k"] = top_k
            if typical_p and typical_p < 1.0:
                gen_kwargs["typical_p"] = typical_p
        else:
            gen_kwargs.update(dict(do_sample=False))  # greedy/beam 등 필요 시 여기서 확장

        gen_ids = model.generate(**gen_kwargs)
        out_ids = gen_ids[0, enc.shape[1]:]
        pred = tokenizer.decode(out_ids, skip_special_tokens=True)

        d2 = distinct_n_ratio(pred, n=2)
        print(f"[metrics] len={len(pred.split())} | distinct-2={d2:.3f}")
        print(f"\n=== SAMPLE {i} ===")
        print("User:\n", user)
        print("\nPred:\n", pred)
        if ex.get("output"):
            print("\nRef:\n", ex["output"])

    model.config.use_cache = prev_cache

def make_labelled_batch(tokenizer, examples: List[Dict], max_len: int, device: str):
    # training과 동일하게 user 부분은 label=-100 처리
    input_ids_list, attn_list, labels_list = [], [], []
    for ex in examples:
        messages = build_messages(ex)

        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, return_tensors="pt",
            truncation=True, max_length=max_len,
        ).view(-1)

        full_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            truncation=True, max_length=max_len, continue_final_message=True,
        ).view(-1)

        input_ids = full_ids
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[: prompt_ids.shape[0]] = -100

        input_ids_list.append(input_ids)
        attn_list.append(attention_mask)
        labels_list.append(labels)

    def pad_stack(seqs, pad):
        return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad)

    input_ids = pad_stack(input_ids_list, tokenizer.pad_token_id).to(device)
    attention_mask = pad_stack(attn_list, 0).to(device)
    labels = pad_stack(labels_list, -100).to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

@torch.no_grad()
def eval_small_ppl(model, tokenizer, jsonl_path: str, limit: int = 256, batch_size: int = 4, max_len: int = 1024, device: str = "cuda"):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= limit: break
            ex = json.loads(l)
            if ex.get("instruction") and ex.get("output"):
                rows.append(ex)
    if not rows:
        print("[eval] no rows loaded.")
        return

    def batcher(iterable, n):
        for idx in range(0, len(iterable), n):
            yield iterable[idx:idx+n]

    model.eval()
    losses = []
    for chunk in batcher(rows, batch_size):
        batch = make_labelled_batch(tokenizer, chunk, max_len, device)
        out = model(**batch)
        # out.loss는 평균(토큰/배치) loss
        losses.append(out.loss.item())

    avg_loss = sum(losses)/len(losses)
    ppl = math.exp(avg_loss) if math.isfinite(avg_loss) else float("inf")
    print(f"[eval] avg_loss={avg_loss:.4f} | ppl={ppl:.2f} on {len(rows)} samples (bsz={batch_size})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True, help="베이스 모델 경로(HF 포맷)")
    ap.add_argument("--output_dir", type=str, required=True, help="학습 출력 디렉토리 (checkpoint-XXXX 들이 있는 곳)")
    ap.add_argument("--jsonl", type=str, required=True, help="인퍼런스/평가에 사용할 jsonl (val 권장)")
    ap.add_argument("--num_preview", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--ppl_limit", type=int, default=256)
    ap.add_argument("--ppl_batch", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--do_sample", action="store_true", help="샘플링 사용 여부 (True면 temperature>0 필요)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)  # 0이면 무시
    ap.add_argument("--typical_p", type=float, default=1.0)  # 1.0이면 무시
    ap.add_argument("--repetition_penalty", type=float, default=1.12)
    ap.add_argument("--no_repeat_ngram", type=int, default=3)
    args = ap.parse_args()

    latest_ckpt = find_latest_checkpoint(args.output_dir)
    if latest_ckpt is None:
        raise FileNotFoundError(f"checkpoint-* or lora_stage1 not found under {args.output_dir}")

    print(f"[info] using adapter from: {latest_ckpt}")

    # Processor/Tokenizer
    proc = AutoProcessor.from_pretrained(args.model_id)
    tok = proc.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 4bit 로드 (학습과 동일)
    compute_dtype = torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = VoxtralForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=compute_dtype,
        quantization_config=bnb,
        device_map={"": 0} if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    # 오디오 경로 고정(학습 코드와 일치시킴) – 없으면 무시
    try:
        for n, p in base.named_parameters():
            if any(k in n.lower() for k in ["audio_tower","audio","whisper","feature","extractor","projector","adapter","bridge"]):
                p.requires_grad_(False)
    except Exception:
        pass

    base.config.use_cache = True  # 인퍼런스이므로 켬

    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(base, latest_ckpt)
    model.eval()

    # 1) 프리뷰: 몇 개만 생성 출력
    rows = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= args.num_preview: break
            ex = json.loads(l)
            if ex.get("instruction"):
                rows.append(ex)

    if rows:
        preview_generate(
            model, tok, rows,
            max_new_tokens=args.max_new_tokens, device=device,
            do_sample=args.do_sample, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, typical_p=args.typical_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram=args.no_repeat_ngram,
        )
    else:
        print("[preview] no rows to preview")

    # 2) 소규모 ppl 측정
    eval_small_ppl(model, tok, args.jsonl, limit=args.ppl_limit, batch_size=args.ppl_batch, max_len=args.max_len, device=device)

if __name__ == "__main__":
    main()