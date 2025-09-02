# -*- coding: utf-8 -*-
"""
Voxtral 멀티태스크(ASR/SQA/TQA) 인퍼런스 스크립트
- 학습 시 저장한 LoRA 어댑터 + base 모델 로드
- 학습과 동일한 템플릿 파이프라인(ASR 전사 템플릿 / 일반 Chat-QA)
- val 데이터셋을 그대로 돌며 ASR은 CER, SQA/TQA는 ROUGE-L 계산
- 샘플 N개 결과 프린트

예시:
python infer_voxtral_mt.py \
  --model_id ${MODEL_PATH:-/shared/models/Voxtral-Mini-3B-2507} \
  --lora_dir ckpt_voxtral_mt \
  --val_jsonl data/data/val_ASR.jsonl,data/data/val_SQA.jsonl,data/data/val_TQA.jsonl \
  --per_device_bsz 1 \
  --max_len 1280 --gen_max_new_tokens 256 \
  --bf16
"""

import os, json, argparse, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoProcessor,
    VoxtralForConditionalGeneration,
)
from peft import PeftModel

# --------------------
# 유틸: CER / ROUGE-L
# --------------------
def _edit_distance(a: List[str], b: List[str]) -> int:
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): dp[i][0] = i
    for j in range(len(b)+1): dp[0][j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + (0 if a[i-1]==b[j-1] else 1)
            )
    return dp[-1][-1]

def char_error_rate(ref: str, hyp: str) -> float:
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars)==0 else 1.0
    return _edit_distance(ref_chars, hyp_chars) / max(1, len(ref_chars))

def _lcs(a: List[str], b: List[str]) -> int:
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

def rouge_l(ref: str, hyp: str) -> float:
    ref_toks = ref.split()
    hyp_toks = hyp.split()
    if not ref_toks or not hyp_toks: return 0.0
    lcs = _lcs(ref_toks, hyp_toks)
    prec = lcs / len(hyp_toks)
    rec  = lcs / len(ref_toks)
    if prec+rec==0: return 0.0
    beta2 = 1.2*1.2
    return ((1+beta2)*prec*rec) / (rec + beta2*prec + 1e-8)

# --------------------
# 데이터셋
# --------------------
Task = str  # 'asr' | 'sqa' | 'tqa'

def detect_task(x: Dict[str, Any]) -> Task:
    if x.get('audio_path') and x.get('transcript'): return 'asr'
    if x.get('audio_path') and (x.get('instruction') or x.get('input')): return 'sqa'
    return 'tqa'

class JsonlDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.rows: List[Dict[str, Any]] = []
        for p in paths:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ex = json.loads(line)
                    ex['task'] = detect_task(ex)
                    self.rows.append(ex)
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

# --------------------
# Collator (학습 때와 동일한 템플릿 적용, labels는 메트릭 계산용)
# --------------------
@dataclass
class DataCollatorVoxtral:
    processor: Any
    model_id: str
    max_len: int
    language: str = 'ko'

    def _encode_asr(self, ex: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        req = self.processor.apply_transcription_request(
            language=ex.get('language', self.language),
            audio=ex.get('audio_path') or ex.get('audio'),
            model_id=self.model_id,
        )
        # prompt
        prompt_ids = req['input_ids']
        if not torch.is_tensor(prompt_ids):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        prompt_ids = prompt_ids.view(-1)

        # audio features (T, F)
        feats = req['input_features']
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats, dtype=torch.float)

        # target text → ids
        target_text = (ex.get('transcript') or '').strip()
        tgt = self.processor.tokenizer(target_text, return_tensors='pt', add_special_tokens=True)
        target_ids = tgt['input_ids'].view(-1)

        # concat
        input_ids = torch.cat([prompt_ids, target_ids], dim=0)[:self.max_len]
        labels    = torch.cat([torch.full_like(prompt_ids, -100), target_ids.clone()], dim=0)[:self.max_len]
        attn      = torch.ones_like(input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attn,
            'labels': labels,
            'input_features': feats,  # (T, F)
        }

    def _encode_chat(self, ex: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        q = (ex.get('instruction') or ex.get('input') or '').strip()
        a = (ex.get('output') or ex.get('answer') or '').strip()

        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        tok = self.processor.tokenizer

        prompt_ids = tok.apply_chat_template(
            messages[:-1], tokenize=True, return_tensors="pt",
            truncation=True, max_length=self.max_len
        ).view(-1)

        full_ids = tok.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            truncation=True, max_length=self.max_len,
            continue_final_message=True,
        ).view(-1)

        input_ids = full_ids
        labels = input_ids.clone()
        labels[: prompt_ids.shape[0]] = -100
        attn = torch.ones_like(input_ids)

        out = {
            "input_ids": input_ids[:self.max_len],
            "attention_mask": attn[:self.max_len],
            "labels": labels[:self.max_len],
        }

        # SQA 오디오는 feature 추가
        if ex.get("audio_path"):
            req = self.processor.apply_transcription_request(
                language=ex.get("language", self.language),
                audio=ex["audio_path"],
                model_id=self.model_id,
            )
            feats = req["input_features"]
            if not torch.is_tensor(feats):
                feats = torch.tensor(feats, dtype=torch.float)
            out["input_features"] = feats  # (T, F)

        return out

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        enc = []
        for ex in features:
            enc.append(self._encode_asr(ex) if ex['task']=='asr' else self._encode_chat(ex))

        pad_id = (
            self.processor.tokenizer.pad_token_id
            if getattr(self.processor.tokenizer, "pad_token_id", None) is not None
            else self.processor.tokenizer.eos_token_id
        )
        ids = pad_sequence([e["input_ids"] for e in enc], batch_first=True, padding_value=pad_id)
        am  = pad_sequence([e["attention_mask"] for e in enc], batch_first=True, padding_value=0)
        lbs = pad_sequence([e["labels"] for e in enc], batch_first=True, padding_value=-100)

        batch = {
            "input_ids": ids,
            "attention_mask": am,
            "labels": lbs,
        }

        # 오디오 피처 패딩 (B, T, F)
        feat_list = [e.get("input_features", None) for e in enc]
        if any(f is not None for f in feat_list):
            norm = []
            for f in feat_list:
                if f is None:
                    norm.append(None); continue
                t = f if torch.is_tensor(f) else torch.tensor(f, dtype=torch.float)
                if t.dim() == 3 and t.shape[0] == 1:  # (1, T, F) → (T, F)
                    t = t.squeeze(0)
                if t.dim() != 2:
                    t = t.view(-1, t.shape[-1])
                norm.append(t)
            Tm = max((t.shape[0] for t in norm if t is not None), default=0)
            Fm = max((t.shape[1] for t in norm if t is not None), default=0)
            B = len(norm)
            out = torch.zeros(B, Tm, Fm, dtype=norm[0].dtype if norm[0] is not None else torch.float)
            for i, t in enumerate(norm):
                if t is None: continue
                Ti, Fi = t.shape
                out[i, :Ti, :Fi] = t
            batch["input_features"] = out

        return batch

# --------------------
# 인퍼런스
# --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_id', type=str, required=True, help='base model id or path')
    ap.add_argument('--lora_dir', type=str, required=True, help='학습 산출물 디렉토리(LoRA 어댑터)')
    ap.add_argument('--val_jsonl', type=str, required=True, help='ASR,SQA,TQA 순서로 comma-separated')
    ap.add_argument('--per_device_bsz', type=int, default=1)
    ap.add_argument('--max_len', type=int, default=1280)
    ap.add_argument('--gen_max_new_tokens', type=int, default=256)
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--print_samples', type=int, default=5)
    return ap.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if (device=='cuda' and args.bf16) else torch.float16 if device=='cuda' else torch.float32
    print(f"[Device] {device} | dtype={dtype}")

    # Processor
    processor = AutoProcessor.from_pretrained(args.model_id)
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Base + LoRA
    base = VoxtralForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=dtype, device_map=device)
    model = PeftModel.from_pretrained(base, args.lora_dir, is_trainable=False)
    # 추론 최적화: merge & unload (필요시)
    try:
        model = model.merge_and_unload()
    except Exception:
        pass
    model.eval()

    # Datasets
    val_asr, val_sqa, val_tqa = [p.strip() for p in args.val_jsonl.split(',')]
    ds_asr = JsonlDataset([val_asr])
    ds_sqa = JsonlDataset([val_sqa])
    ds_tqa = JsonlDataset([val_tqa])

    collator = DataCollatorVoxtral(processor=processor, model_id=args.model_id, max_len=args.max_len, language='ko')

    dl_asr = DataLoader(ds_asr, batch_size=args.per_device_bsz, shuffle=False, collate_fn=collator, num_workers=2)
    dl_sqa = DataLoader(ds_sqa, batch_size=args.per_device_bsz, shuffle=False, collate_fn=collator, num_workers=2)
    dl_tqa = DataLoader(ds_tqa, batch_size=args.per_device_bsz, shuffle=False, collate_fn=collator, num_workers=2)

    # ---------------- metrics & prints ----------------
    def _decode_continuation(input_ids: torch.Tensor, gen_ids: torch.Tensor) -> List[str]:
        # gen_ids: (B, L_gen) or from generate()
        if gen_ids.shape[1] <= input_ids.shape[1]:
            cont = torch.zeros((gen_ids.shape[0], 0), dtype=gen_ids.dtype, device=gen_ids.device)
        else:
            cont = gen_ids[:, input_ids.shape[1]:]
        return processor.batch_decode(cont, skip_special_tokens=True)

    @torch.no_grad()
    def run_eval(dl, task: Task, print_prefix: str):
        preds, refs = [], []
        shown = 0
        for batch in dl:
            # move to device
            inputs = {}
            for k, v in batch.items():
                if k == "labels": continue
                if torch.is_tensor(v):
                    v = v.to(device)
                inputs[k] = v

            # generate
            prev = getattr(model.config, "use_cache", False)
            model.config.use_cache = True
            gen = model.generate(
                **inputs,
                max_new_tokens=args.gen_max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            model.config.use_cache = prev

            # decode
            hyp = _decode_continuation(inputs["input_ids"], gen)
            # reference from labels
            lab = batch["labels"]
            for i in range(lab.size(0)):
                ref_ids = lab[i][lab[i] != -100].tolist()
                ref_txt = processor.batch_decode([ref_ids], skip_special_tokens=True)[0] if ref_ids else ""
                refs.append(ref_txt)
            preds.extend(hyp)

            # print samples
            while shown < args.print_samples and shown < len(preds):
                print(f"\n[{print_prefix} sample #{shown+1}]")
                print("REF :", refs[shown][:400])
                print("PRED:", preds[shown][:400])
                shown += 1

        # metric
        if task == 'asr':
            vals = [char_error_rate(r, p) for r, p in zip(refs, preds)]
            score = sum(vals) / max(1, len(vals))
            print(f"\n[{print_prefix}] CER = {score:.4f} (n={len(vals)})")
        else:
            vals = [rouge_l(r, p) for r, p in zip(refs, preds)]
            score = sum(vals) / max(1, len(vals))
            print(f"\n[{print_prefix}] ROUGE-L = {score:.4f} (n={len(vals)})")

    # ---------------- run ----------------
    print("\n[Infer] ASR (val)")
    run_eval(dl_asr, 'asr', 'ASR')

    print("\n[Infer] SQA (val)")
    run_eval(dl_sqa, 'sqa', 'SQA')

    print("\n[Infer] TQA (val)")
    run_eval(dl_tqa, 'tqa', 'TQA')

if __name__ == "__main__":
    main()