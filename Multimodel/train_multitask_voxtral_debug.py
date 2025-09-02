# -*- coding: utf-8 -*-
"""
멀티태스크 Voxtral 학습 (ASR/SQA/TQA) — DEBUG 빌드
- Whisper encoder: freeze
- Projector + LLM: LoRA finetune
- 두 템플릿(전사 전용 / 일반 Chat-QA) 동시 학습
- Task-weighted sampler (확률 샘플링)
- Metrics: CER(ASR), ROUGE-L(SQA/TQA)
- Preview(TensorBoard) + 수동 주기 평가(구형 Transformers 대응)
- Gradient clipping(TrainingArguments) + Label smoothing(TrainingArguments)
- **디버그 프린트**: Dataset→Sampler→Collator(ASR/QA 인코딩/트렁케이션/레이블)→Batch pad→Preview in/out→Train step

A100 40GB 단일 GPU 예시:
python train_multitask_voxtral.py \
  --model_id ${MODEL_PATH:-/shared/models/Voxtral-Mini-3B-2507} \
  --train_jsonl data/data/ASR.jsonl,data/data/SQA.jsonl,data/data/TQA.jsonl \
  --val_jsonl   data/data/val_ASR.jsonl,data/data/val_SQA.jsonl,data/data/val_TQA.jsonl \
  --task_sampling 0.5,0.3,0.2 \
  --output_dir ckpt_voxtral_mt \
  --epochs 3 \
  --per_device_bsz 4 --grad_accum 48 \
  --max_len 1280 --eval_max_new_tokens 256 \
  --bf16 --tf32 --grad_ckpt \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --lr 3e-4 --warmup_steps 2500 --cosine \
  --num_workers 8 --prefetch_factor 2 --pin_memory --drop_last \
  --eval_steps 1500 --eval_accumulation_steps 4 \
  --preview_steps 1500 --preview_n 2 --preview_max_new_tokens 128 \
  --max_grad_norm 1.0 --label_smoothing 0.1
"""

import os, json, argparse, random
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoProcessor,
    VoxtralForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.tensorboard import SummaryWriter

# ==============================
# Debugger (경량 프린터)
# ==============================
class Debugger:
    def __init__(self, on=False, every=200, cap=5, prefix="[DBG]"):
        self.on = on
        self.every = max(1, every)
        self.cap = cap
        self.prefix = prefix
        self.counters = {}
    def tick(self, key):
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]
    def print(self, key, *msg):
        if not self.on:
            return
        n = self.tick(key)
        if (n <= self.cap) and (n % self.every == 0 or n == 1):
            safe = []
            for m in msg:
                try: safe.append(str(m))
                except Exception: safe.append("<unprintable>")
            print(self.prefix, f"[{key} #{n}]", " ".join(safe), flush=True)

# ==============================
# Metrics: CER / ROUGE-L
# ==============================

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
            if a[i-1]==b[j-1]: dp[i][j] = dp[i-1][j-1] + 1
            else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
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

# ==============================
# 데이터셋/샘플러
# ==============================
Task = str  # 'asr' | 'sqa' | 'tqa'

def detect_task(x: Dict[str, Any]) -> Task:
    if x.get('audio_path') and x.get('transcript'): return 'asr'
    if x.get('audio_path') and (x.get('instruction') or x.get('input')): return 'sqa'
    return 'tqa'

class JsonlDataset(Dataset):
    def __init__(self, paths: List[str], dbg: Optional[Debugger] = None):
        self.rows: List[Dict[str, Any]] = []
        for p in paths:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    ex = json.loads(line)
                    ex['task'] = detect_task(ex)
                    self.rows.append(ex)
        self.idxs_by_task: Dict[Task, List[int]] = {'asr': [], 'sqa': [], 'tqa': []}
        for i, ex in enumerate(self.rows):
            self.idxs_by_task[ex['task']].append(i)
        if dbg:
            dbg.print("dataset/init", f"loaded={len(self.rows)}",
                      f"asr={len(self.idxs_by_task['asr'])}",
                      f"sqa={len(self.idxs_by_task['sqa'])}",
                      f"tqa={len(self.idxs_by_task['tqa'])}")
            # 샘플 프리뷰
            for t in ['asr','sqa','tqa']:
                if self.idxs_by_task[t]:
                    i = self.idxs_by_task[t][0]
                    ex = self.rows[i]
                    head = {k:ex.get(k) for k in ['task','audio_path','transcript','instruction','input','output','answer']}
                    short = {k:(v[:80]+"...") if isinstance(v,str) and len(v)>80 else v for k,v in head.items()}
                    dbg.print("dataset/sample", f"{t}:", json.dumps(short, ensure_ascii=False))
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

class TaskWeightedSampler(WeightedRandomSampler):
    def __init__(self, dataset: JsonlDataset, task_probs: Tuple[float, float, float], num_samples: Optional[int]=None, dbg: Optional[Debugger]=None):
        asr_p, sqa_p, tqa_p = task_probs
        n_asr = max(1, len(dataset.idxs_by_task['asr']))
        n_sqa = max(1, len(dataset.idxs_by_task['sqa']))
        n_tqa = max(1, len(dataset.idxs_by_task['tqa']))
        weights = torch.zeros(len(dataset), dtype=torch.float)
        for i in dataset.idxs_by_task['asr']: weights[i] = asr_p / n_asr
        for i in dataset.idxs_by_task['sqa']: weights[i] = sqa_p / n_sqa
        for i in dataset.idxs_by_task['tqa']: weights[i] = tqa_p / n_tqa
        if num_samples is None: num_samples = len(dataset)
        super().__init__(weights=weights, num_samples=num_samples, replacement=True)
        if dbg:
            dbg.print("sampler/init", f"p={task_probs}", f"n=({n_asr},{n_sqa},{n_tqa})", f"num_samples={num_samples}")

# ==============================
# Collator (템플릿 + 레이블 생성)
# ==============================
@dataclass
class DataCollatorVoxtral:
    processor: Any
    model_id: str
    max_len: int
    language: str = 'ko'
    dbg: Optional[Debugger] = None

    def _pack_prompt_target(self, prompt_ids: torch.Tensor, target_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1D 텐서 보장
        if prompt_ids.dim()==2: prompt_ids = prompt_ids.view(-1)
        if target_ids.dim()==2: target_ids = target_ids.view(-1)
        input_ids = torch.cat([prompt_ids, target_ids], dim=0)
        labels    = torch.cat([torch.full_like(prompt_ids, -100), target_ids.clone()], dim=0)
        attn      = torch.ones_like(input_ids)
        if self.dbg:
            self.dbg.print("truncate/pack", f"p={prompt_ids.numel()} t={target_ids.numel()} in={input_ids.numel()} sup={(labels!=-100).sum().item()} max={self.max_len}")
        return input_ids[:self.max_len], labels[:self.max_len], attn[:self.max_len]

    def _encode_asr(self, ex: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        lang = ex.get('language', self.language)
        req = self.processor.apply_transcription_request(
            language=lang,
            audio=ex.get('audio_path') or ex.get('audio'),
            model_id=self.model_id,
        )
        # input_ids
        prompt_ids = req['input_ids']
        if not torch.is_tensor(prompt_ids):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        if prompt_ids.dim()==2: prompt_ids = prompt_ids.squeeze(0)
        # input_features (T,F) 또는 (1,T,F)
        feats = req['input_features']
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats, dtype=torch.float)
        if feats.dim()==3 and feats.shape[0]==1: feats = feats.squeeze(0)
        if feats.dim()!=2: feats = feats.view(-1, feats.shape[-1])
        # target_ids (transcript)
        target_text = (ex.get('transcript') or '').strip()
        tgt = self.processor.tokenizer(target_text, return_tensors='pt', add_special_tokens=True)
        target_ids = tgt['input_ids'].view(-1)
        input_ids, labels, attn = self._pack_prompt_target(prompt_ids, target_ids)
        if self.dbg:
            self.dbg.print("collate/asr", f"path={ex.get('audio_path')}", f"prompt_len={prompt_ids.numel()}", f"tgt_len={target_ids.numel()}", f"in_len={input_ids.numel()}", f"lab_sup={(labels!=-100).sum().item()}", f"feats_shape={tuple(feats.shape)}")
        return {
            'input_ids': input_ids,
            'attention_mask': attn,
            'labels': labels,
            'input_features': feats,  # (T,F)
        }

    def _encode_chat(self, ex: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        question = (ex.get('instruction') or ex.get('input') or '').strip()
        answer   = (ex.get('output') or ex.get('answer') or '').strip()
        tok = self.processor.tokenizer
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        prompt_ids = tok.apply_chat_template(
            messages[:-1], tokenize=True, return_tensors="pt",
            truncation=True, max_length=self.max_len,
        ).view(-1)
        full_ids = tok.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            truncation=True, max_length=self.max_len,
            continue_final_message=True,
        ).view(-1)
        # 라벨 마스킹
        input_ids = full_ids
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[: prompt_ids.shape[0]] = -100
        out = {
            "input_ids":      input_ids[:self.max_len],
            "attention_mask": attention_mask[:self.max_len],
            "labels":         labels[:self.max_len],
        }
        if ex.get('audio_path'):
            req = self.processor.apply_transcription_request(
                language=ex.get('language', self.language),
                audio=ex['audio_path'],
                model_id=self.model_id,
            )
            feats = req['input_features']
            if not torch.is_tensor(feats):
                feats = torch.tensor(feats, dtype=torch.float)
            if feats.dim()==3 and feats.shape[0]==1: feats = feats.squeeze(0)
            if feats.dim()!=2: feats = feats.view(-1, feats.shape[-1])
            out['input_features'] = feats
        if self.dbg:
            self.dbg.print("collate/qa", f"audio={'Y' if ex.get('audio_path') else 'N'}", f"prompt_len={prompt_ids.numel()}", f"tgt_len={(labels!=-100).sum().item()}", f"in_len={input_ids.numel()}")
        return out

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded: List[Dict[str, torch.Tensor]] = []
        for ex in features:
            if ex['task']== 'asr': encoded.append(self._encode_asr(ex))
            else:                 encoded.append(self._encode_chat(ex))
        # pad text fields
        pad_id = (
            self.processor.tokenizer.pad_token_id
            if getattr(self.processor.tokenizer, 'pad_token_id', None) is not None
            else self.processor.tokenizer.eos_token_id
        )
        ids = pad_sequence([e['input_ids'] for e in encoded], batch_first=True, padding_value=pad_id)
        am  = pad_sequence([e['attention_mask'] for e in encoded], batch_first=True, padding_value=0)
        lbs = pad_sequence([e['labels'] for e in encoded], batch_first=True, padding_value=-100)
        batch = {
            'input_ids': ids[:, :self.max_len],
            'attention_mask': am[:, :self.max_len],
            'labels': lbs[:, :self.max_len],
        }
        # audio features → (B,T_max,F)
        feat_list = [e.get('input_features', None) for e in encoded]
        if any(f is not None for f in feat_list):
            norm = []
            for f in feat_list:
                if f is None:
                    norm.append(None); continue
                t = f if torch.is_tensor(f) else torch.tensor(f, dtype=torch.float)
                if t.dim()==3 and t.shape[0]==1: t = t.squeeze(0)
                if t.dim()!=2: t = t.view(-1, t.shape[-1])
                norm.append(t)
            T_max = max((t.shape[0] for t in norm if t is not None), default=0)
            F_dim = max((t.shape[1] for t in norm if t is not None), default=0)
            B = len(norm)
            out = torch.zeros(B, T_max, F_dim, dtype=(norm[0].dtype if norm[0] is not None else torch.float))
            for i, t in enumerate(norm):
                if t is None: continue
                Ti, Fi = t.shape
                out[i, :Ti, :Fi] = t
            batch['input_features'] = out
        if self.dbg:
            msg = [f"ids={tuple(batch['input_ids'].shape)}", f"am={tuple(batch['attention_mask'].shape)}", f"lab={tuple(batch['labels'].shape)}", f"sup={(batch['labels']!=-100).sum().item()}"]
            if 'input_features' in batch:
                msg.append(f"feats={tuple(batch['input_features'].shape)}")
            self.dbg.print('collate/batch', ' '.join(msg))
            if torch.all(batch['labels']==-100):
                self.dbg.print('warn/no-supervised', 'all labels are -100 in this batch')
        return batch

# ==============================
# Preview 콜백 (TensorBoard + 콘솔 요약)
# ==============================
class PreviewCallback:
    def __init__(self, processor, writer: SummaryWriter, preview_samples: Dict[Task, List[Dict[str, Any]]],
                 device: str, max_new_tokens: int = 128, global_step_interval: int = 1500,
                 language: str = 'ko', model_id: str = '', dbg: Optional[Debugger]=None):
        self.processor = processor; self.writer = writer; self.samples = preview_samples
        self.device = device; self.max_new_tokens = max_new_tokens
        self.interval = global_step_interval; self.language = language; self.model_id = model_id
        self._last_logged = -1; self.dbg = dbg
    def maybe_log(self, trainer: Trainer):
        step = int(trainer.state.global_step or 0)
        if step==0 or step==self._last_logged or step%self.interval!=0: return
        self._last_logged = step
        model = trainer.model; model.eval()
        with torch.no_grad():
            for task, samples in self.samples.items():
                for i, ex in enumerate(samples):
                    if task=='asr':
                        enc = self.processor.apply_transcription_request(language=self.language, audio=ex['audio_path'], model_id=self.model_id)
                        input_ids = enc['input_ids']
                        if not torch.is_tensor(input_ids): input_ids = torch.tensor(input_ids, dtype=torch.long)
                        if input_ids.dim()==1: input_ids = input_ids.unsqueeze(0)
                        feats = enc['input_features']
                        if not torch.is_tensor(feats): feats = torch.tensor(feats, dtype=torch.float)
                        if feats.dim()==2: feats = feats.unsqueeze(0)
                        elif feats.dim()==3 and feats.shape[0]!=1: feats = feats[:1]
                        inputs = {'input_ids': input_ids.to(model.device), 'input_features': feats.to(model.device)}
                    else:
                        q = (ex.get('instruction') or ex.get('input') or '').strip()
                        tok = self.processor.tokenizer
                        enc_ids = tok.apply_chat_template([{"role":"user","content":q}], tokenize=True, return_tensors="pt")
                        inputs = {'input_ids': enc_ids.to(model.device)}
                        if ex.get('audio_path'):
                            req = self.processor.apply_transcription_request(language=self.language, audio=ex['audio_path'], model_id=self.model_id)
                            feats = req['input_features']
                            if not torch.is_tensor(feats): feats = torch.tensor(feats, dtype=torch.float)
                            if feats.dim()==2: feats = feats.unsqueeze(0)
                            inputs['input_features'] = feats.to(model.device)
                    if self.dbg:
                        sh = {k: tuple(v.shape) for k,v in inputs.items() if torch.is_tensor(v)}
                        self.dbg.print('preview/in', f"task={task} i={i} shapes={sh}")
                    gen = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                    out = self.processor.batch_decode(gen[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                    ref = ex.get('transcript') if task=='asr' else (ex.get('output') or ex.get('answer') or '')
                    user = (ex.get('instruction') or ex.get('input') or 'ASR')
                    text = (f"[Q] {user[:200]}\n"
                            f"[REF] {ref}\n[ PRED ] {out}")
                    if self.writer is not None:
                        tag = f"preview/{task}_{i}"
                        self.writer.add_text(tag, text, global_step=step)
                    if self.dbg:
                        self.dbg.print('preview/out', f"task={task} i={i} q={user[:40]!r} ref={ref[:40]!r} pred={out[:40]!r}")
        model.train()

# ==============================
# 커스텀 Trainer: 로더/평가/수동 주기 평가 + 디버그 로그
# ==============================
class FastTrainer(Trainer):
    def __init__(self, *args, preview_cb: Optional[PreviewCallback]=None, dbg: Optional[Debugger]=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.preview_cb = preview_cb
        self.dbg = dbg
    def _build_loader(self, dataset, batch_size: int, shuffle: bool, data_collator, num_workers: int,
                      prefetch_factor: int, pin_memory: bool, drop_last: bool, sampler=None):
        return DataLoader(dataset, batch_size=batch_size, shuffle=(shuffle and sampler is None), sampler=sampler,
                          collate_fn=data_collator, num_workers=num_workers,
                          prefetch_factor=(prefetch_factor if num_workers>0 else None), pin_memory=pin_memory,
                          drop_last=drop_last, persistent_workers=(num_workers>0))
    def get_train_dataloader(self):
        args = self.args; train_dataset = self.train_dataset; sampler = None
        if isinstance(train_dataset, JsonlDataset) and hasattr(args, 'task_probs'):
            sampler = TaskWeightedSampler(train_dataset, args.task_probs, dbg=getattr(self, 'dbg', None))
        loader = self._build_loader(train_dataset, args.train_batch_size, (sampler is None), self.data_collator,
                                  getattr(args,'num_workers',0), getattr(args,'prefetch_factor',2), getattr(args,'pin_memory',False), getattr(args,'drop_last',False), sampler)
        if self.dbg:
            self.dbg.print('loader/train', f'batch_size={args.train_batch_size} num_workers={getattr(args,"num_workers",0)} drop_last={getattr(args,"drop_last",False)} sampler={sampler is not None}')
        return loader
    def get_eval_dataloader(self, eval_dataset=None):
        args = self.args; eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        loader = self._build_loader(eval_dataset, args.eval_batch_size, False, self.data_collator,
                                  getattr(args,'num_workers',0), getattr(args,'prefetch_factor',2), getattr(args,'pin_memory',False), False)
        if self.dbg:
            self.dbg.print('loader/eval', f'batch_size={args.eval_batch_size} num_workers={getattr(args,"num_workers",0)}')
        return loader
    def evaluate(self, eval_dataset: Optional[Dataset]=None, ignore_keys=None, metric_key_prefix: str="eval"):
        if isinstance(self.eval_dataset, dict):
            logs = {}
            for task, ds in self.eval_dataset.items():
                out = super().evaluate(eval_dataset=ds, ignore_keys=ignore_keys, metric_key_prefix=f"{task}")
                logs.update(out)
            return logs
        return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
    def training_step(self, *args, **kwargs):
        if self.preview_cb is not None:
            self.preview_cb.maybe_log(self)
        out = super().training_step(*args, **kwargs)
        if self.dbg:
            step = int(self.state.global_step or 0)
            try:
                loss_val = out.item() if torch.is_tensor(out) else float(out)
            except Exception:
                loss_val = None
            self.dbg.print('train/step', f'step={step} loss={loss_val}')
        # 수동 평가 (구형 Transformers에서 evaluation_strategy 미지원 시)
        step = int(self.state.global_step or 0)
        interval = getattr(self.args, '_manual_eval_interval', 0)
        if interval and step>0 and step % interval == 0:
            try:
                self.evaluate()
            except Exception as e:
                print(f"[warn] manual evaluate failed at step {step}: {e}")
        return out

# ==============================
# 메인
# ==============================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_id', type=str, required=True)
    ap.add_argument('--train_jsonl', type=str, required=True, help='comma-separated')
    ap.add_argument('--val_jsonl', type=str, required=True, help='comma-separated(ASR,SQA,TQA 순서)')
    ap.add_argument('--task_sampling', type=str, default='0.5,0.3,0.2')
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--per_device_bsz', type=int, default=1)
    ap.add_argument('--per_device_eval_bsz', type=int, default=1)
    ap.add_argument('--grad_accum', type=int, default=24)
    ap.add_argument('--max_len', type=int, default=1280)
    ap.add_argument('--eval_max_new_tokens', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--warmup_steps', type=int, default=1500)
    ap.add_argument('--weight_decay', type=float, default=0.02)
    ap.add_argument('--cosine', action='store_true')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--tf32', action='store_true')
    ap.add_argument('--grad_ckpt', action='store_true')
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--prefetch_factor', type=int, default=2)
    ap.add_argument('--pin_memory', action='store_true')
    ap.add_argument('--drop_last', action='store_true')
    ap.add_argument('--eval_steps:deprecated', dest='eval_steps', type=int, default=1500)
    ap.add_argument('--eval_accumulation_steps', type=int, default=8)
    ap.add_argument('--preview_steps', type=int, default=1500)
    ap.add_argument('--preview_n', type=int, default=2)
    ap.add_argument('--preview_max_new_tokens', type=int, default=128)
    ap.add_argument('--max_grad_norm', type=float, default=1.0)
    ap.add_argument('--label_smoothing', type=float, default=0.1)
    ap.add_argument('--lora_r', type=int, default=16)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--load_4bit', action='store_true')
    ap.add_argument('--bnb_dtype', type=str, default='bfloat16')
    # Debug
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--debug_every', type=int, default=200)
    ap.add_argument('--debug_max', type=int, default=5)
    return ap.parse_args()


def main():
    args = parse_args()
    dbg = Debugger(on=args.debug, every=args.debug_every, cap=args.debug_max)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] {device}")

    processor = AutoProcessor.from_pretrained(args.model_id)
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        if dbg: dbg.print('tokenizer', f'set pad_token -> eos_token_id={tok.eos_token_id}')

    if args.load_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=getattr(torch, args.bnb_dtype),
            bnb_4bit_use_double_quant=True,
        )
        model = VoxtralForConditionalGeneration.from_pretrained(args.model_id, quantization_config=bnb, device_map='auto')
    else:
        dtype = torch.bfloat16 if args.bf16 and device=='cuda' else torch.float32
        model = VoxtralForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=dtype, device_map=device)

    # Freeze Whisper encoder
    n_frozen = 0
    for n, p in model.named_parameters():
        if 'whisper' in n or 'audio_encoder' in n:
            p.requires_grad_(False)
            n_frozen += 1
    if dbg: dbg.print('model/freeze', f'whisper-like params frozen = {n_frozen}')

    # LoRA targets
    lora_targets = set()
    for name, module in model.named_modules():
        if 'multi_modal_projector' in name and (name.endswith('linear_1') or name.endswith('linear_2')):
            lora_targets.add(name.split('.')[-1])
        if name.endswith(('q_proj','v_proj')):
            lora_targets.add(name.split('.')[-1])
    if not lora_targets: lora_targets = {'q_proj','v_proj','linear_1','linear_2'}
    peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                          target_modules=list(lora_targets), bias='none', task_type='CAUSAL_LM')
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model)
    # LoRA 주입
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # Gradient checkpointing/ cache
    if args.grad_ckpt:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
        if hasattr(model, 'config'):
            try:
                model.config.use_cache = False
            except Exception:
                pass
        if dbg: dbg.print('model/gc', 'enabled; set use_cache=False & input_require_grads')

    # Datasets
    train_paths = [p.strip() for p in args.train_jsonl.split(',') if p.strip()]
    val_paths = [p.strip() for p in args.val_jsonl.split(',') if p.strip()]
    if len(val_paths) != 3:
        raise ValueError('--val_jsonl 은 ASR,SQA,TQA 3개를 쉼표로 전달하세요')
    train_ds = JsonlDataset(train_paths, dbg=dbg)
    val_asr  = JsonlDataset([val_paths[0]], dbg=dbg)
    val_sqa  = JsonlDataset([val_paths[1]], dbg=dbg)
    val_tqa  = JsonlDataset([val_paths[2]], dbg=dbg)

    # Collator
    collator = DataCollatorVoxtral(processor=processor, model_id=args.model_id, max_len=args.max_len, language='ko', dbg=dbg)

    # Preview 샘플 pick
    def _pick(ds: JsonlDataset, task: Task, k: int) -> List[Dict[str, Any]]:
        idxs = ds.idxs_by_task[task][:]
        random.shuffle(idxs)
        out = [ds[i] for i in idxs[:k]]
        if dbg:
            dbg.print('preview/pick', f'task={task} n={len(out)}')
        return out
    preview_samples = {'asr': _pick(train_ds,'asr',args.preview_n), 'sqa': _pick(train_ds,'sqa',args.preview_n), 'tqa': _pick(train_ds,'tqa',args.preview_n)}

    # TrainingArguments
    asr_p, sqa_p, tqa_p = map(float, args.task_sampling.split(','))
    total_bsz = args.per_device_bsz * max(1, torch.cuda.device_count()) * args.grad_accum
    print(f"[Batch] per_device={args.per_device_bsz}, grad_accum={args.grad_accum}, global_tokens/batch≈{total_bsz}")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_bsz,
        per_device_eval_batch_size=args.per_device_eval_bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        save_steps=args.eval_steps,
        logging_steps=50,
        bf16=args.bf16,
        tf32=args.tf32,
        gradient_checkpointing=args.grad_ckpt,
        report_to=['tensorboard'],
        eval_accumulation_steps=args.eval_accumulation_steps,
        lr_scheduler_type=('cosine' if args.cosine else 'linear'),
        dataloader_num_workers=args.num_workers,
        max_grad_norm=args.max_grad_norm,
        label_smoothing_factor=args.label_smoothing,
    )
    # 커스텀 속성(샘플러/수동평가)
    targs.task_probs = (asr_p, sqa_p, tqa_p)
    targs.num_workers = args.num_workers
    targs.prefetch_factor = args.prefetch_factor
    targs.pin_memory = args.pin_memory
    targs.drop_last = args.drop_last
    targs._manual_eval_interval = args.eval_steps

    # TensorBoard + Preview
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'runs'))
    preview_cb = PreviewCallback(processor=processor, writer=writer, preview_samples=preview_samples,
                                 device=device, max_new_tokens=args.preview_max_new_tokens,
                                 global_step_interval=args.preview_steps, language='ko', model_id=args.model_id, dbg=dbg)

    trainer = FastTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset={'asr': val_asr, 'sqa': val_sqa, 'tqa': val_tqa},
        data_collator=collator,
        tokenizer=processor,  # processing_class 경고는 무시 가능
        preview_cb=preview_cb,
        dbg=dbg,
    )

    trainer.train()
    print('[Evaluate] ASR/SQA/TQA (final)')
    trainer.evaluate()
    trainer.save_model(); processor.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
