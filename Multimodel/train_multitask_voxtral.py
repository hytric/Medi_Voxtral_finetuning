# -*- coding: utf-8 -*-
"""
멀티태스크 Voxtral 학습 (ASR/SQA/TQA)
- Whisper encoder: freeze
- Projector + LLM: LoRA finetune
- 두 템플릿(전사 전용 / 일반 Chat-QA) 동시 학습
- Task-weighted sampler (확률 샘플링)
- Metrics: CER(ASR), ROUGE-L(SQA/TQA)
- Preview(TensorBoard) + 수동 주기 평가(구형 Transformers 대응)
- Gradient clipping + Label smoothing
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

# --------------------
# 데이터셋/샘플러
# --------------------
Task = str  # 'asr' | 'sqa' | 'tqa'

def detect_task(x: Dict[str, Any]) -> Task:
    if x.get('audio_path') and x.get('transcript'): return 'asr'
    if x.get('audio_path') and (x.get('instruction') or x.get('input')): return 'sqa'
    return 'tqa'

class JsonlDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.rows: List[Dict[str, Any]] = []
        self.idxs_by_task: Dict[Task, List[int]] = {'asr': [], 'sqa': [], 'tqa': []}

        for p in paths:
            with open(p, 'r', encoding='utf-8') as f:
                for ln, line in enumerate(f, start=1):
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        ex = json.loads(s)
                    except Exception as e:
                        raise ValueError(f"[JSONL parse] {p}:{ln} parse error: {e}\nline={s[:200]}")

                    if not isinstance(ex, dict):
                        raise TypeError(f"[JSONL type] {p}:{ln} not a dict: {repr(ex)[:200]}")

                    task = detect_task(ex)
                    if task not in ('asr','sqa','tqa'):
                        raise ValueError(f"[JSONL task] {p}:{ln} unknown task for sample: {ex}")

                    ex['task'] = task
                    self.rows.append(ex)
                    self.idxs_by_task[task].append(len(self.rows)-1)

        if len(self.rows) == 0:
            raise ValueError("[JSONL] No valid samples loaded. Check your paths and file contents.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 방어: 음수/범위 초과
        if not (0 <= idx < len(self.rows)):
            raise IndexError(f"[JSONL] index out of range: {idx} / {len(self.rows)}")
        return self.rows[idx]

class TaskWeightedSampler(WeightedRandomSampler):
    def __init__(self, dataset: JsonlDataset, task_probs: Tuple[float, float, float], num_samples: Optional[int]=None):
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

# --------------------
# Collator(템플릿 + 레이블 생성)
# --------------------
@dataclass
class DataCollatorVoxtral:
    processor: Any
    model_id: str
    max_len: int
    language: str = 'ko'

    def _encode_asr(self, ex: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # ASR 전용 템플릿 (processing_voxtral.py)
        req = self.processor.apply_transcription_request(
            language=ex.get('language', self.language),
            audio=ex.get('audio_path') or ex.get('audio'),  # path/ndarray/base64 허용
            model_id=self.model_id,
        )

        # prompt_ids: (L,) 보장
        prompt_ids = req['input_ids']
        if not torch.is_tensor(prompt_ids):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        if prompt_ids.dim() == 2:
            prompt_ids = prompt_ids.squeeze(0)

        # input_features: (T, F) 보장
        feats = req['input_features']
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats, dtype=torch.float)
        if feats.dim() == 3 and feats.shape[0] == 1:
            feats = feats.squeeze(0)
        if feats.dim() != 2:
            feats = feats.view(-1, feats.shape[-1])

        # 라벨: transcript 토크나이즈
        target_text = (ex.get('transcript') or '').strip()
        tok = self.processor.tokenizer
        tgt = tok(target_text, return_tensors='pt', add_special_tokens=True)['input_ids']
        if tgt.dim() == 2: tgt = tgt.squeeze(0)  # (L,)

        # concat + 마스킹
        input_ids = torch.cat([prompt_ids, tgt], dim=0)
        labels    = torch.cat([torch.full_like(prompt_ids, -100), tgt.clone()], dim=0)
        attn      = torch.ones_like(input_ids)

        return {
            'input_ids':      input_ids[:self.max_len],
            'attention_mask': attn[:self.max_len],
            'labels':         labels[:self.max_len],
            'input_features': feats,  # (T, F)
        }

    def _encode_chat(self, ex: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # 멀티모달 QA: tokenizer.apply_chat_template로 프롬프트/라벨 생성
        question = (ex.get('instruction') or ex.get('input') or '').strip()
        answer   = (ex.get('output') or ex.get('answer') or '').strip()

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        tok = self.processor.tokenizer

        # user-only (prompt) / user+assistant (full)
        prompt_ids = tok.apply_chat_template(
            messages[:-1], tokenize=True, return_tensors="pt",
            truncation=True, max_length=self.max_len,
        ).view(-1)

        full_ids = tok.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            truncation=True, max_length=self.max_len,
            continue_final_message=True,  # 일부 버전에서 필요
        ).view(-1)

        labels = full_ids.clone()
        labels[: prompt_ids.shape[0]] = -100

        batch = {
            "input_ids": full_ids[:self.max_len],
            "attention_mask": torch.ones_like(full_ids[:self.max_len]),
            "labels": labels[:self.max_len],
        }

        # 오디오가 있으면 feature만 추가
        if ex.get("audio_path"):
            req = self.processor.apply_transcription_request(
                language=ex.get("language", self.language),
                audio=ex["audio_path"], model_id=self.model_id,
            )
            feats = req["input_features"]
            if not torch.is_tensor(feats):
                feats = torch.tensor(feats, dtype=torch.float)
            if feats.dim() == 3 and feats.shape[0] == 1:
                feats = feats.squeeze(0)
            if feats.dim() != 2:
                feats = feats.view(-1, feats.shape[-1])
            batch["input_features"] = feats

        return batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded: List[Dict[str, torch.Tensor]] = []
        for i, ex in enumerate(features):
            if not isinstance(ex, dict):
                raise TypeError(f"[Collator] Expected dict but got {type(ex)} at batch_idx={i}: {repr(ex)[:200]}")
            if "task" not in ex:
                raise KeyError(f"[Collator] Missing 'task' key at batch_idx={i}, ex={repr(ex)[:200]}")
    
            if ex['task'] == 'asr':
                # ASR 필수 필드 확인
                if not (ex.get('audio_path') or ex.get('audio')):
                    raise KeyError(f"[Collator-ASR] Missing 'audio_path'/'audio' at batch_idx={i}")
                if 'transcript' not in ex:
                    # 학습에선 라벨 필요
                    raise KeyError(f"[Collator-ASR] Missing 'transcript' at batch_idx={i}")
                encoded.append(self._encode_asr(ex))
            else:
                # SQA/TQA 필수 필드 확인
                if not (ex.get('instruction') or ex.get('input')):
                    raise KeyError(f"[Collator-CHAT] Missing 'instruction'/'input' at batch_idx={i}")
                if not (ex.get('answer') or ex.get('output')):
                    raise KeyError(f"[Collator-CHAT] Missing 'answer'/'output' at batch_idx={i}")
                encoded.append(self._encode_chat(ex))

        # ↓↓↓ pad_id도 eos로 고정 권장 (pad_token_id가 '.' 인 케이스 방지)
        pad_id = (
            self.processor.tokenizer.eos_token_id
            if getattr(self.processor.tokenizer, "eos_token_id", None) is not None
            else 0
        )
        ids = pad_sequence([e["input_ids"] for e in encoded], batch_first=True, padding_value=pad_id)
        am  = pad_sequence([e["attention_mask"] for e in encoded], batch_first=True, padding_value=0)
        lbs = pad_sequence([e["labels"] for e in encoded], batch_first=True, padding_value=-100)

        batch = {
            "input_ids":      ids[:, : self.max_len],
            "attention_mask": am[:, : self.max_len],
            "labels":         lbs[:, : self.max_len],
        }

        # 오디오 피처 pad → (B, T_max, F)
        feat_list = [e.get("input_features", None) for e in encoded]
        if any(f is not None for f in feat_list):
            norm = []
            for f in feat_list:
                if f is None: norm.append(None); continue
                t = f if torch.is_tensor(f) else torch.tensor(f, dtype=torch.float)
                if t.dim() == 3 and t.shape[0] == 1: t = t.squeeze(0)
                if t.dim() != 2: t = t.view(-1, t.shape[-1])
                norm.append(t)
            T_max = max((t.shape[0] for t in norm if t is not None), default=0)
            F_dim = max((t.shape[1] for t in norm if t is not None), default=0)
            B = len(norm)
            out = torch.zeros(B, T_max, F_dim, dtype=norm[0].dtype if norm[0] is not None else torch.float)
            for i, t in enumerate(norm):
                if t is None: continue
                Ti, Fi = t.shape
                out[i, :Ti, :Fi] = t
            batch["input_features"] = out

        return batch

# --------------------
# Preview 콜백(샘플 텍스트/음성 로깅) — 학습 경로와 정합
# --------------------
# --------------------
# Preview 콜백(샘플 텍스트 로깅) - ASR/QA 분기 일치
# --------------------
class PreviewCallback:
    """
    일정 global_step마다 몇 개 샘플을 골라,
    - ASR: Whisper 프롬프트 + 오디오 피처를 넣어 디코딩
    - SQA/TQA: chat template로 디코딩
    결과 텍스트를 TensorBoard에 기록
    """
    def __init__(
        self,
        processor,                       # ⟵ AutoProcessor 전체를 받는다
        writer: SummaryWriter,
        preview_samples: List[Dict[str, Any]],
        device: str,
        max_new_tokens: int = 128,
        global_step_interval: int = 1500,
        debug: bool = False,
    ):
        self.proc = processor            # ⟵ 저장
        self.tok = processor.tokenizer   # ⟵ tokenizer 핸들
        self.writer = writer
        self.samples = preview_samples
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.interval = global_step_interval
        self._last_logged = -1
        self.debug = debug

    def _build_asr_inputs(self, processor, ex: Dict[str, Any], device):
        """
        학습 collator의 _encode_asr와 동일한 입력을 만든다.
        반환: dict(input_ids, input_features)
        """
        req = processor.apply_transcription_request(
            language=ex.get("language", "ko"),
            audio=ex.get("audio_path") or ex.get("audio"),
            model_id=getattr(processor, "model_input_name", None) or "",  # model_id가 필요없으면 무시됨
        )
        prompt_ids = req["input_ids"]
        if not torch.is_tensor(prompt_ids):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        if prompt_ids.dim() == 2 and prompt_ids.shape[0] == 1:
            prompt_ids = prompt_ids.squeeze(0)

        feats = req["input_features"]
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats, dtype=torch.float)
        if feats.dim() == 3 and feats.shape[0] == 1:
            feats = feats.squeeze(0)
        if feats.dim() != 2:  # (T, F)
            feats = feats.view(-1, feats.shape[-1])

        return {
            "input_ids": prompt_ids.to(device).unsqueeze(0),     # (1, Lprompt)
            "input_features": feats.to(device).unsqueeze(0),     # (1, T, F)
        }

    def _build_chat_inputs(self, tokenizer, ex: Dict[str, Any], device):
        """
        user-only 프롬프트를 만들어 generate용 input_ids를 반환.
        """
        question = (ex.get("instruction") or ex.get("input") or "").strip()
        messages = [{"role": "user", "content": question}]
        enc = tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt"
        ).to(device)
        return {"input_ids": enc}

    def maybe_log(self, trainer: Trainer):
        step = int(trainer.state.global_step or 0)
        if step == 0 or step == self._last_logged or step % self.interval != 0:
            return
        self._last_logged = step
        if self.debug:
            print(f"[preview] logging at step={step}, num_samples={len(self.samples)}")

        model = trainer.model
        model.eval()
        model_device = next(model.parameters()).device

        had_gc = getattr(model, "is_gradient_checkpointing", False)
        if had_gc:
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                pass

        with torch.no_grad():
            for idx, ex in enumerate(self.samples):
                task = detect_task(ex)  # 'asr' | 'sqa' | 'tqa'
                ref = (ex.get("output") or ex.get("answer") or ex.get("transcript") or "").strip()

                # generate 파라미터 공통
                prev_cache = getattr(model.config, "use_cache", False)
                if hasattr(model.config, "use_cache"):
                    model.config.use_cache = True

                try:
                    if task == "asr":
                        # ASR: whisper prompt + audio features
                        io = self._build_asr_inputs(self.proc, ex, model_device)
                        gen_ids = model.generate(
                            input_ids=io["input_ids"],                 # (1, Lprompt)
                            input_features=io["input_features"],       # (1, T, F)
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            eos_token_id=model.config.eos_token_id,
                            pad_token_id=model.config.pad_token_id,
                        )
                        # ASR은 prompt 이후가 모델의 전사 결과
                        out_ids = gen_ids[0, io["input_ids"].shape[1]:]

                    else:
                        # SQA/TQA: chat-template 프롬프트
                        enc = self._build_chat_inputs(self.tok, ex, model_device)["input_ids"]
                        gen_ids = model.generate(
                            input_ids=enc,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            eos_token_id=model.config.eos_token_id,
                            pad_token_id=model.config.pad_token_id,
                        )
                        out_ids = gen_ids[0, enc.shape[1]:]

                    pred = self.tok.decode(out_ids, skip_special_tokens=True)

                    # 로그 텍스트
                    question = (ex.get("instruction") or ex.get("input") or "ASR")[:200]
                    text = f"[Task] {task}\n[Q] {question}\n[REF] {ref}\n[PRED] {pred}"

                    if self.writer is not None:
                        self.writer.add_text(f"preview/sample_{idx}", text, global_step=step)

                    if self.debug:
                        print(f"[preview] sample#{idx}\n{text}")

                finally:
                    if hasattr(model.config, "use_cache"):
                        model.config.use_cache = prev_cache

        if had_gc:
            try:
                model.gradient_checkpointing_enable()
                if hasattr(model.config, "use_cache"):
                    model.config.use_cache = False
            except Exception:
                pass

        model.train()

# --------------------
# 커스텀 Trainer: 수동 주기 평가 + 미리보기
# --------------------
class FastTrainer(Trainer):
    def __init__(self, *args, preview_cb: Optional[PreviewCallback]=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.preview_cb = preview_cb

    def _build_loader(self, dataset, batch_size: int, shuffle: bool, data_collator,
                      num_workers: int, prefetch_factor: int, pin_memory: bool, drop_last: bool, sampler=None):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            collate_fn=data_collator,
            num_workers=num_workers,
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=(num_workers > 0),
        )

    def get_train_dataloader(self):
        args = self.args
        train_dataset = self.train_dataset
        sampler = None
        if isinstance(train_dataset, JsonlDataset) and hasattr(args, 'task_probs'):
            sampler = TaskWeightedSampler(train_dataset, args.task_probs)
        return self._build_loader(
            train_dataset, args.train_batch_size, (sampler is None), self.data_collator,
            getattr(args, 'num_workers', 0), getattr(args, 'prefetch_factor', 2),
            getattr(args, 'pin_memory', False), getattr(args, 'drop_last', False), sampler
        )

    def get_eval_dataloader(self, eval_dataset=None):
        args = self.args
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return self._build_loader(
            eval_dataset, args.eval_batch_size, False, self.data_collator,
            getattr(args, 'num_workers', 0), getattr(args, 'prefetch_factor', 2),
            getattr(args, 'pin_memory', False), False
        )

    def training_step(self, *args, **kwargs):
        # 미리보기
        if self.preview_cb is not None:
            self.preview_cb.maybe_log(self)

        out = super().training_step(*args, **kwargs)

        # 수동 평가(구형 트레이너 호환)
        step = int(self.state.global_step or 0)
        interval = getattr(self.args, '_manual_eval_interval', 0)
        if interval and step > 0 and step % interval == 0:
            try:
                self.evaluate()
            except Exception as e:
                import traceback
                print(f"[warn] manual evaluate failed at step {step}: {e}")
                traceback.print_exc()
        return out

# --------------------
# 메인
# --------------------
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

    ap.add_argument('--epochs', type=int, default=1)
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
    return ap.parse_args()


def main():
    args = parse_args()

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] {device}")

    processor = AutoProcessor.from_pretrained(args.model_id)

    # 모델 로드
    if args.load_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=getattr(torch, args.bnb_dtype),
            bnb_4bit_use_double_quant=True
        )
        model = VoxtralForConditionalGeneration.from_pretrained(
            args.model_id, quantization_config=bnb, device_map='auto'
        )
    else:
        dtype = torch.bfloat16 if (args.bf16 and device=='cuda') else torch.float32
        model = VoxtralForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype=dtype, device_map=device
        )

    tok = processor.tokenizer
    # eos는 기존대로 tokenizer가 제공하는 값 사용
    eos_id = tok.eos_token_id
    assert eos_id is not None and eos_id != -1, "Tokenizer must have eos_token_id"
    
    # pad는 tokenizer에 정의된 값 그대로 사용
    pad_id = tok.pad_token_id
    assert pad_id is not None and pad_id != -1, "Tokenizer must have pad_token_id"
    
    # 모델 설정
    model.config.eos_token_id = eos_id
    model.config.pad_token_id = pad_id   # 🚨 eos가 아니라 tokenizer.pad_token_id 사용
    try:
        tok.pad_token_id = pad_id        # tokenizer에도 반영
    except Exception:
        pass
    
    print(f"[Tokenizer] eos_id={eos_id}, pad_id={pad_id}")

    # Whisper/audio encoder freeze
    for n, p in model.named_parameters():
        ln = n.lower()
        if any(k in ln for k in ["whisper", "audio_tower", "audio", "feature", "extractor", "projector", "adapter", "bridge"]):
            p.requires_grad_(False)

    # GC와 use_cache 호환
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # LoRA 타깃
    lora_targets = {"q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj","linear_1","linear_2"}
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=list(lora_targets), bias='none', task_type='CAUSAL_LM'
    )
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # 데이터셋
    train_paths = [p.strip() for p in args.train_jsonl.split(',') if p.strip()]
    val_paths = [p.strip() for p in args.val_jsonl.split(',') if p.strip()]
    if len(val_paths) != 3:
        raise ValueError('--val_jsonl 은 ASR,SQA,TQA 3개를 쉼표로 전달하세요')

    train_ds = JsonlDataset(train_paths)
    val_asr  = JsonlDataset([val_paths[0]])
    val_sqa  = JsonlDataset([val_paths[1]])
    val_tqa  = JsonlDataset([val_paths[2]])

    collator = DataCollatorVoxtral(processor=processor, model_id=args.model_id, max_len=args.max_len, language='ko')

    def _pick(ds: JsonlDataset, task: Task, k: int) -> List[Dict[str, Any]]:
        pool = ds.idxs_by_task.get(task, [])
        if not pool:
            return []  # 해당 태스크 표본이 없으면 빈 리스트
        idxs = pool[:]  # shallow copy
        random.shuffle(idxs)
        # ✅ 인덱싱 시 __getitem__ 정상 동작
        return [ds[i] for i in idxs[:k]]
    preview_samples = []
    preview_samples += _pick(train_ds, 'asr', args.preview_n)
    preview_samples += _pick(train_ds, 'sqa', args.preview_n)
    preview_samples += _pick(train_ds, 'tqa', args.preview_n)

    # TrainingArguments (evaluation_strategy 미지정 → 수동평가)
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
        remove_unused_columns=False,
    )
    # 커스텀 속성(샘플러/수동평가/로더)
    targs.task_probs = (asr_p, sqa_p, tqa_p)
    targs.num_workers = args.num_workers
    targs.prefetch_factor = args.prefetch_factor
    targs.pin_memory = args.pin_memory
    targs.drop_last = args.drop_last
    targs._manual_eval_interval = args.eval_steps

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'runs'))
    preview_cb = PreviewCallback(
        processor=processor,
        writer=writer,
        preview_samples=preview_samples,
        device=device,
        max_new_tokens=args.preview_max_new_tokens,
        global_step_interval=args.preview_steps,
    )

    from torch.utils.data import ConcatDataset
    eval_union = ConcatDataset([val_asr, val_sqa, val_tqa])

    trainer = FastTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_union,
        data_collator=collator,
        processing_class=processor,
        preview_cb=preview_cb,
    )

    assert len(train_ds) > 0, "train_jsonl에서 유효 샘플을 찾지 못했습니다."
    print("[Train pool sizes]", {k: len(v) for k, v in train_ds.idxs_by_task.items()})

    trainer.train()
    print('[Evaluate] ASR/SQA/TQA (final)')
    trainer.evaluate()
    trainer.save_model()
    try:
        processor.save_pretrained(args.output_dir)
    except Exception:
        pass


if __name__ == '__main__':
    main()

    