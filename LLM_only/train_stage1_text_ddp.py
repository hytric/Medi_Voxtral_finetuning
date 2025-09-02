# train_stage1_text_ddp.py  (2x A6000 / DDP + TensorBoard + Preview 전략 + group_by_length)
'''
torchrun --standalone --nproc_per_node=2 \
  train_stage1_text_ddp.py \
  --model_id ${MODEL_PATH:-/shared/models/Voxtral-Mini-3B-2507} \
  --train_jsonl data/text_medqa.train.jsonl \
  --val_jsonl   data/text_medqa.val.jsonl \
  --output_dir  ckpt_stage1_text \
  --per_device_bsz 6 --grad_accum 8
'''
import os, json, argparse, random, time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    VoxtralForConditionalGeneration,
    AutoProcessor,
    TrainingArguments as _TA,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from collections import deque
import math

# ==============================
#  Utils
# ==============================
def set_speed_flags():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")  # pytorch>=2.0
    except Exception:
        pass

def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))

def is_rank0():
    return int(os.environ.get("RANK", "0")) == 0

def build_messages(ex: Dict[str, str]) -> List[Dict[str, str]]:
    ins = (ex.get("instruction") or "").strip()
    cond = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    user = ins if ins else ""
    if cond:
        user = f"{user}\n\n조건: {cond}"
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": out},
    ]

# ==== GC 탐지 유틸 ====
def _unwrap_base_model(m):
    # PEFT/PeftModel → .base_model → (가능하면) .model 까지 풀어서 내부 레이어 접근성 확보
    base = m
    for attr in ["base_model", "model"]:
        if hasattr(base, attr):
            base = getattr(base, attr)
    return base

def detect_gradient_checkpointing(model) -> bool:
    """
    다양한 HF/PEFT 버전 호환: 
    - 모델 최상단 flag
    - 각 레이어(module.gradient_checkpointing) 순회
    둘 다 확인해서 하나라도 True면 켜진 걸로 판단.
    """
    try:
        # 1) 최상단 플래그류
        for k in ["is_gradient_checkpointing", "gradient_checkpointing"]:
            v = getattr(model, k, None)
            if isinstance(v, bool) and v:
                return True

        base = _unwrap_base_model(model)

        # 2) 모듈 순회로 레이어 단위 체크포인트 설정 확인
        for _, mod in base.named_modules():
            if hasattr(mod, "gradient_checkpointing") and bool(getattr(mod, "gradient_checkpointing")):
                return True
    except Exception:
        pass
    return False


# ==============================
#  Dataset: 미리 인코딩(속도 ↑)
# ==============================
class EncodedTQADataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int, max_rows: Optional[int] = None):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for i, l in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                ex = json.loads(l)
                if not (ex.get("instruction") and ex.get("output")):
                    continue
                self.rows.append(ex)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        ex = self.rows[i]
        messages = build_messages(ex)

        prompt_ids = self.tok.apply_chat_template(
            messages[:-1],
            tokenize=True, return_tensors="pt",
            truncation=True, max_length=self.max_len,
        ).view(-1)

        full_ids = self.tok.apply_chat_template(
            messages,
            tokenize=True, return_tensors="pt",
            truncation=True, max_length=self.max_len,
            continue_final_message=True,
        ).view(-1)

        input_ids = full_ids
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[: prompt_ids.shape[0]] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ==============================
#  Collator: padding-only
# ==============================
@dataclass
class DataCollatorPadOnly:
    tokenizer: Any
    label_pad_id: int = -100
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        def pad_stack(seqs, pad):
            return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad)
        return {
            "input_ids":      pad_stack([f["input_ids"]      for f in features], self.tokenizer.pad_token_id),
            "attention_mask": pad_stack([f["attention_mask"] for f in features], 0),
            "labels":         pad_stack([f["labels"]         for f in features], self.label_pad_id),
        }


# ==============================
#  모델 유틸
# ==============================
def freeze_audio_path(model):
    for n, p in model.named_parameters():
        if any(k in n.lower() for k in ["audio_tower","audio","whisper","feature","extractor","projector","adapter","bridge"]):
            p.requires_grad_(False)


# ==============================
#  미리보기 콜백 (중간 인퍼런스)
# ==============================
class PreviewCallback(TrainerCallback):
    """
    일정 스텝마다 샘플 N개에 대해 generate() 실행 후
    TensorBoard에 텍스트로 기록. (rank 0 전용)
    초반에는 가볍게(max_new_tokens 작게), ramp_after 이후에는 ramp_max로 자동 증가.
    """
    def __init__(
        self,
        tokenizer,
        processor,
        preview_samples: List[Dict[str, str]],
        every_steps: int = 1500,
        max_new_tokens: int = 64,
        ramp_after: Optional[int] = 8000,
        ramp_max_new_tokens: Optional[int] = 128,
    ):
        self.tok = tokenizer
        self.proc = processor
        self.samples = preview_samples
        self.every_steps = max(1, every_steps)
        self.max_new_tokens = max_new_tokens
        self.ramp_after = ramp_after
        self.ramp_max_new_tokens = ramp_max_new_tokens
        self.writer: Optional[SummaryWriter] = None

    def _is_rank0(self, state):
        # 호환성: 일부 버전에 state.is_world_process_zero 없음
        return getattr(state, "is_world_process_zero", False) or is_rank0()

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._is_rank0(state):
            return
        log_dir = os.path.join(args.output_dir, "runs")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_step_end(self, args, state, control, **kwargs):
        if (state.global_step == 0) or (state.global_step % self.every_steps != 0) or (not self._is_rank0(state)):
            return

        model = kwargs["model"]
        model.eval()
        device = next(model.parameters()).device

        # 프리뷰 토큰 길이 자동 램프업
        cur_max_new = self.max_new_tokens
        if self.ramp_after is not None and self.ramp_max_new_tokens is not None:
            if state.global_step >= self.ramp_after:
                cur_max_new = self.ramp_max_new_tokens

        with torch.no_grad():
            for idx, ex in enumerate(self.samples):
                ins = (ex.get("instruction") or "").strip()
                cond = (ex.get("input") or "").strip()
                ref  = (ex.get("output") or "").strip()

                user = ins if ins else ""
                if cond:
                    user = f"{user}\n\n조건: {cond}"

                messages = [{"role":"user","content":user}]
                enc = self.tok.apply_chat_template(messages, tokenize=True, return_tensors="pt").to(device)

                prev_cache = getattr(model.config, "use_cache", False)
                model.config.use_cache = True
                gen_ids = model.generate(
                    input_ids=enc,
                    max_new_tokens=cur_max_new,
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=self.tok.eos_token_id,
                    pad_token_id=self.tok.pad_token_id,
                )
                model.config.use_cache = prev_cache

                out_ids = gen_ids[0, enc.shape[1]:]
                pred = self.tok.decode(out_ids, skip_special_tokens=True)

                if self.writer is not None:
                    tag = f"preview/idx{idx}"
                    self.writer.add_text(tag, f"### User\n{user}\n\n### Pred\n{pred}\n\n### Ref\n{ref}", global_step=state.global_step)

        model.train()

class MetricsCallback(TrainerCallback):
    """
    - step당 처리 토큰/초
    - step time (초)
    - 평균 시퀀스 길이, 패딩/낭비 비율
    - GPU 메모리(alloc/reserved/max_alloc) [GB]
    - loss, ppl(exp(loss)), EMA loss
    - 길이 히스토그램
    전부 rank0에서만 TensorBoard 기록
    """
    def __init__(self, log_every_steps: int = 10, ema_beta: float = 0.98, hist_every_steps: int = 100):
        self.log_every_steps = max(1, log_every_steps)
        self.hist_every_steps = max(1, hist_every_steps)
        self.ema_beta = ema_beta
        self.ema_loss: Optional[float] = None
        self.last_step_t0: Optional[float] = None
        self.step_times = deque(maxlen=50)
        self.writer: Optional[SummaryWriter] = None

    def _is_rank0(self, state):
        return getattr(state, "is_world_process_zero", False) or is_rank0()

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._is_rank0(state):
            return
        log_dir = os.path.join(args.output_dir, "runs")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    # 가능한 폭넓은 버전 호환을 위해 둘 다 구현
    def on_step_begin(self, args, state, control, **kwargs):
        self.last_step_t0 = time.time()

    def on_train_batch_begin(self, args, state, control, **kwargs):
        if self.last_step_t0 is None:
            self.last_step_t0 = time.time()

    def on_train_batch_end(self, args, state, control, **kwargs):
        if not self._is_rank0(state) or self.writer is None:
            return

        step = state.global_step
        if step == 0 or (step % self.log_every_steps != 0):
            return

        t1 = time.time()
        step_time = None
        if self.last_step_t0 is not None:
            step_time = max(1e-8, t1 - self.last_step_t0)
            self.step_times.append(step_time)

        inputs = kwargs.get("inputs", None)
        tokens_this_step = None
        pad_ratio = None
        avg_len = None
        waste_ratio = None

        if inputs is not None and "attention_mask" in inputs:
            am = inputs["attention_mask"]
            # (bsz, seq) 텐서 가정
            if isinstance(am, torch.Tensor):
                if am.dim() == 2:
                    valid_tokens = am.sum().item()
                    total_tokens = am.numel()
                    pad_tokens = total_tokens - valid_tokens
                    tokens_this_step = int(valid_tokens)
                    pad_ratio = pad_tokens / max(1, total_tokens)
                    # 평균 실제 길이
                    avg_len = (am.sum(dim=1).float().mean().item())
                    # 낭비율(패딩이 차지하는 비율)
                    waste_ratio = pad_ratio
                    # 길이 히스토그램(가끔만)
                    if step % self.hist_every_steps == 0:
                        lens = am.sum(dim=1).detach().cpu().numpy()
                        self.writer.add_histogram("length/hist", lens, global_step=step)

        # 손실/EMA/PPL은 on_log에서 안정적으로 잡히지만,
        # 여기서는 outputs에서 즉시 잡히면 같이 기록(버전별로 outputs 형식 상이 → 방어적)
        outputs = kwargs.get("outputs", None)
        loss_val = None
        if outputs is not None:
            # outputs may be dict with 'loss' or object with .loss
            try:
                if isinstance(outputs, dict) and "loss" in outputs:
                    lv = outputs["loss"]
                    loss_val = lv.item() if torch.is_tensor(lv) else float(lv)
                elif hasattr(outputs, "loss"):
                    lv = outputs.loss
                    loss_val = lv.item() if torch.is_tensor(lv) else float(lv)
            except Exception:
                pass

        ppl = math.exp(loss_val) if loss_val is not None else None
        if loss_val is not None:
            if self.ema_loss is None:
                self.ema_loss = loss_val
            else:
                self.ema_loss = self.ema_beta * self.ema_loss + (1 - self.ema_beta) * loss_val

        # 토큰/초
        toks_per_sec = None
        if tokens_this_step is not None and step_time is not None:
            toks_per_sec = tokens_this_step / step_time

        # GPU 메모리
        gpu_alloc = gpu_reserved = gpu_max = None
        try:
            dev = torch.cuda.current_device()
            gpu_alloc = torch.cuda.memory_allocated(dev) / 1e9
            gpu_reserved = torch.cuda.memory_reserved(dev) / 1e9
            gpu_max = torch.cuda.max_memory_allocated(dev) / 1e9
        except Exception:
            pass

        # 스칼라 기록
        if step_time is not None:
            self.writer.add_scalar("perf/step_time_sec", step_time, step)
        if len(self.step_times) > 0:
            avg_time = sum(self.step_times) / len(self.step_times)
            self.writer.add_scalar("perf/step_time_sec_ma50", avg_time, step)

        if tokens_this_step is not None:
            self.writer.add_scalar("tokens/step_tokens", tokens_this_step, step)
        if toks_per_sec is not None:
            self.writer.add_scalar("tokens/tokens_per_sec", toks_per_sec, step)
        if avg_len is not None:
            self.writer.add_scalar("length/avg_seq_len", avg_len, step)
        if pad_ratio is not None:
            self.writer.add_scalar("length/pad_ratio", pad_ratio, step)
        if waste_ratio is not None:
            self.writer.add_scalar("length/padding_waste_ratio", waste_ratio, step)

        if gpu_alloc is not None:
            self.writer.add_scalar("gpu_mem/allocated_GB", gpu_alloc, step)
        if gpu_reserved is not None:
            self.writer.add_scalar("gpu_mem/reserved_GB", gpu_reserved, step)
        if gpu_max is not None:
            self.writer.add_scalar("gpu_mem/max_alloc_GB", gpu_max, step)

        if loss_val is not None:
            self.writer.add_scalar("loss/train_step", loss_val, step)
        if self.ema_loss is not None:
            self.writer.add_scalar("loss/ema", self.ema_loss, step)
        if ppl is not None and math.isfinite(ppl):
            self.writer.add_scalar("loss/ppl_step", ppl, step)

    def on_log(self, args, state, control, **kwargs):
        # HF가 기본으로 올리는 메트릭들(learning_rate, loss, epoch 등)을 TB에 재기록 + ppl
        if not self._is_rank0(state) or self.writer is None:
            return
        logs = kwargs.get("logs", {})
        step = state.global_step
        for k, v in logs.items():
            # 이미 Trainer가 report_to="tensorboard"면 기록됨. 하지만 명확히 분리된 네임스페이스로도 남겨둔다.
            try:
                self.writer.add_scalar(f"trainer/{k}", float(v), step)
            except Exception:
                pass
        if "loss" in logs:
            try:
                ppl = math.exp(float(logs["loss"]))
                if math.isfinite(ppl):
                    self.writer.add_scalar("trainer/ppl_from_log", ppl, step)
            except Exception:
                pass
                
    def _log_kv(self, tag: str, val, step: int, prefix: str = ""):
        """간단한 스칼라/텍스트 로거 (writer 없으면 무시)"""
        if self.writer is None:
            return
        if isinstance(val, (int, float)):
            self.writer.add_scalar(f"{prefix}{tag}", val, step)
        else:
            self.writer.add_text(f"{prefix}{tag}", str(val), step)

    def _log_gc_status(self, args, state, model):
        gc_on = detect_gradient_checkpointing(model)
        # 텐서보드(스칼라: 0/1) + 텍스트 상세
        self._log_kv("gc/enabled", 1 if gc_on else 0, step=state.global_step or 0)
        detail = (
            f"GC detected: {gc_on}\n"
            f"HF args.gradient_checkpointing: {getattr(args, 'gradient_checkpointing', None)}\n"
            f"HF args.gradient_checkpointing_kwargs: {getattr(args, 'gradient_checkpointing_kwargs', None)}\n"
            f"use_cache (model.config): {getattr(getattr(model, 'config', None), 'use_cache', None)}"
        )
        self._log_kv("gc/detail", detail, step=state.global_step or 0)

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._is_rank0(state):
            return
        log_dir = os.path.join(args.output_dir, "runs")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # ▼▼▼ 여기 한 줄 추가: 시작 시 GC 상태를 TB에 기록
        model = kwargs.get("model", None)
        if model is not None:
            self._log_gc_status(args, state, model)
            

# ==============================
#  Main
# ==============================
def main():
    set_speed_flags()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default=os.getenv("MODEL_PATH", "/shared/models/Voxtral-Mini-3B-2507"))
    ap.add_argument("--train_jsonl", type=str, default="data/text_medqa.train.jsonl")
    ap.add_argument("--val_jsonl",   type=str, default="data/text_medqa.val.jsonl")
    ap.add_argument("--output_dir",  type=str, default="ckpt_stage1_text")

    # 학습 하이퍼파라미터
    ap.add_argument("--epochs",  type=float, default=3)
    ap.add_argument("--lr",      type=float, default=5e-5)
    ap.add_argument("--max_len", type=int,   default=1024)

    # 배치/가속
    ap.add_argument("--per_device_bsz", type=int, default=16)   # GPU당 batch size
    ap.add_argument("--grad_accum",     type=int, default=8)
    ap.add_argument("--seed",           type=int, default=42)

    # LoRA
    ap.add_argument("--lora_r",       type=int,   default=32)
    ap.add_argument("--lora_alpha",   type=int,   default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.1)

    # 미리보기 (가볍게 시작 → 램프업)
    ap.add_argument("--preview_steps", type=int, default=1500)
    ap.add_argument("--preview_n",     type=int, default=3)
    ap.add_argument("--preview_max_new_tokens", type=int, default=64)
    ap.add_argument("--preview_ramp_after", type=int, default=8000)
    ap.add_argument("--preview_ramp_max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    # 분산 관련
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    set_seed(args.seed)

    # Processor/Tokenizer
    proc = AutoProcessor.from_pretrained(args.model_id)
    tok  = proc.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 4bit QLoRA (A6000 → FP16 compute 권장)
    compute_dtype = torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map={"": local_rank},     # 프로세스별 전용 GPU
        torch_dtype=compute_dtype,
        quantization_config=bnb,
        low_cpu_mem_usage=True,
    )

    freeze_audio_path(model)
    model.config.use_cache = False
    
    # GC OFF
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if hasattr(model, "gradient_checkpointing_disable"):
        try: model.gradient_checkpointing_disable()
        except Exception: pass
    model.enable_input_require_grads()

    lconf = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lconf)
    if is_rank0():
        model.print_trainable_parameters()

    # Dataset (미리 인코딩)
    train_ds = EncodedTQADataset(args.train_jsonl, tok, args.max_len)
    eval_ds  = EncodedTQADataset(args.val_jsonl, tok, args.max_len) if args.val_jsonl else None

    # Collator
    collator = DataCollatorPadOnly(tok)

    # TrainingArguments (분산 + 속도 + group_by_length 조건부 적용)
    _base_args = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_bsz,   # GPU당 배치
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_ratio=0.07,
        weight_decay=0.01,
        max_grad_norm=0.3,
        bf16=(compute_dtype is torch.bfloat16),  # FP16이면 False, BF16이면 True
        logging_steps=10,
        save_steps=1000,
        save_total_limit=2,
        gradient_checkpointing=False,   # 메모리 부족 시 True
        report_to="tensorboard",
        optim="adamw_torch",
        remove_unused_columns=False,
        dataloader_num_workers=6,       # 총 12코어 → 프로세스당 6
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        dataloader_drop_last=True,
    )
    
    fields = getattr(_TA, "__dataclass_fields__", {})

    if "gradient_checkpointing" in fields:
        _base_args["gradient_checkpointing"] = True
    if "gradient_checkpointing_kwargs" in fields:
        _base_args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    
    # 평가 전략(버전 있을 때만)
    if eval_ds and ("evaluation_strategy" in fields):
        _base_args.update(evaluation_strategy="steps", eval_steps=1000)
    
    # 길이 버킷팅(지원 시 켬) + length_column_name 명시(안전)
    if "group_by_length" in fields:
        _base_args["group_by_length"] = True
    if "length_column_name" in fields:
        _base_args["length_column_name"] = "input_ids"
    
    # ====== DDP 세팅 (여기가 핵심 수정) ======
    if "ddp_backend" in fields:
        _base_args["ddp_backend"] = "nccl"
    
    # Voxtral/LoRA에서 계층 경로에 따라 일부 어댑터가 스텝별로 미사용 판정될 수 있음 → True 권장
    if "ddp_find_unused_parameters" in fields:
        _base_args["ddp_find_unused_parameters"] = True   # <<<<<< 변경: True 로!
    
    # 버퍼 브로드캐스트 비활성(가능하면)
    if "ddp_broadcast_buffers" in fields:
        _base_args["ddp_broadcast_buffers"] = False
    
    # (가능 시) fused AdamW
    try:
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            _base_args["optim"] = "adamw_torch_fused"
    except Exception:
        pass
    
    targs = _TA(**_base_args)

    # 미리보기 샘플(VAL 우선, 없으면 TRAIN 상위 N개) — rank0만 사용
    preview_rows = []
    src_path = args.val_jsonl if args.val_jsonl else args.train_jsonl
    try:
        with open(src_path, "r", encoding="utf-8") as f:
            for i, l in enumerate(f):
                if i >= args.preview_n: break
                preview_rows.append(json.loads(l))
    except Exception:
        pass

    preview_cb = PreviewCallback(
        tokenizer=tok,
        processor=proc,
        preview_samples=preview_rows,
        every_steps=args.preview_steps,
        max_new_tokens=args.preview_max_new_tokens,
        ramp_after=args.preview_ramp_after if args.preview_ramp_after > 0 else None,
        ramp_max_new_tokens=args.preview_ramp_max_new_tokens,
    )

    metrics_cb = MetricsCallback(log_every_steps=10, ema_beta=0.98, hist_every_steps=100)

        # ===== 콘솔에 GC/캐시 상태 요약 출력 (rank0만) =====
    if is_rank0():
        gc_flag = detect_gradient_checkpointing(model)
        print(f"[GC] detected_enabled={gc_flag}")
        print(f"[GC] TrainingArguments.gradient_checkpointing={getattr(targs, 'gradient_checkpointing', None)}")
        print(f"[GC] TrainingArguments.gradient_checkpointing_kwargs={getattr(targs, 'gradient_checkpointing_kwargs', None)}")
        print(f"[GC] model.config.use_cache={getattr(getattr(model, 'config', None), 'use_cache', None)}")
        
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tok,            # processing_class 대신 tokenizer만 넘김(호환성↑)
        callbacks=[preview_cb, metrics_cb],
    )

    trainer.train()

    # 최종 저장은 world process zero만
    if trainer.is_world_process_zero():
        model.save_pretrained(os.path.join(args.output_dir, "lora_stage1"))
        tok.save_pretrained(args.output_dir)
        try:
            proc.save_pretrained(args.output_dir)
        except Exception:
            pass

    if is_rank0():
        print("Done.")


if __name__ == "__main__":
    main()