# -*- coding: utf-8 -*-
import os, json, argparse, random, gc
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader


from transformers import (
    VoxtralForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,

)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# =========================
# 커스텀 Trainer: DataLoader + evaluate 오버라이드
# =========================
class FastTrainer(Trainer):
    def _build_loader(
        self, dataset, batch_size: int, shuffle: bool, data_collator, num_workers: int,
        prefetch_factor: int, pin_memory: bool, drop_last: bool
    ):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=data_collator,
            num_workers=num_workers,
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=(num_workers > 0),
        )

    def get_train_dataloader(self):
        args = self.args
        return self._build_loader(
            dataset=self.train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            data_collator=self.data_collator,
            num_workers=getattr(args, "dataloader_num_workers", 0),
            prefetch_factor=getattr(args, "dataloader_prefetch_factor", 2),
            pin_memory=getattr(args, "dataloader_pin_memory", True),
            drop_last=getattr(args, "dataloader_drop_last", True),
        )





# ========== 유틸 ==========

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_bridge_linear_fullnames(model) -> List[str]:
    found = []
    for parent_name, parent in model.named_modules():
        pn = parent_name.lower()
        if "multi_modal_projector" in pn or "multimodal_projector" in pn or "mm_projector" in pn:
            for child_name, child in parent.named_modules():
                full = f"{parent_name}.{child_name}".strip(".")
                if isinstance(child, torch.nn.Linear):
                    found.append(full)
    return sorted(set(found))

def preview_some_modules(model, contains="project", limit=60):
    cnt = 0
    for n, _ in model.named_modules():
        if contains.lower() in n.lower():
            print(f"[{cnt:03d}] {n}")
            cnt += 1
            if cnt >= limit:
                print("... (truncated)")
                break



# ========== 데이터셋/콜레이터 ==========

class ASRDataset(Dataset):
    def __init__(self, path: str):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for l in f:
                ex = json.loads(l)
                if ex.get("cache_path") and ex.get("prompt_ids") and ex.get("ref_text"):
                    self.rows.append(ex)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

@dataclass
class DataCollatorCached:
    tokenizer: Any
    max_len: int = 1024
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        input_features = []
        batch_input_ids, batch_attention_mask, batch_labels = [], [], []

        for f in features:
            obj = torch.load(f["cache_path"])  # {input_features: (80,T)}
            input_features.append(obj["input_features"])

            prompt = torch.tensor(f["prompt_ids"], dtype=torch.long)
            ans_ids = self.tokenizer(f["ref_text"], add_special_tokens=False)["input_ids"] + [eos_id]
            ans = torch.tensor(ans_ids, dtype=torch.long)

            full = torch.cat([prompt, ans], dim=0)
            am   = torch.ones_like(full)
            lab  = full.clone()
            lab[: prompt.numel()] = -100

            if full.numel() > self.max_len:
                full = full[:self.max_len]; am = am[:self.max_len]; lab = lab[:self.max_len]

            batch_input_ids.append(full); batch_attention_mask.append(am); batch_labels.append(lab)

        def pad_stack(seqs, pad):
            return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad)

        # input_features는 길이가 다르니 pad 필요 없음(모델이 내부에서 처리). stack만.
        return {
            "input_ids": pad_stack(batch_input_ids, pad_id),
            "attention_mask": pad_stack(batch_attention_mask, 0),
            "labels": pad_stack(batch_labels, -100),
            "input_features": torch.stack(input_features, dim=0),
        }





# ========== 메인 ==========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="mistralai/Voxtral-Mini-3B-2507")
    ap.add_argument("--train_jsonl", type=str, default="dataset/train_data.jsonl")

    ap.add_argument("--output_dir",  type=str, default="ckpt_stage2_asr")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--per_device_bsz", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--language", type=str, default="ko")
    ap.add_argument("--num_workers", type=int, default=min(8,(os.cpu_count() or 8)))
    ap.add_argument("--prefetch_factor", type=int, default=4)

    # pin/drop 토글 가능하도록
    ap.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    ap.add_argument("--no-pin_memory", dest="pin_memory", action="store_false")
    ap.set_defaults(pin_memory=True)
    ap.add_argument("--drop_last", dest="drop_last", action="store_true")
    ap.add_argument("--no-drop_last", dest="drop_last", action="store_false")
    ap.set_defaults(drop_last=True)


    ap.add_argument("--tf32", action="store_true", default=True)




    args = ap.parse_args()
    set_seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    proc = AutoProcessor.from_pretrained(args.model_id)
    tok  = proc.tokenizer
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model_id, device_map="auto", torch_dtype=torch.bfloat16,
        quantization_config=bnb_cfg, low_cpu_mem_usage=True
    )
    model.config.use_cache=False
    # ===== [Flash-Attn 2 + SDP 선호] =====
    from transformers.utils import is_flash_attn_2_available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)
    
    if is_flash_attn_2_available():
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("flash_attention_2")
        else:
            model.config._attn_implementation = "flash_attention_2"
        print("[FA2] Using Flash-Attention 2")
    

    # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = prepare_model_for_kbit_training(model)

    bridge_targets = find_bridge_linear_fullnames(model)
    if not bridge_targets:
        preview_some_modules(model,contains="project")
        raise RuntimeError("multi_modal_projector.* not found")
    for p in model.parameters(): p.requires_grad_(False)
    lconf = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                       target_modules=bridge_targets, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model,lconf)
    model.print_trainable_parameters()
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=False)
            print("[compile] enabled")
        except Exception as e:
            print(f"[compile] skipped: {e}")

    # train_ds = ASRDataset(args.train_jsonl)
    train_ds = ASRDataset("dataset/train_data.cached.jsonl")
    # train_collator = DataCollatorASRTranscription(...)
    train_collator = DataCollatorCached(tokenizer=tok, max_len=args.max_len)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_steps=2000,
        save_total_limit=2,
        gradient_checkpointing=False,
        report_to="tensorboard",
        optim="adamw_torch_fused",
        remove_unused_columns=False,
        
        # DataLoader 파라미터
        dataloader_num_workers=args.num_workers,
        dataloader_prefetch_factor=args.prefetch_factor,
        dataloader_pin_memory=args.pin_memory,
        dataloader_drop_last=args.drop_last,
    )



    trainer = FastTrainer(
        model=model, args=targs,
        train_dataset=train_ds,
        data_collator=train_collator,
        processing_class=proc,
    )

    trainer.train()

    model.save_pretrained(os.path.join(args.output_dir,"lora_stage2_adapter"))
    proc.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()