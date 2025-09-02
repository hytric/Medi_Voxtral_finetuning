# -*- coding: utf-8 -*-
"""
preflight_voxtral.py
- Voxtral 멀티태스크 학습 전 환경/데이터/라이브러리 호환성 점검 & 자동 보정
- 안전한 실행 커맨드 생성 (옵션: --autorun 으로 즉시 실행)

사용 예:
python preflight_voxtral.py \
  --nproc 2 \
  --model_id ${MODEL_PATH:-/shared/models/Voxtral-Mini-3B-2507} \
  --train_jsonl data/data/ASR.jsonl,data/data/SQA.jsonl,data/data/TQA.jsonl \
  --val_jsonl   data/data/ASR.dev.jsonl,data/data/SQA.dev.jsonl,data/data/TQA.dev.jsonl \
  --out_dir ckpt_voxtral_mt \
  --per_device_bsz 2 --grad_accum 16 \
  --max_len 1280 --eval_max_new_tokens 256 \
  --prefer_bf16 --tf32 --grad_ckpt \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --num_workers auto --prefetch_factor 4 --pin_memory \
  # --autorun   # (원하면 자동 실행)
"""

import os, sys, json, argparse, shutil, subprocess, random, time, platform
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

def now(): return time.strftime("%m-%d %H:%M:%S")

# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="ckpt_voxtral_mt")

    ap.add_argument("--per_device_bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=1280)
    ap.add_argument("--eval_max_new_tokens", type=int, default=256)

    ap.add_argument("--prefer_bf16", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--num_workers", default="auto", help="int 또는 'auto'")
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true")

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_steps", type=int, default=1500)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--autorun", action="store_true", help="프리플라이트 후 즉시 학습 실행")
    ap.add_argument("--train_script", type=str, default="train_multitask_voxtral.py")
    ap.add_argument("--skip_schema_errors", action="store_true", help="스키마 오류 샘플은 스킵")
    ap.add_argument("--sample_check", type=int, default=200, help="스키마 검사 샘플 수")

    return ap.parse_args()

# -------------------------
# Utilities
# -------------------------
def split_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def file_exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False

def sizeof_fmt(num, suffix="B"):
    for unit in ["","K","M","G","T"]:
        if abs(num) < 1024.0: return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"

# -------------------------
# Checks
# -------------------------
@dataclass
class EnvReport:
    cuda_available: bool
    device_count: int
    bf16_supported: bool
    kernel: str
    transformers_version: str
    trainingargs_signature: List[str]
    deps_ok: bool
    missing_deps: List[str]

def check_env() -> EnvReport:
    print(f"[{now()}] [ENV] checking GPU/precision/libraries...")
    missing = []
    try:
        import torch
        cuda = torch.cuda.is_available()
        devc = torch.cuda.device_count() if cuda else 0
        bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception as e:
        print(f"[ENV] PyTorch not importable: {e}")
        cuda, devc, bf16 = False, 0, False
        missing.append("torch")

    try:
        import transformers
        from transformers import TrainingArguments
        tver = transformers.__version__
        sig = list(transformers.TrainingArguments.__init__.__code__.co_varnames)
    except Exception as e:
        print(f"[ENV] transformers not importable: {e}")
        tver, sig = "unknown", []
        missing.append("transformers")

    try:
        import mistral_common  # type: ignore
    except Exception:
        missing.append("mistral-common")

    kernel = platform.release()
    deps_ok = (len(missing) == 0)
    rep = EnvReport(cuda, devc, bf16, kernel, tver, sig, deps_ok, missing)
    print(f"  - cuda_available={rep.cuda_available} device_count={rep.device_count} bf16_supported={rep.bf16_supported}")
    print(f"  - kernel={rep.kernel} transformers={rep.transformers_version}")
    if not deps_ok:
        print(f"  - missing deps: {rep.missing_deps} (예: pip install mistral-common)")
    return rep

@dataclass
class DataReport:
    train_files: List[str]
    val_files: List[str]
    missing_val: List[str]
    schema_ok: bool
    schema_errors: List[str]
    counts: Dict[str, int]

REQUIRED_KEYS = {
    "asr": ["audio_path", "transcript"],
    "sqa": ["audio_path", "output"],   # instruction 미사용
    "tqa": ["output"],                 # instruction(+input)→user, output→assistant
}

def detect_task(ex: Dict[str, Any]) -> str:
    if "audio_path" in ex and "transcript" in ex: return "asr"
    if "audio_path" in ex and ex.get("output") and not ex.get("instruction"): return "sqa"
    return "tqa"

def check_jsonl_schema(files: List[str], sample_n=200) -> (bool, List[str], Dict[str,int]):
    errs = []
    counts = {"asr":0, "sqa":0, "tqa":0}
    for f in files:
        try:
            sz = os.path.getsize(f)
            print(f"[{now()}] [SCAN] {f} ({sizeof_fmt(sz)})")
            with open(f, "r", encoding="utf-8") as rf:
                lines = rf.readlines()
            # 샘플
            idxs = list(range(len(lines)))
            random.shuffle(idxs)
            idxs = idxs[:min(sample_n, len(lines))]
            for i in idxs:
                try:
                    ex = json.loads(lines[i])
                except Exception as e:
                    errs.append(f"{f}#L{i+1}: JSON parse error: {e}")
                    continue
                t = detect_task(ex)
                counts[t]+=1
                # 필수키 검사
                for k in REQUIRED_KEYS[t]:
                    if k not in ex or (isinstance(ex[k], str) and ex[k].strip()==""):
                        errs.append(f"{f}#L{i+1}: task={t} missing key: {k}")
        except Exception as e:
            errs.append(f"{f}: read error: {e}")
    ok = len([e for e in errs if "missing key" in e or "parse error" in e or "read error" in e])==0
    return ok, errs, counts

def check_data(train_csv: str, val_csv: str, sample_n=200) -> DataReport:
    trains = split_csv(train_csv)
    vals   = split_csv(val_csv) if val_csv else []
    # 존재 확인
    missing_val = [v for v in vals if not file_exists(v)]
    vals = [v for v in vals if file_exists(v)]
    if missing_val:
        print(f"[{now()}] [DATA] missing val files: {missing_val} -> 평가 비활성 또는 일부만 사용")
    # 스키마 검사(훈련 데이터만 빠르게)
    ok, errs, counts = check_jsonl_schema(trains, sample_n=sample_n)
    if not ok:
        print(f"[{now()}] [DATA] schema issues detected ({len(errs)}). 대표 10개:")
        for m in errs[:10]:
            print("   -", m)
        print("    (옵션 --skip_schema_errors 로 문제 라인 스킵하고 진행 가능)")
    else:
        print(f"[{now()}] [DATA] schema OK. counts(sampled)={counts}")
    return DataReport(trains, vals, missing_val, ok, errs, counts)

# -------------------------
# Recommender
# -------------------------
@dataclass
class Plan:
    use_bf16: bool
    use_fp16: bool
    num_workers: int
    pin_memory: bool
    eval_enabled: bool
    eval_flag: str  # "evaluation_strategy" or "evaluate_during_training" or "none"
    notes: List[str]

def recommend(rep: EnvReport, dat: DataReport, prefer_bf16: bool, num_workers_opt: str, pin_memory_flag: bool) -> Plan:
    notes=[]
    # precision
    use_bf16 = bool(prefer_bf16 and rep.cuda_available and rep.bf16_supported)
    use_fp16 = bool((prefer_bf16 and rep.cuda_available and not rep.bf16_supported) or (not prefer_bf16 and rep.cuda_available))
    if use_bf16:
        use_fp16 = False
    if not rep.cuda_available:
        notes.append("GPU가 보이지 않습니다. login 노드가 아닌 GPU 노드에서 실행하세요.")
    if rep.kernel and rep.kernel.startswith(("5.4.", "4.")):
        notes.append(f"커널 {rep.kernel}: 5.5.0 이상 권장(멈춤/행 이슈 예방).")

    # eval strategy 호환
    if "evaluation_strategy" in rep.trainingargs_signature:
        eval_flag = "evaluation_strategy"
    elif "evaluate_during_training" in rep.trainingargs_signature:
        eval_flag = "evaluate_during_training"
    else:
        eval_flag = "none"
        notes.append("현재 transformers에서 평가전략 인자를 지원하지 않습니다(do_eval만 가능).")

    eval_enabled = bool(len(dat.val_files)>0)

    # num_workers 추천
    if isinstance(num_workers_opt, str) and num_workers_opt.lower()=="auto":
        cpu_cnt = os.cpu_count() or 8
        # 시스템 경고 대비: 과도한 워커는 느려지거나 프리징 → 보수적으로
        base = max(2, cpu_cnt//8)
        base = min(base, 8)
        # 단일 GPU일 때는 4~6, 멀티면 6~8 권장
        if rep.device_count <= 1: base = min(base, 6)
        num_workers = base
        notes.append(f"num_workers 자동 설정: {num_workers} (cpu={cpu_cnt})")
    else:
        try:
            num_workers = max(0, int(num_workers_opt))
        except Exception:
            num_workers = 4
            notes.append("num_workers 파싱 실패 → 4로 설정")

    return Plan(use_bf16, use_fp16, num_workers, bool(pin_memory_flag), eval_enabled, eval_flag, notes)

# -------------------------
# Command builder
# -------------------------
def build_command(args, plan: Plan, dat: DataReport) -> List[str]:
    cmd = [
        "torchrun", f"--nproc_per_node={args.nproc}", args.train_script,
        "--model_id", args.model_id,
        "--train_jsonl", ",".join(dat.train_files),
        "--output_dir", args.out_dir,
        "--per_device_bsz", str(args.per_device_bsz),
        "--grad_accum", str(args.grad_accum),
        "--max_len", str(args.max_len),
        "--eval_max_new_tokens", str(args.eval_max_new_tokens),
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--lr", str(args.lr),
        "--warmup_steps", str(args.warmup_steps),
        "--seed", str(args.seed),
        "--num_workers", str(plan.num_workers),
        "--prefetch_factor", str(args.prefetch_factor),
    ]
    if plan.pin_memory: cmd += ["--pin_memory"]
    if args.cosine: cmd += ["--cosine"]
    if args.grad_ckpt: cmd += ["--grad_ckpt"]
    if args.tf32: cmd += ["--tf32"]

    # precision
    if plan.use_bf16: cmd += ["--bf16"]
    elif plan.use_fp16: cmd += ["--fp16"]

    # eval files
    if plan.eval_enabled:
        cmd += ["--val_jsonl", ",".join(dat.val_files)]

    return cmd

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    # 1) 환경 점검
    rep = check_env()
    if not rep.deps_ok:
        print(f"[{now()}] [HINT] pip install -U transformers peft accelerate soundfile librosa sentencepiece mistral-common")
    if not rep.cuda_available:
        print(f"[{now()}] [FATAL] GPU 미탐지 → GPU 노드에서 실행하거나 CUDA 빌드 PyTorch로 재설치 필요.")
        sys.exit(1)

    # 2) 모델 Processor 로딩만 먼저 시도(토크나이저 백엔드 확인)
    try:
        from transformers import AutoProcessor
        _ = AutoProcessor.from_pretrained(args.model_id)
        print(f"[{now()}] [MODEL] AutoProcessor OK: {args.model_id}")
    except ImportError as e:
        print(f"[{now()}] [FATAL] Processor 로드 실패(백엔드 누락): {e}")
        print("        → pip install mistral-common (설치 후 셸 재시작 권장)")
        sys.exit(1)
    except Exception as e:
        print(f"[{now()}] [WARN] Processor 로드 중 경고: {e}")

    # 3) 데이터 점검(존재/스키마)
    dat = check_data(args.train_jsonl, args.val_jsonl, sample_n=args.sample_check)
    if not dat.schema_ok and not args.skip_schema_errors:
        print(f"[{now()}] [FATAL] 스키마 오류가 있습니다. (--skip_schema_errors 로 무시 가능)")
        sys.exit(1)
    elif not dat.schema_ok and args.skip_schema_errors:
        print(f"[{now()}] [INFO] 스키마 오류가 있어도 계속 진행합니다(문제 라인은 로더에서 스킵되도록 코드 필요).")

    # 4) 추천 설정 산출
    plan = recommend(rep, dat, prefer_bf16=args.prefer_bf16, num_workers_opt=args.num_workers, pin_memory_flag=args.pin_memory)
    for n in plan.notes:
        print(f"[NOTE] {n}")

    # 5) 실행 커맨드 생성
    cmd = build_command(args, plan, dat)

    # 6) 최종 커맨드 출력
    print("\n================= RECOMMENDED COMMAND =================")
    print(" ".join(cmd))
    print("======================================================\n")

    # 7) autorun
    if args.autorun:
        print(f"[{now()}] [RUN] launching training ...\n")
        # 안전하게 동일한 환경변수 세팅(필요시 추가)
        env = os.environ.copy()
        # env["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[{now()}] [ERR] training process failed: {e.returncode}")
            sys.exit(e.returncode)

if __name__ == "__main__":
    main()