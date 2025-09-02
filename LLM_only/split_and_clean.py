# split_and_clean.py
import json, argparse, random, hashlib
from collections import defaultdict

def norm(s): return " ".join((s or "").split())

def key_user(ex):
    ins = norm(ex.get("instruction",""))
    cond = norm(ex.get("input",""))
    u = ins if ins else ""
    if cond: u = f"{u}\n\n조건: {cond}"
    return u

def hash_group(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_val", required=True)
    ap.add_argument("--out_test", default="")
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--test_ratio", type=float, default=0.00)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    groups = defaultdict(list)
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            # 제거: Q=A
            if norm(ex.get("instruction","")) == norm(ex.get("output","")):
                continue
            # 제거: 지나치게 긴 샘플(옵션) - full_len 계산 안 하는 대신 보수적으로 필터
            if len((ex.get("instruction") or "")) + len((ex.get("input") or "")) + len((ex.get("output") or "")) > 30000:
                continue
            g = hash_group(key_user(ex))
            groups[g].append(ex)

    keys = list(groups.keys()); random.shuffle(keys)
    n = len(keys)
    n_test = int(n * args.test_ratio)
    n_val  = int(n * args.val_ratio)
    test_keys = set(keys[:n_test])
    val_keys  = set(keys[n_test:n_test+n_val])
    train_keys= set(keys[n_test+n_val:])

    def choose_one_if_conflict(lst):
        # 같은 user에 서로 다른 output이 있으면, 더 긴 output(정보량 높은 쪽)만 남김
        outs = {}
        for ex in lst:
            o = norm(ex.get("output",""))
            outs[o] = max(outs.get(o,0), len(o))
        best_out = max(outs, key=outs.get)
        # best_out만 남긴다
        kept = [ex for ex in lst if norm(ex.get("output","")) == best_out]
        return kept

    def write_subset(keys_set, path):
        cnt = 0
        with open(path, "w", encoding="utf-8") as w:
            for k in keys_set:
                items = choose_one_if_conflict(groups[k])
                for ex in items:
                    w.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    cnt += 1
        print(f"[write] {path}: {cnt} rows")

    write_subset(train_keys, args.out_train)
    write_subset(val_keys,   args.out_val)
    if args.out_test:
        write_subset(test_keys,  args.out_test)

if __name__ == "__main__":
    main()