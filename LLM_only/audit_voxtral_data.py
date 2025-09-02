# audit_voxtral_data.py
import os, sys, json, random

def detect_task(ex):
    if "transcript" in ex and isinstance(ex["transcript"], str) and ex["transcript"].strip():
        return "asr"
    return "sqa" if "audio_path" in ex else "tqa"

def normalize_target(ex):
    for k in ("output","answer","label","target"):
        v = ex.get(k)
        if isinstance(v,str) and v.strip():
            return v.strip()
        if isinstance(v,list) and v:
            return " ".join(map(lambda x: str(x).strip(), v)).strip()
    return None

def audit(files):
    stats = {"asr":0,"sqa":0,"tqa":0}
    bad = 0
    examples = {"asr":[], "sqa":[], "tqa":[]}
    missing_audio = 0

    for fn in files:
        if not os.path.exists(fn):
            print(f"[MISS] {fn}")
            continue
        print(f"[SCAN] {fn}")
        with open(fn,encoding="utf-8") as f:
            for ln,l in enumerate(f,1):
                if not l.strip(): 
                    continue
                try:
                    ex=json.loads(l)
                except Exception as e:
                    print(f"[JSONERR] {fn}:{ln}  {e}")
                    bad+=1; 
                    continue

                t = detect_task(ex)
                if t=="asr":
                    ap, tr = ex.get("audio_path"), ex.get("transcript")
                    if not (isinstance(ap,str) and ap.strip() and isinstance(tr,str) and tr.strip()):
                        print(f"[BAD-ASR] {fn}:{ln} keys={list(ex.keys())}")
                        bad+=1; 
                        continue
                    if not os.path.exists(ap):
                        missing_audio+=1
                else:
                    tgt = normalize_target(ex)
                    if not tgt:
                        print(f"[BAD-CHAT] {fn}:{ln}  keys={list(ex.keys())}")
                        bad+=1; 
                        continue

                stats[t]+=1
                if len(examples[t])<3: 
                    examples[t].append(ex)

    print("\n=== SUMMARY ===")
    total = sum(stats.values())
    print(f"total={total:,}, bad={bad:,}, missing_audio={missing_audio:,}")
    print(f"asr={stats['asr']:,}, sqa={stats['sqa']:,}, tqa={stats['tqa']:,}")

    for t in ("asr","sqa","tqa"):
        print(f"\n[{t}] sample examples")
        for i,ex in enumerate(examples[t],1):
            show = {k:ex.get(k) for k in ("audio_path","transcript","instruction","input","output","answer","label","target")}
            print(f"- ex{i}: {show}")

if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage: python audit_voxtral_data.py file1.jsonl [file2.jsonl ...]")
        sys.exit(1)
    audit(sys.argv[1:])