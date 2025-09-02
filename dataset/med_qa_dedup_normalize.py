# -*- coding: utf-8 -*-
# SLM 전용: 리스트 JSON -> (instruction,input,output) JSONL + 고속 디듈링
# - 최대한 단순: messages/controls 없음
# - 태그는 instruction 맨 앞에만 (옵션)
# - 짧고 간결한 답변, 일관 규칙, 빠른 근사중복 제거

import os, re, json, zlib, hashlib, random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from tqdm import tqdm

# --------------------
# 0) 설정
# --------------------
DIR = '초거대'
SRC = '초거대TTS_이비인후과.json'          # 입력(리스트 JSON)
MID = 'TTS_medqa.slm.jsonl'          # 중간 산출(JSONL)
OUT = 'TTS_medqa.slm.dedup.jsonl'    # 최종(디듈링)

USE_TAGS = False        # True면 instruction 앞에 [DEPT=..][DX=..] 접두 태그 부착
MAX_OUTPUT_SENT = 4    # 답변 최대 문장수 (SLM용 간결화)
USE_SAFETY_TAIL = True # 맨끝에 간단 안전문구 추가(중복 시 자동 억제)

# 디듈링 파라미터(빠름·간단)
JACCARD_TH = 0.92
SEQMATCH_TH = 0.90
BANDS = 4
HAM_TH = 3
SHINGLE_N = 3

random.seed(7)

# --------------------
# 1) 간단 사전/정규화
# --------------------
DEPT_ALIAS = {'비뇨기과':'비뇨의학과','소아청년과':'소아청소년과','\xa0신경과':'신경과','\xa0신경외과':'신경외과'}
DEPT_TO_MACRO = {
    '내과':'IM','감염내과':'IM','혈액내과':'IM','내분비내과':'IM',
    '호흡기내과':'RESP','심장내과':'CV','순환기내과':'CV','류마티스내과':'IM',
    '외과':'SURG','정형외과':'ORTH','신경외과':'NSURG','성형외과':'PLAST',
    '마취통증의학과':'ANES','재활의학과':'REHAB',
    '소아신경과':'PED','소아청소년과':'PED','산부인과':'GYN',
    '이비인후과':'ENT','안과':'OPH','치과':'DENT',
    '신경과':'NEU','정신건강의학과':'PSY',
    '비뇨의학과':'URO','응급의학과':'ER','가정의학과':'FM',
    '방사선과':'RAD','방사선종양학과':'RT','종양내과':'ONC','피부과':'DERM',
    '소아청소년과감염내과':'PED',
}

def _norm_ws(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '').strip())

def normalize_dept(name: str) -> str:
    name = (name or '').replace('\xa0','').strip()
    return DEPT_ALIAS.get(name, name)

def map_dept_macro(raw) -> Optional[str]:
    if not raw: return None
    if isinstance(raw, str): raw = [raw]
    for d in raw:
        d = normalize_dept(d)
        if d in DEPT_TO_MACRO: return DEPT_TO_MACRO[d]
    return None

def slug_ascii(text: str, keep_korean=True) -> str:
    t = (text or '').strip().lower().replace(' ','_')
    t = re.sub(r'[^0-9a-zA-Z가-힣_]+','',t) if keep_korean else re.sub(r'[^0-9a-zA-Z_]+','',t)
    t = re.sub(r'_+','_',t).strip('_')
    return t or 'unk'

def short_hash(text: str, n=6) -> str:
    import hashlib
    return hashlib.sha1((text or '').encode('utf-8')).hexdigest()[:n]

# 괄호/특수 정리(단순)
def sentence_filter(s: str) -> str:
    # 괄호 안 교차 표기 제거, 잡철자/연속공백 정리
    s = re.sub(r'\([^)]*\)', ' ', s)
    s = re.sub(r'[#/+\*\@\$\^\&\[\]=:;,_]+', ' ', s)
    s = _norm_ws(s)
    return s

SAFETY_TAIL = "증상이 지속되거나 악화되면 의료진과 상담을 권합니다."
def add_safety_tail(ans: str) -> str:
    if not USE_SAFETY_TAIL: return ans
    if re.search(r'(상담|진료|검사).*(권|바랍니다|필요)', ans): return ans
    return (ans + ' ' + SAFETY_TAIL).strip()

def trim_sentences(text: str, max_sent: int) -> str:
    sents = re.split(r'(?<=[.!?。？！])\s+', _norm_ws(text))
    if len(sents) <= max_sent: return _norm_ws(text)
    return _norm_ws(' '.join(sents[:max_sent]))

# --------------------
# 2) SLM용 레코드 변환
# --------------------
def build_instruction(sample: Dict[str,Any]) -> str:
    base_q = _norm_ws(sample.get('generated_text') or '')
    if not base_q:
        dk = (sample.get('disease_name') or {}).get('kor') or ''
        de = (sample.get('disease_name') or {}).get('eng') or ''
        topic = dk or de or '해당 질환'
        base_q = f"{topic}의 전파 경로, 주요 증상, 1차 치료를 간단히 설명해 주세요."
    q = sentence_filter(base_q)
    if USE_TAGS:
        dept = map_dept_macro(sample.get('department'))
        dk = (sample.get('disease_name') or {}).get('kor') or ''
        de = (sample.get('disease_name') or {}).get('eng') or ''
        dx = slug_ascii(dk, True) if dk else (slug_ascii(de, False) if de else f"dx_{short_hash(q)}")
        tags = []
        if dept: tags.append(f"[DEPT={dept}]")
        if dx:   tags.append(f"[DX={dx}]")
        if tags: q = "".join(tags) + " " + q
    return q

def build_output(sample: Dict[str,Any]) -> str:
    ans = _norm_ws(sample.get('answer') or '')
    ans = sentence_filter(ans)
    ans = trim_sentences(ans, MAX_OUTPUT_SENT)
    ans = add_safety_tail(ans)
    return ans

def make_slm_record(sample: Dict[str,Any]) -> Optional[Dict[str,str]]:
    instr = build_instruction(sample)
    out = build_output(sample)
    if len(instr) < 4 or len(out) < 4:
        return None
    return {"instruction": instr, "input": "", "audio_path":sample["audio_path"], "output": out}

# --------------------
# 3) 변환 I/O
# --------------------
def convert_listjson_to_jsonl(in_json: str, out_jsonl: str):
    src, dst = Path(in_json), Path(out_jsonl)
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = json.load(src.open('r', encoding='utf-8'))
    kept = 0
    with dst.open('w', encoding='utf-8') as fo:
        for it in tqdm(data, desc='convert→slm'):
            rec = make_slm_record(it)
            if rec:
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
    print(f"[convert] kept={kept} → {dst}")

# --------------------
# 4) 고속 디듈링(간결판)
#   - instruction+output 결합 텍스트 기준
#   - 해시(정확중복) + SimHash LSH 후보 축소 + (Jaccard/길이패널티) 최종확인
# --------------------
_ws = re.compile(r'\s+')
def _ntext(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\[[A-Z]+=[^\]]+\]', ' ', s)     # [DEPT=..][DX=..] 제거
    s = re.sub(r'[^0-9a-z가-힣\s]', ' ', s)
    return _ws.sub(' ', s).strip()

def _shingles(s: str, n=3) -> List[str]:
    toks = _ntext(s).split()
    if len(toks) < n: return [" ".join(toks)] if toks else []
    return [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]

def _jacc(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B: return 0.0
    inter = len(A & B)
    if inter == 0: return 0.0
    return inter / (len(A) + len(B) - inter)

def _seqratio(a: str, b: str) -> float:
    a, b = _ntext(a), _ntext(b)
    if not a or not b: return 0.0
    la, lb = len(a), len(b)
    set_sim = _jacc(a.split(), b.split())
    len_pen = min(la, lb) / max(la, lb)
    return 0.5*set_sim + 0.5*len_pen

def _simhash(features: List[str]) -> int:
    v = [0]*64
    for f in features:
        h = zlib.adler32(f.encode('utf-8'))
        h64 = ((h & 0xffffffff) << 32) | ((h * 2654435761) & 0xffffffff)
        for i in range(64):
            v[i] += 1 if ((h64 >> i) & 1) else -1
    out = 0
    for i,val in enumerate(v):
        if val >= 0: out |= (1<<i)
    return out

def _bands(h: int, bands=BANDS, bits=64):
    r = bits // bands
    for i in range(bands):
        yield (i, (h >> (i*r)) & ((1<<r)-1))

def _ham(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def dedupe_jsonl_fast(in_jsonl: str, out_jsonl: str,
                      jac_th=JACCARD_TH, seq_th=SEQMATCH_TH,
                      bands=BANDS, ham_th=HAM_TH, shingle_n=SHINGLE_N):
    src, dst = Path(in_jsonl), Path(out_jsonl)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open('r', encoding='utf-8') as f:
        total = sum(1 for _ in f)

    seen_exact = set()
    band_buckets = [dict() for _ in range(bands)]
    docs_text, docs_sh, docs_shg = [], [], []
    kept = 0

    with src.open('r', encoding='utf-8') as fin, dst.open('w', encoding='utf-8') as fout:
        for line in tqdm(fin, total=total, desc='dedupe-fast'):
            try:
                rec = json.loads(line)
            except:
                continue
            combo = (rec.get('instruction','') + "\n\n" + rec.get('output','')).strip()
            ntxt = _ntext(combo)
            if len(ntxt) < 5: continue

            crc = zlib.crc32(ntxt.encode('utf-8'))
            sig = (crc, len(ntxt))
            if sig in seen_exact:  # 정확 중복
                continue
            seen_exact.add(sig)

            shg = _shingles(ntxt, n=shingle_n)
            sh = _simhash(shg) if shg else _simhash([ntxt])

            cand = set()
            for idx, key in _bands(sh, bands=bands):
                bucket = band_buckets[idx].get(key)
                if bucket: cand.update(bucket)

            dup = False
            for cid in cand:
                if _ham(sh, docs_sh[cid]) <= ham_th:
                    jv = _jacc(shg, docs_shg[cid]) if shg and docs_shg[cid] else 0.0
                    sv = _seqratio(ntxt, docs_text[cid])
                    if jv >= jac_th or sv >= seq_th:
                        dup = True
                        break
            if dup: continue

            did = len(docs_text)
            docs_text.append(ntxt); docs_sh.append(sh); docs_shg.append(shg)
            for idx, key in _bands(sh, bands=bands):
                band_buckets[idx].setdefault(key, []).append(did)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[dedupe] kept={kept} -> {dst} (J>={jac_th}, S>={seq_th}, bands={bands}, ham<={ham_th})")

# --------------------
# 5) 실행
# --------------------
convert_listjson_to_jsonl(os.path.join(DIR, SRC), MID)
dedupe_jsonl_fast(MID, OUT)
print("[done]", OUT)