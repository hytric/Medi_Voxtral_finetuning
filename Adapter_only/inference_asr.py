# -*- coding: utf-8 -*-
import os
import json
import argparse
import unicodedata
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from peft import PeftModel

def _norm(s: str) -> str:
    """텍스트 정규화"""
    s = unicodedata.normalize("NFKC", s)
    return "".join(ch for ch in s if not ch.isspace())

def cer(ref: str, hyp: str) -> float:
    """Character Error Rate 계산"""
    r, h = _norm(ref), _norm(hyp)
    R, H = len(r), len(h)
    if R == 0:
        return 0.0 if H == 0 else 1.0
    
    dp = [[0]*(H+1) for _ in range(R+1)]
    for i in range(R+1): 
        dp[i][0] = i
    for j in range(H+1): 
        dp[0][j] = j
    
    for i in range(1, R+1):
        for j in range(1, H+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    
    return dp[R][H] / R

def wer(ref: str, hyp: str) -> float:
    """Word Error Rate 계산"""
    r = (ref or "").strip().split()
    h = (hyp or "").strip().split()
    R, H = len(r), len(h)
    if R == 0:
        return 0.0 if H == 0 else 1.0
    
    dp = [[0]*(H+1) for _ in range(R+1)]
    for i in range(R+1): 
        dp[i][0] = i
    for j in range(H+1): 
        dp[0][j] = j
    
    for i in range(1, R+1):
        for j in range(1, H+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    
    return dp[R][H] / R

@torch.no_grad()
def inference_single(model, processor, audio_path: str, model_id: str, language: str = "ko", max_new_tokens: int = 256):
    """단일 오디오 파일에 대한 음성 인식"""
    device = next(model.parameters()).device
    
    # 입력 준비
    enc = processor.apply_transcription_request(
        language=language, 
        audio=audio_path, 
        model_id=model_id, 
        return_tensors="pt"
    )
    
    input_ids = enc["input_ids"].to(device)
    input_features = enc["input_features"].to(device)
    
    # 생성
    outputs = model.generate(
        input_ids=input_ids,
        input_features=input_features,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        num_beams=1,
        use_cache=False,
    )
    
    # 디코딩
    generated_ids = outputs[0, input_ids.shape[1]:]
    transcription = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return transcription

def main():
    parser = argparse.ArgumentParser(description="Voxtral ASR Inference")
    parser.add_argument("--base_model", type=str, default="mistralai/Voxtral-Mini-3B-2507", help="Base model ID")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to trained adapter")
    parser.add_argument("--audio_path", type=str, help="Path to single audio file")
    parser.add_argument("--test_jsonl", type=str, help="Path to test JSONL file")
    parser.add_argument("--output_path", type=str, help="Path to save results")
    parser.add_argument("--language", type=str, default="ko", help="Language code")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens for generation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    print(f"🔥 Voxtral ASR Inference 시작")
    print(f"Base Model: {args.base_model}")
    print(f"Adapter Path: {args.adapter_path}")
    print(f"Device: {args.device}")
    
    # 모델 및 프로세서 로드
    print("📥 모델 로딩 중...")
    processor = AutoProcessor.from_pretrained(args.base_model)
    
    base_model = VoxtralForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        low_cpu_mem_usage=True
    )
    
    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    
    print("✅ 모델 로딩 완료!")
    
    # 단일 파일 추론
    if args.audio_path:
        print(f"\n🎵 단일 파일 추론: {args.audio_path}")
        
        if not os.path.exists(args.audio_path):
            print(f"❌ 오디오 파일을 찾을 수 없습니다: {args.audio_path}")
            return
        
        transcription = inference_single(
            model=model,
            processor=processor,
            audio_path=args.audio_path,
            model_id=args.base_model,
            language=args.language,
            max_new_tokens=args.max_new_tokens
        )
        
        print(f"📝 결과: {transcription}")
    
    # 테스트 데이터셋 평가
    if args.test_jsonl:
        print(f"\n📊 테스트 데이터셋 평가: {args.test_jsonl}")
        
        if not os.path.exists(args.test_jsonl):
            print(f"❌ 테스트 파일을 찾을 수 없습니다: {args.test_jsonl}")
            return
        
        results = []
        total_cer = 0.0
        total_wer = 0.0
        count = 0
        
        with open(args.test_jsonl, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    audio_path = data.get("audio") or data.get("audio_path") or ""
                    reference = data.get("text") or data.get("transcript") or ""
                    
                    if not audio_path or not reference:
                        continue
                    
                    if not os.path.exists(audio_path):
                        print(f"⚠️  파일 없음: {audio_path}")
                        continue
                    
                    # 추론 수행
                    hypothesis = inference_single(
                        model=model,
                        processor=processor,
                        audio_path=audio_path,
                        model_id=args.base_model,
                        language=args.language,
                        max_new_tokens=args.max_new_tokens
                    )
                    
                    # 메트릭 계산
                    cer_score = cer(reference, hypothesis)
                    wer_score = wer(reference, hypothesis)
                    
                    total_cer += cer_score
                    total_wer += wer_score
                    count += 1
                    
                    result = {
                        "audio_path": audio_path,
                        "reference": reference,
                        "hypothesis": hypothesis,
                        "cer": cer_score,
                        "wer": wer_score
                    }
                    results.append(result)
                    
                    if count % 10 == 0:
                        print(f"처리 완료: {count}개")
                        print(f"현재 평균 CER: {total_cer/count:.4f}")
                        print(f"현재 평균 WER: {total_wer/count:.4f}")
                    
                except Exception as e:
                    print(f"⚠️  라인 {line_num} 처리 중 오류: {e}")
                    continue
        
        if count > 0:
            avg_cer = total_cer / count
            avg_wer = total_wer / count
            
            print(f"\n📈 최종 결과:")
            print(f"총 처리된 샘플: {count}개")
            print(f"평균 CER: {avg_cer:.4f}")
            print(f"평균 WER: {avg_wer:.4f}")
            
            # 결과 저장
            if args.output_path:
                with open(args.output_path, "w", encoding="utf-8") as f:
                    for result in results:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"💾 결과 저장됨: {args.output_path}")
        else:
            print("❌ 처리된 샘플이 없습니다.")

if __name__ == "__main__":
    main()
