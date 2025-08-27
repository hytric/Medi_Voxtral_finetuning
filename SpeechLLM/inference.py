#!/usr/bin/env python3
"""
단순화된 Ultravox 추론 스크립트
"""

import os
import argparse
import torch
import torchaudio
import logging
from pathlib import Path

from model.ultravox import SimpleUltravox
from model.config import ModelConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_audio(audio_path: str, sample_rate: int = 16000):
    """
    오디오 파일 로드 및 전처리
    
    Args:
        audio_path: 오디오 파일 경로
        sample_rate: 샘플링 레이트
    
    Returns:
        torch.Tensor: 전처리된 오디오
    """
    try:
        # 오디오 로드
        waveform, sr = torchaudio.load(audio_path)
        
        # 모노로 변환 (필요시)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 샘플링 레이트 변환 (필요시)
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
        # 정규화
        waveform = waveform / torch.max(torch.abs(waveform))
        
        return waveform.squeeze()
        
    except Exception as e:
        logger.error(f"오디오 로드 실패 {audio_path}: {e}")
        raise

def inference_single(model: SimpleUltravox, audio_path: str, text_input: str = "", 
                    system_prompt: str = "", max_new_tokens: int = 512, **kwargs):
    """
    단일 오디오 파일 추론
    
    Args:
        model: 로드된 모델
        audio_path: 오디오 파일 경로
        text_input: 텍스트 입력
        system_prompt: 시스템 프롬프트
        max_new_tokens: 최대 생성 토큰 수
        **kwargs: 추가 생성 파라미터
    
    Returns:
        str: 생성된 텍스트
    """
    # 오디오 로드
    audio = load_audio(audio_path, model.config.audio_sample_rate)
    audio = audio.unsqueeze(0)  # 배치 차원 추가
    
    # 디바이스로 이동
    device = next(model.parameters()).device
    audio = audio.to(device)
    
    # 추론
    with torch.no_grad():
        generated_texts = model.generate(
            audio_input=audio,
            text_input=[text_input],
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    return generated_texts[0] if generated_texts else ""

def main():
    parser = argparse.ArgumentParser(description="단순화된 Ultravox 추론")
    parser.add_argument("--model_path", type=str, required=True, help="모델 경로")
    parser.add_argument("--audio_file", type=str, required=True, help="오디오 파일 경로")
    parser.add_argument("--text_input", type=str, default="", help="텍스트 입력")
    parser.add_argument("--system_prompt", type=str, default="", help="시스템 프롬프트")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="최대 생성 토큰 수")
    parser.add_argument("--temperature", type=float, default=0.7, help="생성 온도")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p 샘플링")
    parser.add_argument("--task", type=str, default="transcription", 
                       choices=["transcription", "qa", "audio_info_qa", "medical_qa"],
                       help="태스크 유형")
    
    args = parser.parse_args()
    
    # 모델 로드
    logger.info(f"모델 로딩 중: {args.model_path}")
    try:
        model = SimpleUltravox.from_pretrained(args.model_path)
        model.eval()
        logger.info("모델 로딩 완료")
    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}")
        return
    
    # 오디오 파일 확인
    if not os.path.exists(args.audio_file):
        logger.error(f"오디오 파일이 존재하지 않습니다: {args.audio_file}")
        return
    
    # 태스크별 기본 텍스트 입력 설정
    if not args.text_input:
        if args.task == "transcription":
            args.text_input = "다음 음성을 텍스트로 전사해주세요:"
        elif args.task == "qa":
            args.text_input = "다음 질문에 답해주세요:"
        elif args.task == "audio_info_qa":
            args.text_input = "다음 음성 정보를 바탕으로 질문에 답해주세요:"
        elif args.task == "medical_qa":
            args.text_input = "다음 의료 관련 질문에 답해주세요:"
    
    # 추론 실행
    logger.info(f"추론 시작: {args.audio_file}")
    try:
        result = inference_single(
            model=model,
            audio_path=args.audio_file,
            text_input=args.text_input,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True
        )
        
        print("\n" + "="*50)
        print("추론 결과:")
        print("="*50)
        print(f"입력 오디오: {args.audio_file}")
        print(f"텍스트 입력: {args.text_input}")
        print(f"생성된 텍스트: {result}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"추론 실패: {e}")
        raise

if __name__ == "__main__":
    main()
