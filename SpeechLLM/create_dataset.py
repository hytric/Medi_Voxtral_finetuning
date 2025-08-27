#!/usr/bin/env python3
"""
단순화된 데이터셋 생성 스크립트
JSON Lines 형식으로 데이터셋을 생성합니다.
"""

import os
import json
import argparse
import pandas as pd
import torchaudio
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_dataset_item(wav_path, text, output, task="transcription", **kwargs):
    """
    데이터셋 아이템 생성
    
    Args:
        wav_path: 오디오 파일 경로
        text: 입력 텍스트 (질문, 컨텍스트 등)
        output: 출력 텍스트 (답변, 전사 등)
        task: 태스크 유형
        **kwargs: 추가 필드들
    
    Returns:
        dict: 데이터셋 아이템
    """
    if not os.path.exists(wav_path):
        logger.warning(f"오디오 파일이 존재하지 않습니다: {wav_path}")
        return None
    
    try:
        # 오디오 정보 가져오기
        info = torchaudio.info(wav_path)
        duration = info.num_frames / info.sample_rate if info.num_frames and info.sample_rate else 0.0
        
        item = {
            "audio_filepath": wav_path,
            "text": str(text) if text is not None else "",
            "output": str(output),
            "task": task,
            "duration": duration,
            **kwargs
        }
        
        return item
        
    except Exception as e:
        logger.error(f"오디오 파일 처리 실패 {wav_path}: {e}")
        return None

def create_dataset_from_scp(scp_path, output_path, task="transcription", **kwargs):
    """
    SCP 파일에서 데이터셋 생성
    
    Args:
        scp_path: SCP 파일 경로 (wav_path|text 형식)
        output_path: 출력 JSON 파일 경로
        task: 태스크 유형
        **kwargs: 추가 필드들
    """
    logger.info(f"SCP 파일을 읽는 중: {scp_path}")
    
    total_items = 0
    with open(scp_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|')
            if len(parts) != 2:
                logger.warning(f"잘못된 형식의 라인 {line_num}: {line}")
                continue
            
            wav_path, text = parts[0], parts[1]
            
            # 전사 태스크의 경우 text가 output이 됨
            if task == "transcription":
                output = text
                text = ""
            else:
                output = text  # 기본값
            
            item = build_dataset_item(
                wav_path=wav_path,
                text=text,
                output=output,
                task=task,
                **kwargs
            )
            
            if item is not None:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_items += 1
    
    logger.info(f"✅ 데이터셋 생성 완료: {total_items}개 항목이 {output_path}에 저장되었습니다.")

def create_dataset_from_csv(csv_path, output_path, wav_dir=None, task="transcription", **kwargs):
    """
    CSV 파일에서 데이터셋 생성
    
    Args:
        csv_path: CSV 파일 경로
        output_path: 출력 JSON 파일 경로
        wav_dir: WAV 파일 디렉토리 (선택사항)
        task: 태스크 유형
        **kwargs: 추가 필드들
    """
    logger.info(f"CSV 파일을 읽는 중: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"CSV 컬럼: {list(df.columns)}")
    
    # WAV 경로 컬럼 확인
    has_wav_path = 'wav_path' in df.columns or 'audio_filepath' in df.columns
    wav_col = 'wav_path' if 'wav_path' in df.columns else 'audio_filepath'
    
    # 텍스트 컬럼 확인
    text_col = None
    for col in ['text', 'question', 'input']:
        if col in df.columns:
            text_col = col
            break
    
    # 출력 컬럼 확인
    output_col = None
    for col in ['output', 'answer', 'transcript', 'sentence']:
        if col in df.columns:
            output_col = col
            break
    
    if output_col is None:
        logger.error("출력 컬럼을 찾을 수 없습니다. 다음 중 하나가 필요합니다: output, answer, transcript, sentence")
        return
    
    total_items = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        if has_wav_path:
            # WAV 경로가 CSV에 있는 경우
            for _, row in tqdm(df.iterrows(), total=len(df), desc="데이터셋 생성 중"):
                wav_path = str(row[wav_col])
                
                if not os.path.isabs(wav_path) and wav_dir:
                    wav_path = os.path.join(wav_dir, wav_path)
                
                text = str(row[text_col]) if text_col else ""
                output = str(row[output_col])
                
                item = build_dataset_item(
                    wav_path=wav_path,
                    text=text,
                    output=output,
                    task=task,
                    **kwargs
                )
                
                if item is not None:
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_items += 1
        else:
            # WAV 디렉토리에서 파일을 찾는 경우
            if not wav_dir or not os.path.isdir(wav_dir):
                logger.error(f"WAV 디렉토리가 없습니다: {wav_dir}")
                return
            
            wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
            wav_files.sort()
            
            limit = min(len(wav_files), len(df))
            for i in tqdm(range(limit), desc="데이터셋 생성 중"):
                wav_path = os.path.join(wav_dir, wav_files[i])
                text = str(df.iloc[i][text_col]) if text_col else ""
                output = str(df.iloc[i][output_col])
                
                item = build_dataset_item(
                    wav_path=wav_path,
                    text=text,
                    output=output,
                    task=task,
                    **kwargs
                )
                
                if item is not None:
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_items += 1
    
    logger.info(f"✅ 데이터셋 생성 완료: {total_items}개 항목이 {output_path}에 저장되었습니다.")

def main():
    parser = argparse.ArgumentParser(description="단순화된 데이터셋 생성")
    parser.add_argument("--input_scp", type=str, help="SCP 파일 경로")
    parser.add_argument("--input_csv", type=str, help="CSV 파일 경로")
    parser.add_argument("--wav_dir", type=str, help="WAV 파일 디렉토리")
    parser.add_argument("--output_json", type=str, required=True, help="출력 JSON 파일 경로")
    parser.add_argument("--task", type=str, default="transcription", 
                       choices=["transcription", "qa", "audio_info_qa", "medical_qa"],
                       help="태스크 유형")
    parser.add_argument("--target_language", type=str, help="번역 대상 언어 (translation 태스크용)")
    
    args = parser.parse_args()
    
    # 추가 필드들
    extra_fields = {}
    if args.target_language:
        extra_fields["target_language"] = args.target_language
    
    if args.input_scp:
        create_dataset_from_scp(
            scp_path=args.input_scp,
            output_path=args.output_json,
            task=args.task,
            **extra_fields
        )
    elif args.input_csv:
        create_dataset_from_csv(
            csv_path=args.input_csv,
            output_path=args.output_json,
            wav_dir=args.wav_dir,
            task=args.task,
            **extra_fields
        )
    else:
        logger.error("--input_scp 또는 --input_csv 중 하나를 지정해야 합니다.")

if __name__ == "__main__":
    main()
