#!/usr/bin/env python3
"""
단순화된 Ultravox 훈련 스크립트
"""

import os
import argparse
import logging
import yaml
from pathlib import Path

from training.trainer import train_model
from model.config import Config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_default_config(output_path: str, **kwargs):
    """기본 설정 파일 생성"""
    
    config_dict = {
        "model": {
            "text_model_name": kwargs.get("text_model", "meta-llama/Llama-3.1-8B-Instruct"),
            "audio_model_name": kwargs.get("audio_model", "openai/whisper-medium"),
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "audio_use_lora": False,
            "projector_hidden_size": 1024,
            "projector_output_size": 4096,
            "projector_dropout": 0.1,
            "audio_sample_rate": 16000,
            "audio_max_length": 30.0,
            "torch_dtype": "bfloat16",
            "device_map": "auto"
        },
        "training": {
            "batch_size": kwargs.get("batch_size", 4),
            "gradient_accumulation_steps": 1,
            "learning_rate": kwargs.get("learning_rate", 1e-4),
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "max_steps": kwargs.get("max_steps", 1000),
            "save_steps": 500,
            "eval_steps": 100,
            "logging_steps": 10,
            "optimizer": "adamw_torch",
            "lr_scheduler": "cosine",
            "fp16": True,
            "bf16": False,
            "output_dir": kwargs.get("output_dir", "./checkpoints"),
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "logging_dir": "./logs",
            "report_to": "tensorboard"
        },
        "data": {
            "train_dataset_path": kwargs.get("train_dataset", ""),
            "eval_dataset_path": kwargs.get("eval_dataset", ""),
            "task": kwargs.get("task", "transcription"),
            "max_audio_length": 30.0,
            "sample_rate": 16000,
            "num_workers": 4,
            "template_kwargs": {}
        }
    }
    
    # 설정 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"기본 설정 파일이 {output_path}에 생성되었습니다.")

def main():
    parser = argparse.ArgumentParser(description="단순화된 Ultravox 훈련")
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--create_config", type=str, help="기본 설정 파일 생성 경로")
    
    # 모델 설정
    parser.add_argument("--text_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="텍스트 모델")
    parser.add_argument("--audio_model", type=str, default="openai/whisper-medium", help="오디오 모델")
    
    # 데이터 설정
    parser.add_argument("--train_dataset", type=str, help="훈련 데이터셋 경로")
    parser.add_argument("--eval_dataset", type=str, help="평가 데이터셋 경로")
    parser.add_argument("--task", type=str, default=None, 
                       choices=["transcription", "qa", "audio_info_qa", "medical_qa", None],
                       help="태스크 유형 (None이면 멀티태스크 학습)")
    
    # 훈련 설정
    parser.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="학습률")
    parser.add_argument("--max_steps", type=int, default=1000, help="최대 스텝 수")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="출력 디렉토리")
    
    args = parser.parse_args()
    
    # 기본 설정 파일 생성
    if args.create_config:
        create_default_config(
            args.create_config,
            text_model=args.text_model,
            audio_model=args.audio_model,
            train_dataset=args.train_dataset,
            eval_dataset=args.eval_dataset,
            task=args.task,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            output_dir=args.output_dir
        )
        return
    
    # 설정 파일 로드
    if args.config:
        if not os.path.exists(args.config):
            logger.error(f"설정 파일이 존재하지 않습니다: {args.config}")
            return
        
        config = Config.from_yaml(args.config)
    else:
        # 명령행 인자로 설정 생성
        if not args.train_dataset:
            logger.error("--train_dataset이 필요합니다.")
            return
        
        config_dict = {
            "model": {
                "text_model_name": args.text_model,
                "audio_model_name": args.audio_model,
                "use_lora": True,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "audio_use_lora": False,
                "projector_hidden_size": 1024,
                "projector_output_size": 4096,
                "projector_dropout": 0.1,
                "audio_sample_rate": 16000,
                "audio_max_length": 30.0,
                "torch_dtype": "bfloat16",
                "device_map": "auto"
            },
            "training": {
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": 1,
                "learning_rate": args.learning_rate,
                "weight_decay": 0.01,
                "warmup_steps": 100,
                "max_steps": args.max_steps,
                "save_steps": 500,
                "eval_steps": 100,
                "logging_steps": 10,
                "optimizer": "adamw_torch",
                "lr_scheduler": "cosine",
                "fp16": True,
                "bf16": False,
                "output_dir": args.output_dir,
                "save_total_limit": 3,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "logging_dir": "./logs",
                "report_to": "tensorboard"
            },
            "data": {
                "train_dataset_path": args.train_dataset,
                "eval_dataset_path": args.eval_dataset,
                "task": args.task,
                "max_audio_length": 30.0,
                "sample_rate": 16000,
                "num_workers": 4,
                "template_kwargs": {}
            }
        }
        
        config = Config.from_dict(config_dict)
    
    # 출력 디렉토리 생성
    os.makedirs(config.training.output_dir, exist_ok=True)
    os.makedirs(config.training.logging_dir, exist_ok=True)
    
    # 설정 정보 출력
    logger.info("=== 훈련 설정 ===")
    logger.info(f"텍스트 모델: {config.model.text_model_name}")
    logger.info(f"오디오 모델: {config.model.audio_model_name}")
    if config.data.task is None:
        logger.info("태스크: 멀티태스크 학습")
    else:
        logger.info(f"태스크: {config.data.task}")
    logger.info(f"훈련 데이터: {config.data.train_dataset_path}")
    if config.data.eval_dataset_path:
        logger.info(f"평가 데이터: {config.data.eval_dataset_path}")
    logger.info(f"배치 크기: {config.training.batch_size}")
    logger.info(f"학습률: {config.training.learning_rate}")
    logger.info(f"최대 스텝: {config.training.max_steps}")
    logger.info(f"출력 디렉토리: {config.training.output_dir}")
    
    # 훈련 실행
    try:
        trainer = train_model(config)
        logger.info("훈련이 성공적으로 완료되었습니다.")
    except Exception as e:
        logger.error(f"훈련 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()

