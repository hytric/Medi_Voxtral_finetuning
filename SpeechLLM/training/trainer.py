"""
훈련 로직
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    Trainer, TrainingArguments, 
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
import wandb
import logging
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional

from ..data.dataset import SimpleAudioTextDataset, create_dataloader
from ..model.ultravox import SimpleUltravox
from ..model.config import Config

logger = logging.getLogger(__name__)

class SimpleUltravoxTrainer:
    """단순화된 Ultravox 훈련기"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 초기화
        self.model = SimpleUltravox(config.model)
        self.model.to(self.device)
        
        # 데이터셋 초기화
        self.train_dataset = SimpleAudioTextDataset(
            data_path=config.data.train_dataset_path,
            task=config.data.task,
            max_audio_length=config.data.max_audio_length,
            sample_rate=config.data.sample_rate,
            **config.data.template_kwargs
        )
        
        if config.data.eval_dataset_path:
            self.eval_dataset = SimpleAudioTextDataset(
                data_path=config.data.eval_dataset_path,
                task=config.data.task,
                max_audio_length=config.data.max_audio_length,
                sample_rate=config.data.sample_rate,
                **config.data.template_kwargs
            )
        else:
            self.eval_dataset = None
        
        # 옵티마이저 및 스케줄러 초기화
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 로깅 초기화
        if config.training.report_to == "wandb":
            wandb.init(
                project="simple-ultravox",
                config=config.to_dict(),
                name=f"{config.data.task}_{config.model.text_model_name.split('/')[-1]}"
            )
        
        logger.info("SimpleUltravoxTrainer 초기화 완료")
    
    def _create_optimizer(self):
        """옵티마이저 생성"""
        # 학습 가능한 파라미터만 선택
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        if self.config.training.optimizer == "adamw_torch":
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            raise ValueError(f"지원하지 않는 옵티마이저: {self.config.training.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self):
        """학습률 스케줄러 생성"""
        if self.config.training.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.training.warmup_steps,
                num_training_steps=self.config.training.max_steps
            )
        elif self.config.training.lr_scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.training.warmup_steps,
                num_training_steps=self.config.training.max_steps
            )
        else:
            raise ValueError(f"지원하지 않는 스케줄러: {self.config.training.lr_scheduler}")
        
        return scheduler
    
    def train_step(self, batch: Dict) -> Dict:
        """단일 훈련 스텝"""
        self.model.train()
        
        # 배치를 디바이스로 이동
        audio = batch['audio'].to(self.device)
        text_input = batch['user_template']
        system_prompt = batch['system_prompt'][0] if batch['system_prompt'] else ""
        
        # 순전파
        outputs = self.model(
            audio_input=audio,
            text_input=text_input,
            system_prompt=system_prompt
        )
        
        loss = outputs.loss
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def eval_step(self, batch: Dict) -> Dict:
        """단일 평가 스텝"""
        self.model.eval()
        
        with torch.no_grad():
            # 배치를 디바이스로 이동
            audio = batch['audio'].to(self.device)
            text_input = batch['user_template']
            system_prompt = batch['system_prompt'][0] if batch['system_prompt'] else ""
            
            # 순전파
            outputs = self.model(
                audio_input=audio,
                text_input=text_input,
                system_prompt=system_prompt
            )
            
            loss = outputs.loss
            
            # 생성 테스트 (첫 번째 샘플만)
            if len(audio) > 0:
                generated_texts = self.model.generate(
                    audio_input=audio[:1],
                    text_input=text_input[:1],
                    system_prompt=system_prompt,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True
                )
                generated_text = generated_texts[0] if generated_texts else ""
            else:
                generated_text = ""
        
        return {
            'eval_loss': loss.item(),
            'generated_text': generated_text
        }
    
    def train(self):
        """훈련 실행"""
        logger.info("훈련 시작")
        
        # 데이터로더 생성
        train_dataloader = create_dataloader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers
        )
        
        if self.eval_dataset:
            eval_dataloader = create_dataloader(
                self.eval_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers
            )
        else:
            eval_dataloader = None
        
        # 훈련 루프
        global_step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(self.config.training.max_steps // len(train_dataloader) + 1):
            epoch_losses = []
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
            
            for batch in progress_bar:
                # 훈련 스텝
                train_metrics = self.train_step(batch)
                epoch_losses.append(train_metrics['loss'])
                
                # 진행률 업데이트
                progress_bar.set_postfix({
                    'loss': f"{train_metrics['loss']:.4f}",
                    'lr': f"{train_metrics['learning_rate']:.2e}"
                })
                
                global_step += 1
                
                # 로깅
                if global_step % self.config.training.logging_steps == 0:
                    avg_loss = np.mean(epoch_losses[-self.config.training.logging_steps:])
                    log_dict = {
                        'step': global_step,
                        'train_loss': avg_loss,
                        'learning_rate': train_metrics['learning_rate']
                    }
                    
                    if self.config.training.report_to == "wandb":
                        wandb.log(log_dict)
                    
                    logger.info(f"Step {global_step}: train_loss={avg_loss:.4f}")
                
                # 평가
                if eval_dataloader and global_step % self.config.training.eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    
                    if self.config.training.report_to == "wandb":
                        wandb.log({
                            'step': global_step,
                            'eval_loss': eval_metrics['eval_loss'],
                            'generated_text': eval_metrics.get('generated_text', '')
                        })
                    
                    logger.info(f"Step {global_step}: eval_loss={eval_metrics['eval_loss']:.4f}")
                    
                    # 최고 모델 저장
                    if eval_metrics['eval_loss'] < best_eval_loss:
                        best_eval_loss = eval_metrics['eval_loss']
                        self.save_model("best")
                
                # 체크포인트 저장
                if global_step % self.config.training.save_steps == 0:
                    self.save_model(f"checkpoint-{global_step}")
                
                # 최대 스텝 도달 시 종료
                if global_step >= self.config.training.max_steps:
                    break
            
            if global_step >= self.config.training.max_steps:
                break
        
        # 최종 모델 저장
        self.save_model("final")
        
        if self.config.training.report_to == "wandb":
            wandb.finish()
        
        logger.info("훈련 완료")
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict:
        """평가 실행"""
        eval_losses = []
        generated_texts = []
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            eval_metrics = self.eval_step(batch)
            eval_losses.append(eval_metrics['eval_loss'])
            
            if 'generated_text' in eval_metrics:
                generated_texts.append(eval_metrics['generated_text'])
        
        avg_eval_loss = np.mean(eval_losses)
        
        return {
            'eval_loss': avg_eval_loss,
            'generated_text': generated_texts[0] if generated_texts else ""
        }
    
    def save_model(self, name: str):
        """모델 저장"""
        save_path = os.path.join(self.config.training.output_dir, name)
        self.model.save_pretrained(save_path)
        logger.info(f"모델이 {save_path}에 저장되었습니다.")
    
    def load_model(self, name: str):
        """모델 로드"""
        load_path = os.path.join(self.config.training.output_dir, name)
        self.model = SimpleUltravox.from_pretrained(load_path)
        self.model.to(self.device)
        logger.info(f"모델이 {load_path}에서 로드되었습니다.")

def train_model(config_path: str):
    """모델 훈련 실행 함수"""
    # 설정 로드
    config = Config.from_yaml(config_path)
    
    # 출력 디렉토리 생성
    os.makedirs(config.training.output_dir, exist_ok=True)
    os.makedirs(config.training.logging_dir, exist_ok=True)
    
    # 훈련기 생성 및 훈련 실행
    trainer = SimpleUltravoxTrainer(config)
    trainer.train()
    
    return trainer
