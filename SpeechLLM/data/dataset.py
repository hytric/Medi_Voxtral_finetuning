"""
단순화된 데이터셋 로딩 및 처리
"""

import json
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union
import logging

from .templates import get_template, format_prompt, AUDIO_PLACEHOLDER

logger = logging.getLogger(__name__)

class SimpleAudioTextDataset(Dataset):
    """
    단순한 오디오-텍스트 데이터셋 클래스
    JSON Lines 형식의 데이터를 로드하여 템플릿에 맞게 처리
    """
    
    def __init__(
        self,
        data_path: str,
        task: str = None,  # None이면 멀티태스크 학습
        audio_model_name: str = "openai/whisper-medium",
        text_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        max_audio_length: float = 30.0,
        sample_rate: int = 16000,
        **kwargs
    ):
        """
        Args:
            data_path (str): JSON Lines 파일 경로
            task (str): 태스크 유형 (transcription, qa, translation, continuation, medical_qa)
            audio_model_name (str): 오디오 인코더 모델명
            text_model_name (str): 텍스트 모델명
            max_audio_length (float): 최대 오디오 길이 (초)
            sample_rate (int): 샘플링 레이트
        """
        self.data_path = data_path
        self.task = task
        self.audio_model_name = audio_model_name
        self.text_model_name = text_model_name
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        
        # 템플릿 가져오기 (멀티태스크 학습 지원)
        if task is None:
            self.templates = get_template(task_name=None, **kwargs)
            self.task = None
        else:
            self.template = get_template(task, **kwargs)
            self.task = task
        
        # 데이터 로드
        self.data = self._load_data()
        
        if self.task is None:
            # 멀티태스크 학습인 경우 태스크별 통계 출력
            task_counts = {}
            audio_counts = {'with_audio': 0, 'text_only': 0}
            for item in self.data:
                task = item.get('task', 'transcription')
                task_counts[task] = task_counts.get(task, 0) + 1
                
                if item.get('audio_filepath') is not None:
                    audio_counts['with_audio'] += 1
                else:
                    audio_counts['text_only'] += 1
            
            logger.info(f"데이터셋 로드 완료: {len(self.data)}개 샘플 (멀티태스크)")
            logger.info(f"  - 오디오 포함: {audio_counts['with_audio']}개")
            logger.info(f"  - 텍스트만: {audio_counts['text_only']}개")
            for task, count in task_counts.items():
                logger.info(f"  - {task}: {count}개")
        else:
            logger.info(f"데이터셋 로드 완료: {len(self.data)}개 샘플, 태스크: {self.task}")
    
    def _load_data(self) -> List[Dict]:
        """JSON Lines 파일에서 데이터 로드"""
        data = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    
                    # 필수 필드 확인
                    required_fields = ['output']
                    for field in required_fields:
                        if field not in item:
                            logger.warning(f"라인 {line_num}: 필수 필드 '{field}' 누락")
                            continue
                    
                    # 오디오 파일이 있는 경우에만 확인
                    if 'audio_filepath' in item:
                        if not os.path.exists(item['audio_filepath']):
                            logger.warning(f"라인 {line_num}: 오디오 파일 없음 - {item['audio_filepath']}")
                            continue
                    else:
                        # 텍스트만 있는 경우
                        item['audio_filepath'] = None
                    
                    # 오디오 길이 확인 (오디오가 있는 경우에만)
                    if item['audio_filepath'] is not None:
                        try:
                            info = torchaudio.info(item['audio_filepath'])
                            duration = info.num_frames / info.sample_rate
                            if duration > self.max_audio_length:
                                logger.warning(f"라인 {line_num}: 오디오가 너무 김 ({duration:.1f}s > {self.max_audio_length}s)")
                                continue
                        except Exception as e:
                            logger.warning(f"라인 {line_num}: 오디오 정보 읽기 실패 - {e}")
                            continue
                    
                    data.append(item)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"라인 {line_num}: JSON 파싱 오류 - {e}")
                    continue
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """데이터 샘플 반환"""
        item = self.data[idx]
        
        # 오디오 로드 및 전처리 (오디오가 있는 경우에만)
        if item['audio_filepath'] is not None:
            audio = self._load_audio(item['audio_filepath'])
        else:
            # 텍스트만 있는 경우 빈 오디오 생성
            audio = torch.zeros(self.sample_rate)
        
        # 텍스트 필드 처리
        text = item.get('text', '')
        
        # 태스크별 템플릿 선택 (멀티태스크 학습 지원)
        if self.task is None:
            # 데이터에서 태스크 가져오기
            task = item.get('task', 'transcription')
            if task not in self.templates:
                task = 'transcription'  # 기본값
            template = self.templates[task]
        else:
            template = self.template
        
        # 템플릿 포맷
        formatted_template = format_prompt(
            template,
            audio_placeholder=AUDIO_PLACEHOLDER,
            text=text,
            output=item['output'],
            **item
        )
        
        return {
            'audio': audio,
            'text': text,
            'output': item['output'],
            'user_template': formatted_template['user_template'],
            'assistant_template': formatted_template['assistant_template'],
            'system_prompt': formatted_template.get('system_prompt', ''),
            'task': self.task or item.get('task', 'transcription'),
            'audio_filepath': item['audio_filepath'],
            'has_audio': item['audio_filepath'] is not None
        }
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """오디오 파일 로드 및 전처리"""
        try:
            # 오디오 로드
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 모노로 변환 (필요시)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 샘플링 레이트 변환 (필요시)
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # 정규화
            waveform = waveform / torch.max(torch.abs(waveform))
            
            return waveform.squeeze()
            
        except Exception as e:
            logger.error(f"오디오 로드 실패 {audio_path}: {e}")
            # 빈 오디오 반환
            return torch.zeros(self.sample_rate)

def create_dataloader(
    dataset: SimpleAudioTextDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """데이터로더 생성"""
    
    def collate_fn(batch):
        """배치 데이터 정렬"""
        # 오디오 길이 맞추기
        max_length = max(item['audio'].shape[0] for item in batch)
        
        padded_audio = []
        for item in batch:
            audio = item['audio']
            if audio.shape[0] < max_length:
                # 패딩
                padding = torch.zeros(max_length - audio.shape[0])
                audio = torch.cat([audio, padding])
            padded_audio.append(audio)
        
        return {
            'audio': torch.stack(padded_audio),
            'text': [item['text'] for item in batch],
            'output': [item['output'] for item in batch],
            'user_template': [item['user_template'] for item in batch],
            'assistant_template': [item['assistant_template'] for item in batch],
            'system_prompt': [item['system_prompt'] for item in batch],
            'task': [item['task'] for item in batch],
            'audio_filepath': [item['audio_filepath'] for item in batch]
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )

def load_dataset(
    data_path: str,
    task: str = "transcription",
    split: str = "train",
    **kwargs
) -> SimpleAudioTextDataset:
    """데이터셋 로드 헬퍼 함수"""
    return SimpleAudioTextDataset(
        data_path=data_path,
        task=task,
        **kwargs
    )
