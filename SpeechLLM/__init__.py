"""
Simple Ultravox - 단순화된 음성-언어 멀티모달 모델
"""

__version__ = "1.0.0"
__author__ = "Simple Ultravox Team"
__description__ = "Ultravox의 복잡한 구조를 단순화하여 개인 데이터셋 사용을 쉽게 만든 음성-언어 멀티모달 모델"

from .model.ultravox import SimpleUltravox
from .model.config import Config, ModelConfig, TrainingConfig, DataConfig
from .data.dataset import SimpleAudioTextDataset, create_dataloader, load_dataset
from .data.templates import get_template, format_prompt, TEMPLATES

__all__ = [
    "SimpleUltravox",
    "Config",
    "ModelConfig", 
    "TrainingConfig",
    "DataConfig",
    "SimpleAudioTextDataset",
    "create_dataloader",
    "load_dataset",
    "get_template",
    "format_prompt",
    "TEMPLATES"
]
