"""
모델 설정 클래스
"""

import dataclasses
from typing import Optional, Dict, Any
from transformers import AutoConfig

@dataclasses.dataclass
class ModelConfig:
    """모델 설정"""
    
    # 모델 이름들
    text_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    audio_model_name: str = "openai/whisper-large-v3"
    
    # 어댑터 설정
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[list] = None
    
    # 오디오 인코더 LoRA 설정
    audio_use_lora: bool = False
    audio_lora_r: int = 8
    audio_lora_alpha: int = 16
    
    # 프로젝터 설정
    projector_hidden_size: int = 1024
    projector_output_size: int = 4096
    projector_dropout: float = 0.1
    
    # 오디오 처리 설정
    audio_sample_rate: int = 16000
    audio_max_length: float = 30.0
    audio_stack_factor: int = 8
    
    # 토큰화 설정
    audio_token_id: int = 32000
    max_length: int = 2048
    
    # 기타 설정
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    
    def __post_init__(self):
        """기본값 설정"""
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """딕셔너리에서 생성"""
        return cls(**config_dict)

@dataclasses.dataclass
class TrainingConfig:
    """훈련 설정"""
    
    # 기본 훈련 설정
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    
    # 옵티마이저 설정
    optimizer: str = "adamw_torch"
    lr_scheduler: str = "cosine"
    
    # 정밀도 설정
    fp16: bool = True
    bf16: bool = False
    
    # 체크포인트 설정
    output_dir: str = "./checkpoints"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # 로깅 설정
    logging_dir: str = "./logs"
    report_to: str = "tensorboard"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """딕셔너리에서 생성"""
        return cls(**config_dict)

@dataclasses.dataclass
class DataConfig:
    """데이터 설정"""
    
    # 데이터셋 설정
    train_dataset_path: str = ""
    eval_dataset_path: Optional[str] = None
    test_dataset_path: Optional[str] = None
    
    # 태스크 설정 (None이면 멀티태스크 학습)
    task: str = None
    
    # 데이터 처리 설정
    max_audio_length: float = 30.0
    sample_rate: int = 16000
    num_workers: int = 4
    
    # 템플릿 설정
    template_kwargs: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """기본값 설정"""
        if self.template_kwargs is None:
            self.template_kwargs = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DataConfig":
        """딕셔너리에서 생성"""
        return cls(**config_dict)

@dataclasses.dataclass
class Config:
    """전체 설정"""
    
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    
    def __post_init__(self):
        """설정 검증"""
        if not self.data.train_dataset_path:
            raise ValueError("train_dataset_path가 설정되지 않았습니다.")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "data": self.data.to_dict()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """딕셔너리에서 생성"""
        return cls(
            model=ModelConfig.from_dict(config_dict.get("model", {})),
            training=TrainingConfig.from_dict(config_dict.get("training", {})),
            data=DataConfig.from_dict(config_dict.get("data", {}))
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """YAML 파일에서 생성"""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save_yaml(self, yaml_path: str):
        """YAML 파일로 저장"""
        import yaml
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
