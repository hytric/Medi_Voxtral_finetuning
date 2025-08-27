"""
단순화된 Ultravox 모델
오디오 인코더 + 프로젝터 + LLM 구조
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoProcessor,
    LlamaForCausalLM, LlamaTokenizer,
    WhisperModel, WhisperProcessor
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

from .config import ModelConfig

logger = logging.getLogger(__name__)

class AudioProjector(nn.Module):
    """오디오를 텍스트 임베딩 공간으로 투영하는 프로젝터"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # Whisper 모델의 출력 차원 (기본 1280)
        self.audio_dim = 1280
        
        # 프로젝터 레이어들
        self.input_proj = nn.Linear(self.audio_dim, config.projector_hidden_size)
        self.hidden_proj = nn.Linear(config.projector_hidden_size, config.projector_hidden_size)
        self.output_proj = nn.Linear(config.projector_hidden_size, config.projector_output_size)
        
        # 레이어 정규화
        self.layer_norm1 = nn.LayerNorm(config.projector_hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.projector_hidden_size)
        
        # 드롭아웃
        self.dropout = nn.Dropout(config.projector_dropout)
        
        # 활성화 함수
        self.activation = nn.SiLU()
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for module in [self.input_proj, self.hidden_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, audio_features):
        """
        Args:
            audio_features: (batch_size, seq_len, audio_dim)
        
        Returns:
            projected_features: (batch_size, seq_len, projector_output_size)
        """
        # 입력 프로젝션
        x = self.input_proj(audio_features)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 히든 프로젝션
        x = self.hidden_proj(x)
        x = self.layer_norm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 출력 프로젝션
        x = self.output_proj(x)
        
        return x

class SimpleUltravox(nn.Module):
    """단순화된 Ultravox 모델"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.audio_token_id = config.audio_token_id
        
        # 오디오 인코더 (Whisper)
        logger.info(f"오디오 인코더 로딩: {config.audio_model_name}")
        self.audio_encoder = WhisperModel.from_pretrained(config.audio_model_name)
        self.audio_processor = WhisperProcessor.from_pretrained(config.audio_model_name)
        
        # 오디오 인코더 LoRA 적용 (선택사항)
        if config.audio_use_lora:
            audio_lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config.audio_lora_r,
                lora_alpha=config.audio_lora_alpha,
                target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
                bias="none",
            )
            self.audio_encoder = get_peft_model(self.audio_encoder, audio_lora_config)
        
        # 텍스트 모델 (LLaMA)
        logger.info(f"텍스트 모델 로딩: {config.text_model_name}")
        self.text_model = LlamaForCausalLM.from_pretrained(
            config.text_model_name,
            torch_dtype=getattr(torch, config.torch_dtype),
            device_map=config.device_map
        )
        self.text_tokenizer = LlamaTokenizer.from_pretrained(config.text_model_name)
        
        # 텍스트 모델 LoRA 적용
        if config.use_lora:
            text_lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                bias="none",
                lora_dropout=config.lora_dropout,
            )
            self.text_model = get_peft_model(self.text_model, text_lora_config)
        
        # 프로젝터
        self.projector = AudioProjector(config)
        
        # 오디오 토큰 임베딩
        self.audio_embedding = nn.Embedding(1, self.text_model.config.hidden_size)
        
        # 가중치 초기화
        self._init_weights()
        
        logger.info("SimpleUltravox 모델 초기화 완료")
    
    def _init_weights(self):
        """가중치 초기화"""
        # 오디오 임베딩 초기화
        nn.init.normal_(self.audio_embedding.weight, std=0.02)
    
    def encode_audio(self, audio_input):
        """
        오디오를 인코딩합니다.
        
        Args:
            audio_input: (batch_size, audio_length)
        
        Returns:
            audio_features: (batch_size, seq_len, audio_dim)
        """
        # Whisper 프로세서로 전처리
        inputs = self.audio_processor(
            audio_input, 
            sampling_rate=self.config.audio_sample_rate, 
            return_tensors="pt"
        )
        
        # 오디오 인코딩
        with torch.no_grad():
            audio_features = self.audio_encoder(**inputs, output_hidden_states=True)
            # 마지막 레이어의 히든 스테이트 사용
            audio_features = audio_features.hidden_states[-1]
        
        return audio_features
    
    def project_audio(self, audio_features):
        """
        오디오 특징을 텍스트 임베딩 공간으로 투영합니다.
        
        Args:
            audio_features: (batch_size, seq_len, audio_dim)
        
        Returns:
            projected_features: (batch_size, seq_len, text_hidden_size)
        """
        return self.projector(audio_features)
    
    def prepare_inputs(self, audio_input, text_input, system_prompt=""):
        """
        모델 입력을 준비합니다.
        
        Args:
            audio_input: (batch_size, audio_length)
            text_input: (batch_size,) 텍스트 입력 리스트
            system_prompt: 시스템 프롬프트
        
        Returns:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len)
        """
        batch_size = audio_input.shape[0]
        
        # 오디오 인코딩 및 투영
        audio_features = self.encode_audio(audio_input)
        projected_audio = self.project_audio(audio_features)
        
        # 텍스트 토큰화
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for i in range(batch_size):
            # 시스템 프롬프트 + 사용자 입력 + 오디오 토큰 + 어시스턴트 응답
            if system_prompt:
                full_text = f"<s>[INST] {system_prompt}\n\n{text_input[i]} <|audio|> [/INST]"
            else:
                full_text = f"<s>[INST] {text_input[i]} <|audio|> [/INST]"
            
            # 텍스트 토큰화
            text_tokens = self.text_tokenizer(
                full_text,
                return_tensors="pt",
                add_special_tokens=False
            )
            
            # 오디오 토큰 위치 찾기
            audio_token_pos = (text_tokens.input_ids[0] == self.audio_token_id).nonzero(as_tuple=True)[0]
            
            if len(audio_token_pos) == 0:
                logger.warning(f"오디오 토큰을 찾을 수 없음: {text_input[i]}")
                continue
            
            audio_token_pos = audio_token_pos[0]
            
            # 오디오 임베딩 삽입
            audio_embedding = self.audio_embedding(torch.zeros(1, dtype=torch.long))
            
            # 텍스트 임베딩 가져오기
            text_embeddings = self.text_model.model.embed_tokens(text_tokens.input_ids)
            
            # 오디오 임베딩으로 교체
            text_embeddings[0, audio_token_pos] = audio_embedding[0]
            
            # 입력 준비
            input_ids = text_tokens.input_ids
            attention_mask = text_tokens.attention_mask
            
            # 라벨 설정 (오디오 토큰 이후만 학습)
            labels = torch.full_like(input_ids, -100)
            labels[0, audio_token_pos+1:] = input_ids[0, audio_token_pos+1:]
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
        
        # 배치로 스택
        if input_ids_list:
            input_ids = torch.cat(input_ids_list, dim=0)
            attention_mask = torch.cat(attention_mask_list, dim=0)
            labels = torch.cat(labels_list, dim=0)
        else:
            # 빈 배치 처리
            input_ids = torch.zeros((batch_size, 1), dtype=torch.long)
            attention_mask = torch.zeros((batch_size, 1), dtype=torch.long)
            labels = torch.full((batch_size, 1), -100, dtype=torch.long)
        
        return input_ids, attention_mask, labels
    
    def forward(self, audio_input, text_input, system_prompt="", labels=None):
        """
        순전파
        
        Args:
            audio_input: (batch_size, audio_length)
            text_input: (batch_size,) 텍스트 입력 리스트
            system_prompt: 시스템 프롬프트
            labels: 라벨 (선택사항)
        
        Returns:
            outputs: 모델 출력
        """
        # 입력 준비
        input_ids, attention_mask, prepared_labels = self.prepare_inputs(
            audio_input, text_input, system_prompt
        )
        
        # 라벨이 제공되지 않으면 준비된 라벨 사용
        if labels is None:
            labels = prepared_labels
        
        # 텍스트 모델 순전파
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, audio_input, text_input, system_prompt="", max_new_tokens=512, **kwargs):
        """
        텍스트 생성
        
        Args:
            audio_input: (batch_size, audio_length)
            text_input: (batch_size,) 텍스트 입력 리스트
            system_prompt: 시스템 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            **kwargs: 생성 파라미터
        
        Returns:
            generated_texts: 생성된 텍스트 리스트
        """
        # 입력 준비
        input_ids, attention_mask, _ = self.prepare_inputs(
            audio_input, text_input, system_prompt
        )
        
        # 생성
        with torch.no_grad():
            outputs = self.text_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.text_tokenizer.eos_token_id,
                **kwargs
            )
        
        # 생성된 텍스트 디코딩
        generated_texts = []
        for output in outputs:
            # 입력 부분 제거하고 생성된 부분만 추출
            generated_ids = output[input_ids.shape[1]:]
            generated_text = self.text_tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def save_pretrained(self, save_directory):
        """모델 저장"""
        import os
        
        os.makedirs(save_directory, exist_ok=True)
        
        # 설정 저장
        self.config.save_yaml(os.path.join(save_directory, "config.yaml"))
        
        # 모델 가중치 저장
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # 토크나이저 저장
        self.text_tokenizer.save_pretrained(save_directory)
        self.audio_processor.save_pretrained(save_directory)
        
        logger.info(f"모델이 {save_directory}에 저장되었습니다.")
    
    @classmethod
    def from_pretrained(cls, model_path):
        """모델 로드"""
        import os
        
        # 설정 로드
        config = ModelConfig.from_yaml(os.path.join(model_path, "config.yaml"))
        
        # 모델 생성
        model = cls(config)
        
        # 가중치 로드
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        model.load_state_dict(state_dict)
        
        # 토크나이저 로드
        model.text_tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model.audio_processor = WhisperProcessor.from_pretrained(model_path)
        
        logger.info(f"모델이 {model_path}에서 로드되었습니다.")
        
        return model
