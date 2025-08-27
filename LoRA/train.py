import torch
from torch.utils.data import Dataset
from transformers import (
    VoxtralForConditionalGeneration, 
    AutoProcessor,
    Trainer,
    TrainingArguments
)
import json
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.tensorboard import SummaryWriter
from transformers import DataCollatorWithPadding


class VoxtralDataCollator:
    """Voxtral 모델을 위한 커스텀 데이터 콜레이터"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):
        batch = {}
        
        # input_ids와 attention_mask 처리
        if 'input_ids' in features[0]:
            input_ids = [f['input_ids'] for f in features]
            attention_mask = [f['attention_mask'] for f in features]
            
            # 패딩으로 크기 맞추기
            max_length = max(len(ids) for ids in input_ids)
            
            padded_input_ids = []
            padded_attention_mask = []
            
            for ids, mask in zip(input_ids, attention_mask):
                # 패딩 크기 계산
                padding_length = max_length - len(ids)
                
                # input_ids 패딩
                padded_ids = torch.cat([ids, torch.zeros(padding_length, dtype=ids.dtype)])
                padded_input_ids.append(padded_ids)
                
                # attention_mask 패딩
                padded_mask = torch.cat([mask, torch.zeros(padding_length, dtype=mask.dtype)])
                padded_attention_mask.append(padded_mask)
            
            batch['input_ids'] = torch.stack(padded_input_ids)
            batch['attention_mask'] = torch.stack(padded_attention_mask)
        
        # labels 처리
        if 'labels' in features[0]:
            labels = [f['labels'] for f in features]
            max_length = max(len(label) for label in labels)
            
            padded_labels = []
            for label in labels:
                padding_length = max_length - len(label)
                padded_label = torch.cat([label, torch.full((padding_length,), -100, dtype=label.dtype)])
                padded_labels.append(padded_label)
            
            batch['labels'] = torch.stack(padded_labels)
        
        # input_features 처리 (오디오가 있는 경우)
        if 'input_features' in features[0]:
            input_features = [f['input_features'] for f in features if f.get('input_features') is not None]
            if input_features:
                # 모든 input_features가 동일한 크기를 가지도록 보장
                target_size = 3000
                processed_features = []
                
                for feature in input_features:
                    if len(feature.shape) == 2:
                        if feature.shape[1] > target_size:
                            feature = feature[:, :target_size]
                        elif feature.shape[1] < target_size:
                            padding_size = target_size - feature.shape[1]
                            feature = torch.nn.functional.pad(
                                feature, 
                                (0, padding_size), 
                                mode='constant', 
                                value=0
                            )
                    processed_features.append(feature)
                
                if processed_features:
                    batch['input_features'] = torch.stack(processed_features)
        
        # 배치 크기 확인 및 디버깅
        batch_sizes = {k: v.shape[0] if hasattr(v, 'shape') else len(v) for k, v in batch.items()}
        print(f"🔍 배치 크기: {batch_sizes}")
        
        return batch


class VoxtralDomainDataset(Dataset):
    """도메인 특화 데이터셋 클래스"""
    
    def __init__(self, data_path: str, processor, model_id: str, max_length: int = 2048):
        self.processor = processor
        self.model_id = model_id  # model_id 추가
        self.max_length = max_length
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """데이터 로드 (JSON 형식 예상)"""
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        audio_path = item.get('audio_path', None)
        if audio_path is None:      
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": item['input']}
                ]
            }]
            
            inputs = self.processor.apply_chat_template(
                conversation, 
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )
        else:
            # 오디오 처리 시 고정된 크기로 패딩
            inputs = self.processor.apply_transcription_request(
                language="ko", 
                audio=audio_path, 
                model_id=self.model_id,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
        
        target_text = item['output']
        target_encoding = self.processor.tokenizer(
            target_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        
        # input_features 처리 개선
        input_features = inputs.get('input_features')
        if input_features is not None:
            input_features = input_features.squeeze()
            # 오디오 피처가 2D인 경우 처리
            if len(input_features.shape) == 2:
                # 고정된 크기로 패딩 또는 잘라내기
                target_size = 3000  # Qwen2Audio가 요구하는 mel 피처 길이
                if input_features.shape[1] > target_size:
                    input_features = input_features[:, :target_size]
                elif input_features.shape[1] < target_size:
                    # 패딩으로 크기 맞추기
                    padding_size = target_size - input_features.shape[1]
                    input_features = torch.nn.functional.pad(
                        input_features, 
                        (0, padding_size), 
                        mode='constant', 
                        value=0
                    )
        
        # 모든 텐서가 동일한 크기를 가지도록 보장
        result = {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }
        
        # input_features가 있는 경우에만 추가
        if input_features is not None:
            result['input_features'] = input_features
            
        # 디버깅을 위한 정보 추가
        if idx < 3:  # 처음 3개 샘플만 출력
            print(f"📊 샘플 {idx}: input_ids={result['input_ids'].shape}, labels={result['labels'].shape}")
            if 'input_features' in result:
                print(f"   input_features={result['input_features'].shape}")
            
        return result


class VoxtralDomainTrainer:
    """Voxtral 도메인 적응 트레이너"""
    
    def __init__(self, model_id: str = "mistralai/Voxtral-Mini-3B-2507"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = None
        self.writer = None  # 텐서보드 SummaryWriter
    
    def setup_model(self, strategy: str = "full"):
        """파인튜닝 전략에 따른 모델 설정"""
        print(f"모델 설정 중 - 전략: {strategy}")
        
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        
        if strategy == "frozen_audio":
            # 오디오 인코더 동결
            for param in self.model.audio_tower.parameters():
                param.requires_grad = False
            print("✅ 오디오 인코더 동결 완료")
            
        elif strategy == "frozen_lm":
            # 언어 모델 동결
            for param in self.model.language_model.parameters():
                param.requires_grad = False
            print("✅ 언어 모델 동결 완료")
            
        elif strategy == "projector_only":
            # 프로젝터만 학습
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.multi_modal_projector.parameters():
                param.requires_grad = True
            print("✅ 프로젝터만 학습 설정 완료")
            
        elif strategy == "lora":
            # LoRA 어댑터 추가 (언어 모델에만)
            self._setup_lora()
            print("✅ LoRA 어댑터 설정 완료")
            
        elif strategy == "full":
            # 전체 모델 파인튜닝
            print("✅ 전체 모델 파인튜닝 설정 완료")
        
        return self.model
    
    def _setup_lora(self):
        """LoRA 어댑터 설정"""
        from peft import LoraConfig, get_peft_model, TaskType
        
        # LoRA 설정
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        # 언어 모델에만 LoRA 적용
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)
    
    def train(self, 
              train_dataset: Dataset,
              val_dataset: Optional[Dataset] = None,
              output_dir: str = "./voxtral_finetuned",
              num_epochs: int = 3,
              batch_size: int = 1,
              learning_rate: float = 5e-5,
              gradient_accumulation_steps: int = 4):
        """모델 훈련 및 텐서보드 로깅"""
        
        # 텐서보드 SummaryWriter 생성
        self.writer = SummaryWriter(log_dir=output_dir + "/runs")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=10,
            save_steps=500,
            eval_steps=500 if val_dataset else None,
            eval_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            warmup_steps=100,
            fp16=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=["tensorboard"],
            gradient_checkpointing=True, 
            max_grad_norm=1.0, 
            logging_first_step=True,
            save_total_limit=2,
        )
        
        # 커스텀 데이터 콜레이터 사용
        data_collator = VoxtralDataCollator(self.processor)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.processor.tokenizer,
            data_collator=data_collator,
        )
        
        print("🚀 훈련 시작...")
        trainer.train()
        
        # 모델 저장
        trainer.save_model()
        self.processor.save_pretrained(output_dir)
        print(f"✅ 모델 저장 완료: {output_dir}")
        print(f"📈 텐서보드 로그 디렉토리: {output_dir}/runs")
        print("텐서보드 실행: tensorboard --logdir", output_dir + "/runs")
        
        self.writer.close()
        return trainer

strategy = "lora"  # 파인튜닝 전략: full, frozen_audio, frozen_lm, projector_only, lora 중 선택
data_path = "train_data.json"  # 훈련 데이터 경로
val_path = 'validation_data.json'
output_dir = "./voxtral_domain_adapted"  # 출력 디렉토리
model_id = "mistralai/Voxtral-Mini-3B-2507"  # 베이스 모델

# 트레이너 초기화
trainer = VoxtralDomainTrainer(model_id)

# 모델 설정
model = trainer.setup_model(strategy)

# 데이터셋 생성
train_dataset = VoxtralDomainDataset(data_path, trainer.processor, model_id)
val_dataset = VoxtralDomainDataset(val_path, trainer.processor, model_id)  # val_path는 별도 변수로 정의 필요

print(f"📊 훈련 데이터: {len(train_dataset)}개")
print(f"📊 평가 데이터: {len(val_dataset)}개")
print(f"🎯 전략: {strategy}")
print(f"📁 출력: {output_dir}")

# 훈련 실행 (데이터셋 크기 제한으로 테스트)
print("🧪 테스트 모드: 데이터셋 크기 제한")
train_dataset_small = torch.utils.data.Subset(train_dataset, range(min(1000, len(train_dataset))))
val_dataset_small = torch.utils.data.Subset(val_dataset, range(min(100, len(val_dataset))))

trainer.train(
    train_dataset=train_dataset_small,
    val_dataset=val_dataset_small,
    output_dir=output_dir,
    num_epochs=1,  # 테스트를 위해 1 에포크만
)
