import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import open_clip
import bitsandbytes as bnb

class ImageProjectionLayer(nn.Module):
    def __init__(self, clip_embedding_dim, target_dim):
        super().__init__()
        self.projection = bnb.nn.Int8Linear(clip_embedding_dim, target_dim)
        
    def forward(self, x):
        return self.projection(x)

class MultiModalLLM(nn.Module):
    def __init__(self, base_model_name="microsoft/phi-2", lora_r=8, lora_alpha=32, max_length=256, device_map="auto"):
        super().__init__()
        
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            model_max_length=max_length,
            padding_side="right",
            use_fast=True,
        )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=True,
            device_map=device_map,
            max_position_embeddings=max_length,
            torch_dtype=torch.float16,
        )
        
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["query_key_value"],
            bias="none",
            modules_to_save=[],
        )
        
        self.base_model = get_peft_model(self.base_model, peft_config)
        
        self.clip_model, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='laion2b_s34b_b79k',
            device=device_map if device_map != "auto" else "cuda",
            force_quick_gelu=True,  
        )
        
        self.image_projection = ImageProjectionLayer(
            clip_embedding_dim=512,
            target_dim=2560
        )
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
    def process_image(self, image):
        with torch.cuda.amp.autocast(), torch.no_grad():
            image_features = self.clip_model.encode_image(image)
        projected_features = self.image_projection(image_features)
        return projected_features
    
    def forward(self, input_ids, attention_mask=None, image_features=None):
        if input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        if image_features is not None:
            projected_image = self.image_projection(image_features)
            outputs.logits += projected_image.unsqueeze(1)
            
        return outputs
