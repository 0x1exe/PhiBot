import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import whisper
from app.config import DEVICE, MULTIMODAL_MODEL_PATH

class MultimodalPhi(nn.Module):
    def __init__(self, phi_model):
        super().__init__()
        self.phi_model = phi_model
        self.embedding_projection = nn.Linear(512, phi_model.config.hidden_size)
    
    def forward(self, image_embeddings, input_ids, attention_mask):
        if image_embeddings is not None:
            projected_embeddings = self.embedding_projection(image_embeddings).unsqueeze(1)
            inputs_embeds = self.phi_model.get_input_embeddings()(input_ids)
            combined_embeds = torch.cat([projected_embeddings, inputs_embeds], dim=1)
            extended_attention_mask = torch.cat([torch.ones(attention_mask.shape[0], 1).to(attention_mask.device), attention_mask], dim=1)
            outputs = self.phi_model(inputs_embeds=combined_embeds, attention_mask=extended_attention_mask)
        else:
            outputs = self.phi_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

def load_models():
    phi_model = AutoModelForCausalLM.from_pretrained("SkunkworksAI/PlatyPhi-1.5B")
    model = MultimodalPhi(phi_model).to(DEVICE)
    model.load_state_dict(torch.load(MULTIMODAL_MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("SkunkworksAI/PlatyPhi-1.5B")
    tokenizer.pad_token = tokenizer.eos_token
    
    whisper_model = whisper.load_model("base")
    
    return model, tokenizer, whisper_model

def process_text(text, model, tokenizer):
    prompt = f"Human: {text}\n\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    outputs = model(None, inputs.input_ids, inputs.attention_mask)
    return tokenizer.decode(outputs[0].argmax(dim=-1), skip_special_tokens=True)

def process_image(image, model, tokenizer):
    clip_embedding = get_clip_embedding(image)
    prompt = "Describe this image:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    outputs = model(clip_embedding.unsqueeze(0).to(DEVICE), inputs.input_ids, inputs.attention_mask)
    return tokenizer.decode(outputs[0].argmax(dim=-1), skip_special_tokens=True)

def process_audio(audio, whisper_model, model, tokenizer):
    result = whisper_model.transcribe(audio)
    transcription = result["text"]
    return process_text(f"Transcription: {transcription}\nPlease respond to this:", model, tokenizer)