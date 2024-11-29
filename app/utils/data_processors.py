import torch
import whisper
import numpy as np
from PIL import Image

class AudioToTextProcessor:
    def __init__(self, model_size="tiny"):
        self.model = whisper.load_model(model_size)
        
    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result["text"]

class ImageProcessor:
    def __init__(self, clip_preprocess):
        self.preprocess = clip_preprocess
        
    def process_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        processed_image = self.preprocess(image)
        return processed_image

def prepare_text_input(text, tokenizer, max_length=256):
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return inputs
