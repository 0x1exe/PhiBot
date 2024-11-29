import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.multimodal_model import MultiModalLLM
from app.utils.data_processors import prepare_text_input
import gc
from accelerate import Accelerator
from torch.cuda.amp import autocast, GradScaler

def print_gpu_utilization():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) // 1024**2}MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved(0) // 1024**2}MB")

class LLaVADataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
   
        print("Loading dataset...")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} examples")
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        conversation = item['conversations']
        text = ""
        for conv in conversation:
            if conv['from'] == 'human':
                text += "Human: " + conv['value'] + "\n"
            else:
                text += "Assistant: " + conv['value'] + "\n"

        inputs = prepare_text_input(text, self.tokenizer, self.max_length)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
        }

def download_dataset():
    dataset_path = "llava_instruct_150k.json"
    if not os.path.exists(dataset_path):
        url = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json"
        print(f"Downloading dataset from {url}")
        os.system(f"wget -c {url}")
    return dataset_path

def train(device_map="auto"):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("Initializing model...")
    model = MultiModalLLM(device_map=device_map)
    model.train()
    
    print("\nInitial GPU Memory Usage:")
    print_gpu_utilization()

    dataset_path = download_dataset()
    dataset = LLaVADataset(dataset_path, model.tokenizer)

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01
    )

    scaler = GradScaler()
  
    batch_size = 2 
    gradient_accumulation_steps = 4  
    num_epochs = 3
    

    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_map == "auto":
        print("Using automated device mapping")
    else:
        model.to(device)
    
    print("\nStarting training...")
    print(f"Total examples: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    
    for epoch in range(num_epochs):
        print(f"\nStarting epoch {epoch + 1}/{num_epochs}")
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(progress_bar):
            if step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = outputs.loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({
                "loss": loss.item() * gradient_accumulation_steps,
                "gpu_mem": f"{torch.cuda.memory_allocated() // 1024**2}MB"
            })
            
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print("\nSaving checkpoint...")
        torch.save(
            {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            },
            os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        )
        print(f"Saved checkpoint for epoch {epoch+1}")
        print_gpu_utilization()

if __name__ == "__main__":
    train()
