import gradio as gr
import torch
from app.models.multimodal_model import MultiModalLLM
from app.utils.data_processors import AudioToTextProcessor, prepare_text_input
import tempfile
import os

class MultiModalInterface:
    def __init__(self, model_path=None):
        self.model = MultiModalLLM()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.audio_processor = AudioToTextProcessor()
        
    def process_input(self, text_input, image_input=None, audio_input=None):
        combined_input = ""
        
        if text_input:
            combined_input += text_input
        
        if audio_input is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(audio_input)
                temp_audio_path = temp_audio.name
                
            audio_text = self.audio_processor.transcribe(temp_audio_path)
            os.unlink(temp_audio_path) 
            
            if combined_input:
                combined_input += "\nTranscribed Audio: " + audio_text
            else:
                combined_input = audio_text
        
        if not combined_input:
            return "Please provide either text or audio input."
            
        image_features = None
        if image_input is not None:
            image_features = self.model.process_image(image_input)
            
        inputs = prepare_text_input(combined_input, self.model.tokenizer)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_features=image_features
            )
            
        response = self.model.tokenizer.decode(
            outputs.logits.argmax(dim=-1)[0],
            skip_special_tokens=True
        )
        
        return response

def create_interface():
    interface = MultiModalInterface()
    
    demo = gr.Interface(
        fn=interface.process_input,
        inputs=[
            gr.Textbox(label="Text Input", placeholder="Enter your text here..."),
            gr.Image(label="Image Input", type="pil"),
            gr.Audio(label="Audio Input", type="binary")
        ],
        outputs=gr.Textbox(label="Response"),
        title="PhiChat",
        description="Chat with a model using text, images, or audio input.",
    )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
