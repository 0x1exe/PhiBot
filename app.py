import gradio as gr
from app.model import load_models, process_text, process_image, process_audio

model, tokenizer, whisper_model = load_models()

def multi_modal_chat(text, image, audio):
    if audio is not None:
        return process_audio(audio, whisper_model, model, tokenizer)
    elif image is not None:
        return process_image(image, model, tokenizer)
    else:
        return process_text(text, model, tokenizer)

iface = gr.Interface(
    fn=multi_modal_chat,
    inputs=[
        gr.Textbox(placeholder="Enter text here..."),
        gr.Image(type="pil"),
        gr.Audio(type="filepath")
    ],
    outputs="text",
    title="PhiChat",
    description="Chat with a model using text, images, or audio input"
)

iface.launch()