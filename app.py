import gradio as gr
from chatbot import get_ai_response

def chat(user_input):
    text_response, audio_path = get_ai_response(user_input)
    return text_response, audio_path if audio_path else None

demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=2, placeholder="Rubuta sakonka a nan...", label="Mai amfani"),
    outputs=[
        gr.Textbox(label="Amsa daga ECOBRIDGE"),
        gr.Audio(label="Amsa ta Sauti", type="filepath")
    ],
    title="ECOBRIDGE – Hausa Gemini Bot",
    description="Tambayi ECOBRIDGE cikin Hausa! Wannan yana amfani da Gemini da Hausa TTS don samar da amsa da sauti.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
