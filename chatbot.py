import os
import time
import torch
from scipy.io.wavfile import write
import google.generativeai as genai
from transformers import AutoTokenizer, VitsModel

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

DEFAULT_PROMPT = """
Your name is ECOBRIDGE, you are a Hausa conversational assistant. Respond in Hausa concisely (1-2 sentences), unless otherwise asked. Numbers, years and figures should be in words.
"""

chat_memory = [{"role": "system", "content": DEFAULT_PROMPT}]

# Load Hausa TTS model
model = VitsModel.from_pretrained("facebook/mms-tts-hau")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hau")

# Ensure audio output directory exists
AUDIO_DIR = os.path.join("static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

def generate_tts(text):
    """Convert text response to Hausa speech and return audio filename."""
    try:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform.numpy()
        timestamp = int(time.time() * 1000)
        filename = f"response_{timestamp}.wav"
        filepath = os.path.join(AUDIO_DIR, filename)
        write(filepath, model.config.sampling_rate, output[0])
        return filename
    except Exception as e:
        print(f"Error generating TTS: {e}")
        return None

def get_ai_response(user_input):
    """Return Gemini response text and TTS audio filename."""
    try:
        chat_memory.append({"role": "user", "content": user_input})
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content(
            f"{DEFAULT_PROMPT}\n {user_input}\n {chat_memory}"
        )
        text_response = response.text.strip()
        chat_memory.append({"role": "system", "content": text_response})
        audio_file = generate_tts(text_response)
        return text_response, os.path.join(AUDIO_DIR, audio_file) if audio_file else None
    except Exception as e:
        return f"Error: {str(e)}", None