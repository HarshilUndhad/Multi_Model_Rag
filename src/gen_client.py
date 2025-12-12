# Generative AI client using Google Gemini model
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing in .env")

# configure Gemini
genai.configure(api_key=API_KEY)

# choose the stable model
MODEL_NAME = "models/gemini-flash-latest"
model = genai.GenerativeModel(MODEL_NAME)

def generate_answer(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Generate answer using Gemini model.
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        # .text extracts final formatted output
        return response.text
    
    except Exception as e:
        return f"[Error calling Gemini]: {e}"
