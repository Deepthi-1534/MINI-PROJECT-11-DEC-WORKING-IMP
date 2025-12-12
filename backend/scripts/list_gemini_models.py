import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY required in .env")

genai.configure(api_key=API_KEY)

try:
    models = genai.list_models()
    print("Available Gemini models:")
    for m in models:
        print("-", m)
except Exception as e:
    print("Error listing models:", e)
