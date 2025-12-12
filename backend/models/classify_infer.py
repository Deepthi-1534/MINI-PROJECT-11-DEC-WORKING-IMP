import os
import base64
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv
import cv2

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in .env file")

genai.configure(api_key=API_KEY)

# WORKING MODEL FROM YOUR LIST
GEMINI_MODEL = "models/gemini-2.5-flash"   # or "models/gemini-2.5-pro"

def classify_infer(crop_bgr):

    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0

    # convert image to JPEG bytes
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    ok, jpeg = cv2.imencode(".jpg", rgb)
    if not ok:
        return None, 0.0
    img_bytes = jpeg.tobytes()

    # Prepare prompt
    prompt = """
You are an expert wildlife biologist.

Identify the animal in the image with the most accurate and specific common name possible.

Return ONLY a strict JSON object in this format:

{
  "species": "<most accurate common name (e.g., 'Eastern gray squirrel', 'Bengal tiger', 'African bush elephant') or 'unknown'>",
  "confidence": <0.0 - 1.0>
}

Rules:
- Be very specific (example: 'Bengal tiger' NOT just 'tiger').
- Include subspecies or region when confidently identifiable.
- If the exact subspecies cannot be determined, give the most accurate common name (e.g., 'gray squirrel').
- Respond ONLY with JSON. No extra text.
"""


    try:
        response = genai.GenerativeModel(GEMINI_MODEL).generate_content(
            [
                prompt,
                {"mime_type": "image/jpeg", "data": img_bytes}
            ],
            generation_config={"temperature": 0}
        )

        text = response.text.strip()

        # find JSON in response
        start = text.find("{")
        end = text.rfind("}")
        js = text[start:end+1]

        data = json.loads(js)

        species = data.get("species", "unknown")
        confidence = float(data.get("confidence", 0.0))

        if species.lower() == "unknown":
            return None, 0.0

        return species, confidence

    except Exception as e:
        print("Gemini classify error:", e)
        return None, 0.0
