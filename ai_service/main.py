import os, io
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Use 2.5 Flash for low latency
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

SYSTEM = ("You are a concise step-by-step tutor. "
          "Prefer numbered steps, 4–6 points. "
          "Use what you see on the board image when relevant.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True,
)

@app.post("/ask_ai")
async def ask_ai(image: UploadFile, question: str = Form(...)):
    data = await image.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    prompt = f"{SYSTEM}\nUser question: {question}\nBe brief."

    # multimodal: prompt + image
    resp = model.generate_content([prompt, img], request_options={"timeout": 30})
    return {"answer": resp.text.strip()}