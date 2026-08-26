import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from backend.knowledge_base import COMPANY_DATA

load_dotenv()

# New Google GenAI SDK Client Initialization
api_key = os.getenv("GEMINI_API_KEY")
client = None

if api_key:
    client = genai.Client(api_key=api_key)
else:
    print("❌ ERROR: GEMINI_API_KEY is missing in .env file!")

app = FastAPI(title="Swift Sign Group AI Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    brand: str
    message: str

@app.get("/")
def home():
    return {"status": "Active", "message": "Swift Sign Group AI Chatbot Server is active."}

@app.post("/chat")
def chat(request: ChatRequest):
    brand = request.brand.strip().lower()
    user_msg = request.message.strip()

    # Case-insensitive key match
    matched_key = None
    for key in COMPANY_DATA.keys():
        if key.lower() == brand:
            matched_key = key
            break

    if not matched_key:
        return {"response": f"Please select a valid subsidiary branch. Received: '{brand}'"}

    context_data = COMPANY_DATA[matched_key]

    prompt = f"""
You are the official AI Assistant for Swift Sign Group ({matched_key.replace('_', ' ').upper()} division).

Context:
{context_data}

User Question: {user_msg}

Answer concisely based on the context above. If asked general greetings like 'hi' or 'hello', welcome them politely and state how you can assist.
"""

    try:
        if not client:
            raise Exception("Gemini Client not initialized. Check API Key.")

        # Modern SDK call with gemini-2.5-flash
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        response_text = response.text.strip()

    except Exception as e:
        print(f"❌ Gemini API Error Details: {e}")
        response_text = f"Thank you for contacting Swift Sign Group. For detailed inquiries regarding {matched_key.upper()}, please reach out to our team at info@swiftsignbm.com or call +92 334 8399480."

    return {
        "brand": brand,
        "response": response_text
    }