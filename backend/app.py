import os
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

from backend.knowledge_base import COMPANY_DATA

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = None

if api_key:
    client = genai.Client(api_key=api_key)
else:
    print("❌ GEMINI_API_KEY Missing!")

app = FastAPI(title="Swift Sign Group AI Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chat Session History (In-memory store)
# Key: session_id -> Value: List of turn dicts
chat_histories: Dict[str, List[Dict[str, str]]] = {}

class MessageItem(BaseModel):
    role: str   # 'user' ya 'model'
    content: str

class ChatRequest(BaseModel):
    session_id: str = "default"
    brand: str
    message: str

@app.get("/")
def home():
    return {"status": "Active", "message": "Swift Sign Group AI Server is active."}

@app.post("/chat")
def chat(request: ChatRequest):
    session_id = request.session_id
    brand = request.brand.strip().lower()
    user_msg = request.message.strip()

    # Brand matching
    matched_key = None
    for key in COMPANY_DATA.keys():
        if key.lower() == brand:
            matched_key = key
            break

    if not matched_key:
        matched_key = "corporate_info"

    context_data = COMPANY_DATA[matched_key]

    # Session history track
    if session_id not in chat_histories:
        chat_histories[session_id] = []

    # History summary context
    history_str = ""
    for msg in chat_histories[session_id][-6:]:  # Last 6 turns for context
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    prompt = f"""
You are the official AI Assistant for Swift Sign Group ({matched_key.replace('_', ' ').upper()} division).

Subsidiary Knowledge Base Context:
{context_data}

Previous Conversation History:
{history_str}

User Question: {user_msg}

Guidelines:
- Maintain context of the previous messages if relevant.
- Format lists cleanly with line breaks and bullet points (*).
- Answer accurately based on the context.
"""

    try:
        if not client:
            raise Exception("Gemini Client not initialized.")

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        response_text = response.text.strip()

        # Update History
        chat_histories[session_id].append({"role": "user", "content": user_msg})
        chat_histories[session_id].append({"role": "model", "content": response_text})

    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        response_text = f"Thank you for contacting Swift Sign Group. For details regarding {matched_key.upper()}, email info@swiftsignbm.com."

    return {
        "brand": brand,
        "session_id": session_id,
        "response": response_text,
        "history_count": len(chat_histories[session_id])
    }

@app.delete("/clear_history/{session_id}")
def clear_history(session_id: str):
    if session_id in chat_histories:
        chat_histories[session_id] = []
        return {"status": "success", "message": f"History cleared for session {session_id}"}
    return {"status": "not_found", "message": "Session ID not found"}
@app.get("/history/{session_id}")
def get_history(session_id: str):
    return {
        "session_id": session_id,
        "history": chat_histories.get(session_id, [])
    }