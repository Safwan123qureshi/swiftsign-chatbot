import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# Knowledge base ko safely import karne ke liye
try:
    from backend.knowledge_base import KNOWLEDGE_BASE
except ImportError:
    try:
        from knowledge_base import KNOWLEDGE_BASE
    except ImportError:
        KNOWLEDGE_BASE = "Swift Sign Group official services and corporate details."

app = FastAPI()

# Frontend connectivity ke liye CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    brand: str
    message: str

chat_histories = {}

@app.get("/api/status")
def status():
    return {"status": "Active", "message": "Swift Sign Group AI Server is running."}

@app.post("/chat")
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"response": "⚠️ GEMINI_API_KEY environment variable missing."}

        genai.configure(api_key=api_key)

        session_id = request.session_id
        brand = request.brand
        user_message = request.message

        if session_id not in chat_histories:
            system_instruction = f"""
You are the official AI representative for Swift Sign Group of Companies, specifically representing {brand}.
Use the following knowledge base:
{KNOWLEDGE_BASE}
Provide accurate, professional, and helpful responses based strictly on the company info.
"""
            # Updated to standard supported model name
            try:
                model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    system_instruction=system_instruction
                )
                chat_histories[session_id] = model.start_chat(history=[])
            except Exception:
                # Fallback model agar primary name reject ho
                model = genai.GenerativeModel(
                    model_name="gemini-pro",
                    system_instruction=system_instruction
                )
                chat_histories[session_id] = model.start_chat(history=[])

        chat = chat_histories[session_id]
        response = chat.send_message(user_message)
        return {"response": response.text}

    except Exception as e:
        return {"response": f"⚠️ Error: {str(e)}"}

# Vercel entrypoint binding
app = app