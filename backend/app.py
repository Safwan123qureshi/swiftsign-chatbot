import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# Relative import handling
try:
    from .knowledge_base import KNOWLEDGE_BASE
except ImportError:
    from knowledge_base import KNOWLEDGE_BASE

app = FastAPI()

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
    return {"status": "Active", "message": "Swift Sign Group AI Server is active."}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"response": "Error: GEMINI_API_KEY is not set in environment variables."}

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
            # Updated to standard model string
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system_instruction
            )
            chat_histories[session_id] = model.start_chat(history=[])

        chat = chat_histories[session_id]
        response = chat.send_message(user_message)
        return {"response": response.text}

    except Exception as e:
        # Front-end crash ke bajaye exact error return karega UI par
        return {"response": f"Backend Error: {str(e)}"}