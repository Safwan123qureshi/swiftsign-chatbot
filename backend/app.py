import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from backend.knowledge_base import KNOWLEDGE_BASE

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

chat_histories = {}

class ChatRequest(BaseModel):
    session_id: str
    brand: str
    message: str

class ClearHistoryRequest(BaseModel):
    session_id: str

@app.get("/api/status")
def status():
    return {"status": "Active", "message": "Swift Sign Group AI Server is active."}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
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
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system_instruction
            )
            chat_histories[session_id] = model.start_chat(history=[])

        chat = chat_histories[session_id]
        response = chat.send_message(user_message)
        return {"response": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/history")
async def get_history(request: ClearHistoryRequest):
    session_id = request.session_id
    if session_id in chat_histories:
        chat = chat_histories[session_id]
        history_data = []
        for msg in chat.history:
            history_data.append({
                "role": "user" if msg.role == "user" else "model",
                "parts": [p.text for p in msg.parts]
            })
        return {"history": history_data}
    return {"history": []}

@app.post("/clear_history")
async def clear_history(request: ClearHistoryRequest):
    session_id = request.session_id
    if session_id in chat_histories:
        del chat_histories[session_id]
    return {"message": "History cleared successfully."}