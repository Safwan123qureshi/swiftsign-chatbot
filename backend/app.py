import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

try:
    from backend.knowledge_base import KNOWLEDGE_BASE
except ImportError:
    try:
        from knowledge_base import KNOWLEDGE_BASE
    except ImportError:
        KNOWLEDGE_BASE = "Swift Sign Group official services and corporate details."

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

@app.get("/api/status")
def status():
    return {"status": "Active", "message": "Swift Sign Group AI Server is running."}

@app.post("/chat")
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"response": "⚠️ GEMINI_API_KEY environment variable missing on Vercel."}

        client = genai.Client(api_key=api_key)

        system_instruction = f"""
You are the official AI representative for Swift Sign Group of Companies, specifically representing {request.brand}.
Use the following knowledge base:
{KNOWLEDGE_BASE}
Provide accurate, professional, and helpful responses based strictly on the company info.
"""

        # Har request par direct generate_content call karein taake client closed na ho
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=request.message,
            config={"system_instruction": system_instruction}
        )

        return {"response": response.text}

    except Exception as e:
        return {"response": f"⚠️ Runtime Error: {str(e)}"}

app = app