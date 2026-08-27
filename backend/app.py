import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

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
    return {"status": "Active", "message": "Server running successfully!"}

@app.post("/chat")
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"response": "Error: GEMINI_API_KEY environment variable missing on Vercel."}

        genai.configure(api_key=api_key)

        session_id = request.session_id
        brand = request.brand
        user_message = request.message

        if session_id not in chat_histories:
            system_prompt = f"You are the official AI representative for Swift Sign Group ({brand})."
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system_prompt
            )
            chat_histories[session_id] = model.start_chat(history=[])

        chat = chat_histories[session_id]
        res = chat.send_message(user_message)
        return {"response": res.text}

    except Exception as err:
        return {"response": f"Runtime Error: {str(err)}"}

# Mandatory for Vercel Serverless Entrypoint
app = app