from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.travel_agent import TravelAgent

app = FastAPI(title="Travel Genie API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = TravelAgent()

class ChatRequest(BaseModel):
    message: str
    reset: bool = False

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(payload: ChatRequest):
    if payload.reset:
        agent.reset()
    result = agent.chat(payload.message)

    return {
        "assistant_message": result["final_answer"],
        "trace": result["trace"]
    }