from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

from travel_agent import TravelAgent

app = FastAPI(title="Travel Genie API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent: TravelAgent | None = None


def get_agent() -> TravelAgent:
    global agent
    if agent is None:
        agent = TravelAgent()
    return agent


class ChatRequest(BaseModel):
    message: str
    reset: bool = False


class TraceEvent(BaseModel):
    type: str
    content: str


class ChatResponse(BaseModel):
    assistant_message: str
    trace: list[TraceEvent]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        travel_agent = get_agent()

        if req.reset:
            travel_agent.reset()
            return {
                "assistant_message": "Conversation reset.",
                "trace": [],
            }

        result = travel_agent.chat(req.message)

        return {
            "assistant_message": result.get("final_answer", ""),
            "trace": result.get("trace", []),
        }

    except Exception as e:
        return {
            "assistant_message": f"Backend error: {e}",
            "trace": [
                {
                    "type": "error",
                    "content": "".join(traceback.format_exception_only(type(e), e)).strip(),
                }
            ],
        }