from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import traceback
import json

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
    allow_origin_regex=r"https://.*\.ngrok-free\.dev",
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

def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


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

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def event_generator():
        try:
            travel_agent = get_agent()

            if req.reset:
                travel_agent.reset()
                yield sse_event({
                    "type": "reset",
                    "content": "Conversation reset."
                })
                yield sse_event({
                    "type": "done",
                    "content": ""
                })
                return

            for event in travel_agent.chat_stream(req.message):
                yield sse_event(event)

            yield sse_event({
                "type": "done",
                "content": ""
            })

        except Exception as e:
            yield sse_event({
                "type": "error",
                "content": "".join(traceback.format_exception_only(type(e), e)).strip(),
            })
            yield sse_event({
                "type": "done",
                "content": ""
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )