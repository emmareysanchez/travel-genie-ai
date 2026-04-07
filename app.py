from fastapi import FastAPI
from agent.parser import TripRequest
from agent.travel_agent import travel_agent

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Travel planning agent is running."}


@app.post("/plan_trip")
def plan_trip(request: TripRequest):
    return travel_agent(request.model_dump())