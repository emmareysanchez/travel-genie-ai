from pydantic import BaseModel


class TripRequest(BaseModel):
    origin: str
    destination: str
    departure_date: str
    return_date: str
    travelers: int = 1
    budget: float | None = None