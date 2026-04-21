from pydantic import BaseModel, Field


class TripRequest(BaseModel):
    origin: str = Field(..., min_length=2)
    destination: str = Field(..., min_length=2)
    departure_date: str
    return_date: str
    travelers: int = Field(default=1, ge=1)
    budget: float | None = Field(default=None, gt=0)