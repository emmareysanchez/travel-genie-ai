import os
import unicodedata
from functools import lru_cache
from typing import Any

import requests
from dotenv import load_dotenv
import logging


load_dotenv()

SERPAPI_URL = "https://serpapi.com/search.json"
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
SERPAPI_TIMEOUT = int(os.getenv("SERPAPI_TIMEOUT", "20"))
SERPAPI_CURRENCY = os.getenv("SERPAPI_CURRENCY", "EUR")
SERPAPI_LANGUAGE = os.getenv("SERPAPI_LANGUAGE", "es")

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


@lru_cache(maxsize=128)
def _resolve_airport(location: str) -> str:
    if not SERPAPI_API_KEY:
        raise ValueError("SERPAPI_API_KEY not found in environment variables.")

    location = location.strip()

    if len(location) == 3 and location.isalpha():
        return location.upper()

    normalized_location = _normalize_text(location)

    params = {
        "engine": "google_flights_autocomplete",
        "q": normalized_location,
        "api_key": SERPAPI_API_KEY,
        "hl": SERPAPI_LANGUAGE,
    }

    response = requests.get(SERPAPI_URL, params=params, timeout=SERPAPI_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    suggestions = data.get("suggestions", [])

    for suggestion in suggestions:
        airports = suggestion.get("airports", [])
        if airports:
            airport_id = airports[0].get("id")
            if airport_id:
                return airport_id

    raise ValueError(f"No se pudo resolver un aeropuerto para: {location}")


def _build_params(
    origin: str,
    destination: str,
    date: str,
    passengers: int,
    return_date: str | None = None,
) -> dict[str, Any]:
    if not SERPAPI_API_KEY:
        raise ValueError("SERPAPI_API_KEY not found in environment variables.")

    params = {
        "engine": "google_flights",
        "api_key": SERPAPI_API_KEY,
        "departure_id": origin,
        "arrival_id": destination,
        "outbound_date": date,
        "currency": SERPAPI_CURRENCY,
        "hl": SERPAPI_LANGUAGE,
        "adults": passengers,
    }

    if return_date:
        params["type"] = 1
        params["return_date"] = return_date
    else:
        params["type"] = 2

    return params


def _extract_segments(flight_item: dict[str, Any]) -> list[dict[str, Any]]:
    return flight_item.get("flights", [])


def _normalize_flight(
    flight_item: dict[str, Any],
    origin: str,
    destination: str,
    date: str,
    return_date: str | None = None,
) -> dict[str, Any]:
    segments = _extract_segments(flight_item)
    if not segments:
        return {}

    first_segment = segments[0]
    last_segment = segments[-1]

    return {
        "airline": first_segment.get("airline", "Unknown"),
        "origin": origin,
        "destination": destination,
        "departure_date": date,
        "return_date": return_date,
        "departure_time": first_segment.get("departure_airport", {}).get("time"),
        "arrival_time": last_segment.get("arrival_airport", {}).get("time"),
        "price": flight_item.get("price"),
        "arrival_airport": last_segment.get("arrival_airport", {}).get("id"),
        "duration": flight_item.get("total_duration"),
        "stops": len(segments) - 1,
        "carbon_emissions": flight_item.get("carbon_emissions"),
        "flight_type": flight_item.get("type"),
        "departure_token": flight_item.get("departure_token"),
    }


def search_flights(origin, destination, date, passengers=1, return_date=None) -> list:
    origin_iata = _resolve_airport(origin)
    destination_iata = _resolve_airport(destination)

    params = _build_params(
        origin=origin_iata,
        destination=destination_iata,
        date=date,
        passengers=passengers,
        return_date=return_date,
    )

    response = requests.get(
        SERPAPI_URL,
        params=params,
        timeout=SERPAPI_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()

    best_flights = data.get("best_flights", [])
    other_flights = data.get("other_flights", [])
    flights = best_flights + other_flights

    normalized = []
    for flight in flights:
        nf = _normalize_flight(
            flight,
            origin_iata,
            destination_iata,
            date,
            return_date=return_date,
        )
        if nf:
            normalized.append(nf)

    return normalized


def select_best_flight(flights: list) -> dict | None:
    valid_flights = [
        f for f in flights
        if isinstance(f.get("price"), (int, float))
    ]
    if not valid_flights:
        return None

    return min(valid_flights, key=lambda x: x["price"])


if __name__ == "__main__":
    origin = "Madrid"
    destination = "París"
    date = "2026-12-15"
    return_date = "2026-12-20"
    passengers = 1

    try:
        flights = search_flights(origin, destination, date, passengers, return_date)
        best_flight = select_best_flight(flights)

        print(f"Mejor vuelo encontrado de {origin} a {destination}:")
        print(best_flight)
    except Exception as e:
        print(f"Error buscando vuelos: {e}")