# from data.mocks import MOCK_FLIGHTS


# def search_flights(origin, destination, date, passengers) -> list:
#     """
#     Return matching mock flights for the requested route and dates.
#     """
#     results = []

#     for flight in MOCK_FLIGHTS:
#         if (
#             flight["origin"].lower() == origin.lower() and
#             flight["destination"].lower() == destination.lower() and
#             flight["departure_date"] == date
#         ):
#             results.append(flight)

#     return results


# def select_best_flight(flights: list) -> dict | None:
#     """
#     Select the cheapest flight.
#     """
#     if not flights:
#         return None

#     return min(flights, key=lambda x: x["price"])

import os
import unicodedata
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
    """
    Remove accents from text.
    Example: París -> Paris
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _resolve_airport(location: str) -> str:
    """
    Resolve a city or airport-like input into an IATA airport code.
    Uses SerpApi Google Flights Autocomplete.
    """
    if not SERPAPI_API_KEY:
        raise ValueError("SERPAPI_API_KEY not found in environment variables.")

    location = location.strip()

    # If user already passed an IATA code like MAD, CDG, JFK
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


def _build_params(origin: str, destination: str, date: str, passengers: int) -> dict[str, Any]:
    """
    Build query params for SerpApi Google Flights.
    """
    if not SERPAPI_API_KEY:
        raise ValueError("SERPAPI_API_KEY not found in environment variables.")

    return {
        "engine": "google_flights",
        "api_key": SERPAPI_API_KEY,
        "departure_id": origin,
        "arrival_id": destination,
        "outbound_date": date,
        "currency": SERPAPI_CURRENCY,
        "hl": SERPAPI_LANGUAGE,
        "type": 2,  # one way
        "adults": passengers,
    }


def _extract_segments(flight_item: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract flight segments from SerpApi response.
    """
    return flight_item.get("flights", [])


def _normalize_flight(
    flight_item: dict[str, Any],
    origin: str,
    destination: str,
    date: str,
) -> dict[str, Any]:
    """
    Normalize SerpApi flight structure to project format.
    """
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
        "return_date": None,
        "departure_time": first_segment.get("departure_airport", {}).get("time"),
        "arrival_time": last_segment.get("arrival_airport", {}).get("time"),
        "price": flight_item.get("price"),
        "arrival_airport": last_segment.get("arrival_airport", {}).get("id"),
        "duration": flight_item.get("total_duration"),
        "stops": len(segments) - 1,
        "carbon_emissions": flight_item.get("carbon_emissions"),
    }


def search_flights(origin, destination, date, passengers=1) -> list:
    """
    Search flights using SerpApi Google Flights.
    """
    origin_iata = _resolve_airport(origin)
    destination_iata = _resolve_airport(destination)

    params = _build_params(
        origin=origin_iata,
        destination=destination_iata,
        date=date,
        passengers=passengers,
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

    return [
        _normalize_flight(flight, origin_iata, destination_iata, date)
        for flight in flights
        if flight
    ]


def select_best_flight(flights: list) -> dict | None:
    """
    Select cheapest available flight.
    """
    if not flights:
        return None

    return min(flights, key=lambda x: x["price"])


if __name__ == "__main__":
    # Prueba de ejemplo para verificar que la API funciona correctamente.
    origin = "Madrid"
    destination = "París"
    date = "2026-12-15"
    passengers = 1

    try:
        flights = search_flights(origin, destination, date, passengers)
        best_flight = select_best_flight(flights)

        print(f"Mejor vuelo encontrado de {origin} a {destination} el {date}:")
        print(best_flight)
    except Exception as e:
        print(f"Error buscando vuelos: {e}")