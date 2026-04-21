import os
import requests
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_TIMEOUT = int(os.getenv("RAPIDAPI_TIMEOUT", "15"))

BASE_URL = "https://booking-com18.p.rapidapi.com/stays"
HEADERS = {
    "x-rapidapi-host": "booking-com18.p.rapidapi.com",
    "x-rapidapi-key": RAPIDAPI_KEY,
    "Content-Type": "application/json",
}


def _safe_get(url: str, params: dict) -> dict:
    if not RAPIDAPI_KEY:
        raise ValueError("RAPIDAPI_KEY not found in environment variables.")

    resp = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=RAPIDAPI_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _get_destination_id(destination: str) -> str | None:
    data = _safe_get(
        f"{BASE_URL}/auto-complete",
        {"query": destination},
    )
    logger.info(f"auto-complete response: {data}")

    results = data.get("data", [])
    if not results:
        return None

    return results[0]["id"]


def search_hotels(destination, check_in, check_out, guests=1) -> list:
    dest_id = _get_destination_id(destination)
    if not dest_id:
        logger.warning(f"No se encontró destination_id para: {destination}")
        return []

    logger.info(f"Buscando hoteles con locationId: {dest_id}")
    data = _safe_get(
        f"{BASE_URL}/search",
        {
            "locationId": dest_id,
            "checkinDate": check_in,
            "checkoutDate": check_out,
            "adults": guests,
            "units": "metric",
            "temperature": "c",
        },
    )

    logger.info(f"search response keys: {list(data.keys())}")
    hotels_raw = data.get("data", [])

    results = []
    for h in hotels_raw:
        if "latitude" not in h or "longitude" not in h:
            # logger.warning(f"Hotel sin coordenadas, omitido: {h.get('name', 'desconocido')}")
            continue
        try: 
            results.append({
                "name": h["name"],
                "destination": destination,
                "price_per_night": h["priceBreakdown"]["grossPrice"]["value"],
                "rating": h.get("reviewScore", 0),
                "latitude": h["latitude"],
                "longitude": h["longitude"],
            })
        except (KeyError, TypeError) as e:
            logger.warning(f"Error parseando hotel: {e} — {h}")
            continue

    logger.info(f"Hoteles encontrados: {len(results)}")
    logger.debug(f"Hoteles detallados: {results}")
    return results


def select_best_hotel(hotels: list) -> dict | None:
    valid_hotels = [
        h for h in hotels
        if isinstance(h.get("price_per_night"), (int, float)) and h["price_per_night"] > 0
    ]
    if not valid_hotels:
        return None

    return max(valid_hotels, key=lambda x: (x.get("rating", 0) or 0) / x["price_per_night"])


if __name__ == "__main__":
    print(_get_destination_id("Paris"))
    print(search_hotels("Paris", "2026-06-10", "2026-06-13", 1))