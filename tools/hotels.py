# import os
# import requests
# import logging
# # from data.mocks import MOCK_HOTELS

# logger = logging.getLogger(__name__)

# RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
# logger.info(f"Using RAPIDAPI_KEY: {'set' if RAPIDAPI_KEY else 'not set'}")
# BASE_URL = "https://booking-com18.p.rapidapi.com/api/v1/hotels"
# HEADERS = {
#     "X-RapidAPI-Host": "booking-com18.p.rapidapi.com",
#     "X-RapidAPI-Key": RAPIDAPI_KEY,
# }

# def _get_destination_id(destination: str) -> str | None:
#     resp = requests.get(
#         f"{BASE_URL}/searchDestination",
#         headers=HEADERS,
#         params={"query": destination}
#     )
#     data = resp.json()
#     results = data.get("data", [])
#     # Coge el primer resultado de tipo "city"
#     for r in results:
#         if r.get("dest_type") == "city":
#             return r["dest_id"]
#     return results[0]["dest_id"] if results else None

# # def search_hotels(destination, check_in, check_out, guests) -> list:
# #     """
# #     Return matching mock hotels for the destination.
# #     """
# #     results = []

# #     for hotel in MOCK_HOTELS:
# #         if hotel["destination"].lower() == destination.lower():
# #             results.append(hotel)

# #     return results

# def search_hotels(destination, check_in, check_out, guests) -> list:
#     dest_id = _get_destination_id(destination)
#     if not dest_id:
#         return []

#     resp = requests.get(
#         f"{BASE_URL}/searchHotels",
#         headers=HEADERS,
#         params={
#             "dest_id": dest_id,
#             "search_type": "city",
#             "arrival_date": check_in,
#             "departure_date": check_out,
#             "adults": guests,
#         }
#     )
#     hotels_raw = resp.json().get("data", {}).get("hotels", [])

#     # Normaliza al mismo formato que tus mocks
#     return [
#         {
#             "name": h["property"]["name"],
#             "destination": destination,
#             "price_per_night": h["property"]["priceBreakdown"]["grossPrice"]["value"],
#             "rating": h["property"]["reviewScore"],
#             "address": h["property"].get("wishlistName", ""),
#         }
#         for h in hotels_raw
#     ]

# # def select_best_hotel(hotels: list) -> dict | None:
# #     """
# #     Select the hotel with the best rating/price balance.
# #     """
# #     if not hotels:
# #         return None

# #     return max(hotels, key=lambda x: x["rating"] / x["price_per_night"])

# def select_best_hotel(hotels: list) -> dict | None:
#     if not hotels:
#         return None
#     return max(hotels, key=lambda x: x["rating"] / x["price_per_night"])

import os
import requests
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
# print(f"Using RAPIDAPI_KEY: {'set' if RAPIDAPI_KEY else 'not set'}")

BASE_URL = "https://booking-com18.p.rapidapi.com/stays"
HEADERS = {
    "x-rapidapi-host": "booking-com18.p.rapidapi.com",
    "x-rapidapi-key": RAPIDAPI_KEY,
    "Content-Type": "application/json",
}


def _get_destination_id(destination: str) -> str | None:
    # print(f"Buscando destination_id para: {destination}")
    # print(f"{BASE_URL}/auto-complete")
    resp = requests.get(
        f"{BASE_URL}/auto-complete",
        headers=HEADERS,
        params={"query": destination}
    )
    data = resp.json()
    # print(data)
    logger.info(f"auto-complete response: {data}")
    results = data.get("data", [])
    if not results:
        return None
    return results[0]["id"]


def search_hotels(destination, check_in, check_out, guests) -> list:
    dest_id = _get_destination_id(destination)
    if not dest_id:
        logger.warning(f"No se encontró destination_id para: {destination}")
        return []

    logger.info(f"Buscando hoteles con locationId: {dest_id}")
    resp = requests.get(
        f"{BASE_URL}/search",
        headers=HEADERS,
        params={
            "locationId": dest_id,
            "checkinDate": check_in,
            "checkoutDate": check_out,
            "adults": guests,
            "units": "metric",
            "temperature": "c",
        }
    )
    data = resp.json()
    logger.info(f"search response keys: {list(data.keys())}")
    hotels_raw = data.get("data", [])

    results = []
    for h in hotels_raw:
        try:
            results.append({
                "name": h["name"],
                "destination": destination,
                "price_per_night": h["priceBreakdown"]["grossPrice"]["value"],
                "rating": h.get("reviewScore", 0),
                "address": h.get("address", ""),
            })
        except (KeyError, TypeError) as e:
            logger.warning(f"Error parseando hotel: {e} — {h}")
            continue

    logger.info(f"Hoteles encontrados: {len(results)}")
    return results


def select_best_hotel(hotels: list) -> dict | None:
    if not hotels:
        return None
    return max(hotels, key=lambda x: x["rating"] / x["price_per_night"])

if __name__ == "__main__":
    print(_get_destination_id("Paris"))
    print(search_hotels("Paris", "2026-06-10", "2026-06-13", 1))
    print(search_hotels("Madrid", "2026-06-10", "2026-06-13", 1))
    print(search_hotels("London", "2026-06-10", "2026-06-13", 1))