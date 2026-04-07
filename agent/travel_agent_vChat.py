from services.flights import search_flights, select_best_flight
from services.hotels import search_hotels, select_best_hotel
from services.transport import get_airport_to_hotel_transport
from agent.formatter import format_travel_plan


def travel_agent(trip_request: dict) -> dict:
    """
    Main agent that coordinates flight, hotel, and airport transport search.
    """
    flights = search_flights(trip_request)
    selected_flight = select_best_flight(flights)

    hotels = search_hotels(trip_request)
    selected_hotel = select_best_hotel(hotels)

    if not selected_flight:
        return {"error": "No flights found for the selected trip."}

    if not selected_hotel:
        return {"error": "No hotels found for the selected destination."}

    transport = get_airport_to_hotel_transport(
        selected_flight["arrival_airport"],
        selected_hotel["name"]
    )

    if not transport:
        return {"error": "No transport option found from airport to hotel."}

    return format_travel_plan(selected_flight, selected_hotel, transport)