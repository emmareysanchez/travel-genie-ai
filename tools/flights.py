from data.mocks import MOCK_FLIGHTS


def search_flights(trip_request: dict) -> list:
    """
    Return matching mock flights for the requested route and dates.
    """
    results = []

    for flight in MOCK_FLIGHTS:
        if (
            flight["origin"].lower() == trip_request["origin"].lower()
            and flight["destination"].lower() == trip_request["destination"].lower()
            and flight["departure_date"] == trip_request["departure_date"]
            and flight["return_date"] == trip_request["return_date"]
        ):
            results.append(flight)

    return results


def select_best_flight(flights: list) -> dict | None:
    """
    Select the cheapest flight.
    """
    if not flights:
        return None

    return min(flights, key=lambda x: x["price"])