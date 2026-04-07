from data.mocks import MOCK_FLIGHTS


def search_flights(origin, destination, date, passengers) -> list:
    """
    Return matching mock flights for the requested route and dates.
    """
    results = []

    for flight in MOCK_FLIGHTS:
        if (
            flight["origin"].lower() == origin.lower() and
            flight["destination"].lower() == destination.lower() and
            flight["departure_date"] == date
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