from data.mocks import MOCK_HOTELS


def search_hotels(city, check_in, check_out, guests) -> list:
    """
    Return matching mock hotels for the destination.
    """
    results = []

    for hotel in MOCK_HOTELS:
        if hotel["city"].lower() == city.lower():
            results.append(hotel)

    return results


def select_best_hotel(hotels: list) -> dict | None:
    """
    Select the hotel with the best rating/price balance.
    """
    if not hotels:
        return None

    return max(hotels, key=lambda x: x["rating"] / x["price_per_night"])