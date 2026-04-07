from data.mocks import MOCK_TRANSPORT


def get_airport_to_hotel_transport(arrival_airport: str, hotel_name: str) -> dict | None:
    """
    Return the transport option from airport to selected hotel.
    """
    return MOCK_TRANSPORT.get((arrival_airport, hotel_name))