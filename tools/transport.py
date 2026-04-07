from data.mocks import MOCK_TRANSPORT


def get_airport_to_hotel_transport(airport: str, hotel_name: str) -> dict | None:
    """
    Return the transport option from airport to selected hotel.
    """
    return MOCK_TRANSPORT.get((airport, hotel_name))