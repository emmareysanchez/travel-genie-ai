from data.mocks import MOCK_TRANSPORT


def get_airport_to_hotel_transport(airport: str, destination: str, datetime: str) -> dict | None:
    """
    Return the transport option from airport to selected destination.
    """
    return MOCK_TRANSPORT.get((airport, destination))