# Mock data for initial testing

MOCK_FLIGHTS = [
    {
        "airline": "Iberia",
        "origin": "Madrid",
        "destination": "Paris",
        "departure_date": "2026-06-10",
        "return_date": "2026-06-13",
        "departure_time": "08:30",
        "arrival_time": "10:25",
        "price": 145,
        "arrival_airport": "CDG"
    },
    {
        "airline": "Air France",
        "origin": "Madrid",
        "destination": "Paris",
        "departure_date": "2026-06-10",
        "return_date": "2026-06-13",
        "departure_time": "12:10",
        "arrival_time": "14:05",
        "price": 170,
        "arrival_airport": "ORY"
    }
]

MOCK_HOTELS = [
    {
        "name": "Hotel Central Paris",
        "destination": "Paris",
        "price_per_night": 120,
        "rating": 8.7,
        "address": "12 Rue du Centre, Paris"
    },
    {
        "name": "Budget Stay Paris",
        "destination": "Paris",
        "price_per_night": 85,
        "rating": 7.9,
        "address": "45 Avenue de la Gare, Paris"
    }
]

MOCK_TRANSPORT = {
    ("CDG", "Hotel Central Paris"): {
        "mode": "RER B + Metro",
        "estimated_time": "45 min",
        "estimated_price": 12
    },
    ("ORY", "Hotel Central Paris"): {
        "mode": "OrlyBus + Metro",
        "estimated_time": "50 min",
        "estimated_price": 14
    },
    ("CDG", "Budget Stay Paris"): {
        "mode": "Taxi",
        "estimated_time": "35 min",
        "estimated_price": 55
    },
    ("ORY", "Budget Stay Paris"): {
        "mode": "Taxi",
        "estimated_time": "30 min",
        "estimated_price": 40
    }
}