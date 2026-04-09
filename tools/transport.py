# from data.mocks import MOCK_TRANSPORT
# def get_airport_to_hotel_transport(airport: str, destination: str, datetime: str) -> dict | None:
#     """
#     Return the transport option from airport to selected destination.
#     """
#     return MOCK_TRANSPORT.get((airport, destination))

import os
import httpx
from typing import Any, Literal


# ─── Constantes ────────────────────────────────────────────────────────────────

MAPBOX_BASE_URL = "https://api.mapbox.com/directions/v5"
MAPBOX_TOKEN    = os.environ["MAPBOX_ACCESS_TOKEN"]

TransportProfile = Literal[
    "mapbox/driving-traffic",
    "mapbox/driving",
    "mapbox/walking",
    "mapbox/cycling",
]


# ─── Helpers de geocodificación ────────────────────────────────────────────────

def _geocode(place: str) -> tuple[float, float]:
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{httpx.utils.quote(place)}.json"
    resp = httpx.get(url, params={"access_token": MAPBOX_TOKEN, "limit": 1})
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if not features:
        raise ValueError(f"No se pudo geocodificar: {place!r}")
    lon, lat = features[0]["center"]
    return lon, lat


def _build_coordinates(origin: str, destination: str) -> str:
    lon_o, lat_o = _geocode(origin)
    lon_d, lat_d = _geocode(destination)
    return f"{lon_o},{lat_o};{lon_d},{lat_d}"


# ─── Parser de la respuesta ────────────────────────────────────────────────────

def _parse_route(data: dict[str, Any]) -> dict | None:
    if data.get("code") != "Ok":
        return None

    routes = data.get("routes")
    if not routes:
        return None

    route = routes[0]
    leg   = route["legs"][0]

    duration_s = route["duration"]
    distance_m = route["distance"]

    return {
        "duration_seconds": int(duration_s),
        "duration_minutes": round(duration_s / 60),
        "distance_meters":  int(distance_m),
        "distance_km":      round(distance_m / 1000, 2),
        "summary":          leg.get("summary", ""),
    }


# ─── Tool principal ────────────────────────────────────────────────────────────

VALID_PROFILES: set[str] = {
    "mapbox/driving-traffic",
    "mapbox/driving",
    "mapbox/walking",
    "mapbox/cycling",
}

def get_airport_to_hotel_transport(
    airport: str,
    destination: str,
    datetime: str,
    profile: TransportProfile = "mapbox/driving-traffic",
) -> dict | None:
    """
    Devuelve información de transporte desde un aeropuerto hasta un destino.

    Parámetros
    ----------
    airport     : nombre o dirección del aeropuerto de origen.
    destination : nombre o dirección del destino (hotel, zona, etc.).
    datetime    : fecha/hora de salida en formato ISO 8601 (p.ej. "2025-06-15T14:30").
    profile     : perfil de transporte. Opciones:
                    - "mapbox/driving-traffic"  (coche con tráfico en tiempo real, por defecto)
                    - "mapbox/driving"          (coche sin tráfico en tiempo real)
                    - "mapbox/walking"          (a pie)
                    - "mapbox/cycling"          (bicicleta)
    """
    if profile not in VALID_PROFILES:
        raise ValueError(
            f"Perfil no válido: {profile!r}. "
            f"Opciones permitidas: {sorted(VALID_PROFILES)}"
        )

    coordinates = _build_coordinates(airport, destination)

    params: dict[str, Any] = {
        "access_token": MAPBOX_TOKEN,
        "overview":     "false",
        "steps":        "false",
    }

    # depart_at solo está soportado por los perfiles de conducción
    if profile in {"mapbox/driving-traffic", "mapbox/driving"}:
        params["depart_at"] = datetime

    url  = f"{MAPBOX_BASE_URL}/{profile}/{coordinates}"
    resp = httpx.get(url, params=params, timeout=10)
    resp.raise_for_status()

    return _parse_route(resp.json())