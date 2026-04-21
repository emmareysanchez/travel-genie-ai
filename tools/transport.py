"""
transport_tool.py
-----------------
Herramienta para calcular rutas entre un aeropuerto y un hotel.

APIs utilizadas:
  - API Ninjas /v1/airports  →  coordenadas del aeropuerto por código IATA
  - Geoapify Forward Geocoding →  coordenadas del hotel por dirección
  - Geoapify Routing API       →  ruta entre ambos puntos
"""

import os
import urllib.parse
import urllib.request
import json
from typing import TypedDict

# ---------------------------------------------------------------------------
# Configuración de API keys
# Puedes definirlas directamente aquí o como variables de entorno.
# ---------------------------------------------------------------------------
APININJAS_API_KEY: str = os.getenv("APININJAS_API_KEY", "TU_API_KEY_AQUI")
GEOAPIFY_API_KEY: str = os.getenv("GEOAPIFY_API_KEY", "TU_API_KEY_AQUI")

# ---------------------------------------------------------------------------
# Modos de transporte válidos según la Geoapify Routing API
# ---------------------------------------------------------------------------
VALID_TRANSPORT_MODES: set[str] = {
    "drive",
    "light_truck",
    "medium_truck",
    "truck",
    "heavy_truck",
    "truck_dangerous_goods",
    "long_truck",
    "bus",
    "scooter",
    "motorcycle",
    "bicycle",
    "mountain_bike",
    "road_bike",
    "walk",
    "hike",
    "transit",
    "approximated_transit",
}


# ---------------------------------------------------------------------------
# Tipos de retorno
# ---------------------------------------------------------------------------
class Coordinates(TypedDict):
    latitude: float
    longitude: float


class RouteResult(TypedDict):
    distance_meters: float
    distance_units: str
    duration_seconds: float
    duration_formatted: str
    transport_type: str
    origin: dict
    destination: dict


# ---------------------------------------------------------------------------
# Errores personalizados
# ---------------------------------------------------------------------------
class TransportToolError(Exception):
    """Error base de la herramienta de transporte."""


class InvalidTransportModeError(TransportToolError):
    """El modo de transporte no es uno de los 17 aceptados por Geoapify."""


class AirportNotFoundError(TransportToolError):
    """No se encontró el aeropuerto con el código IATA indicado."""


class HotelNotFoundError(TransportToolError):
    """No se pudo geocodificar la dirección del hotel."""


class RoutingError(TransportToolError):
    """La API de rutas devolvió un error o no encontró ruta."""


class APIError(TransportToolError):
    """Error genérico de comunicación con una API externa."""


# ---------------------------------------------------------------------------
# Utilidad interna: petición HTTP GET → dict JSON
# ---------------------------------------------------------------------------
def _get_json(url: str, headers: dict | None = None) -> dict:
    """
    Realiza una petición GET a `url` con las cabeceras opcionales `headers`
    y devuelve la respuesta parseada como dict.

    Raises:
        APIError: si el código HTTP no es 200 o la respuesta no es JSON válido.
    """
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.status
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        raise APIError(
            f"Error HTTP {exc.code} al llamar a {url}.\nRespuesta: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise APIError(
            f"No se pudo conectar con {url}: {exc.reason}"
        ) from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise APIError(
            f"La respuesta de {url} no es JSON válido.\nRespuesta raw: {body}"
        ) from exc


# ---------------------------------------------------------------------------
# Función 1: coordenadas del aeropuerto
# ---------------------------------------------------------------------------
def get_airport_coordinates(iata_code: str) -> Coordinates:
    """
    Obtiene la latitud y longitud de un aeropuerto a partir de su código IATA.

    Endpoint: GET https://api.api-ninjas.com/v1/airports?iata=<IATA>
    Cabecera requerida: X-Api-Key

    Args:
        iata_code: Código IATA de 3 letras (p. ej. "MAD", "JFK", "CDG").

    Returns:
        Diccionario con las claves `latitude` y `longitude` (float).

    Raises:
        ValueError: si el código IATA no tiene exactamente 3 letras.
        AirportNotFoundError: si la API no devuelve ningún aeropuerto.
        APIError: si ocurre un error de comunicación con la API.
    """
    iata_code = iata_code.strip().upper()

    if len(iata_code) != 3 or not iata_code.isalpha():
        raise ValueError(
            f"El código IATA debe tener exactamente 3 letras. "
            f"Recibido: '{iata_code}'"
        )

    url = (
        "https://api.api-ninjas.com/v1/airports"
        f"?iata={urllib.parse.quote(iata_code)}"
    )
    headers = {"X-Api-Key": APININJAS_API_KEY}

    data = _get_json(url, headers=headers)

    # La API devuelve una lista; tomamos el primer resultado
    if not isinstance(data, list) or len(data) == 0:
        raise AirportNotFoundError(
            f"No se encontró ningún aeropuerto con código IATA '{iata_code}'."
        )

    airport = data[0]

    try:
        return {
            "latitude": float(airport["latitude"]),
            "longitude": float(airport["longitude"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise APIError(
            f"La respuesta de API Ninjas no contiene coordenadas válidas "
            f"para el aeropuerto '{iata_code}'. Respuesta: {airport}"
        ) from exc


# ---------------------------------------------------------------------------
# Función 2: coordenadas del hotel
# ---------------------------------------------------------------------------
def get_hotel_coordinates(address: str) -> Coordinates:
    """
    Geocodifica la dirección de un hotel y devuelve sus coordenadas.

    Endpoint: GET https://api.geoapify.com/v1/geocode/search
    Parámetros: text=<dirección>, format=json, limit=1, apiKey=<key>

    Args:
        address: Dirección completa del hotel (p. ej.
                 "Gran Vía 32, 28013 Madrid, España").

    Returns:
        Diccionario con las claves `latitude` y `longitude` (float).

    Raises:
        ValueError: si la dirección está vacía.
        HotelNotFoundError: si la API no devuelve resultados.
        APIError: si ocurre un error de comunicación con la API.
    """
    address = address.strip()
    if not address:
        raise ValueError("La dirección del hotel no puede estar vacía.")

    params = urllib.parse.urlencode(
        {
            "text": address,
            "format": "json",
            "limit": 1,
            "apiKey": GEOAPIFY_API_KEY,
        }
    )
    url = f"https://api.geoapify.com/v1/geocode/search?{params}"

    data = _get_json(url)

    results = data.get("results", [])
    if not results:
        raise HotelNotFoundError(
            f"No se encontraron coordenadas para la dirección: '{address}'. "
            "Comprueba que la dirección es correcta e inténtalo de nuevo."
        )

    first = results[0]
    try:
        return {
            "latitude": float(first["lat"]),
            "longitude": float(first["lon"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise APIError(
            f"La respuesta de Geoapify Geocoding no contiene coordenadas "
            f"válidas. Respuesta: {first}"
        ) from exc


# ---------------------------------------------------------------------------
# Función 3: ruta aeropuerto → hotel
# ---------------------------------------------------------------------------
def get_airport_to_hotel_transport(
    airport: str,
    hotel: str,
    transport_type: str,
) -> RouteResult:
    """
    Calcula la ruta entre un aeropuerto y un hotel usando Geoapify Routing API.

    Args:
        airport:        Código IATA del aeropuerto de origen (p. ej. "MAD").
        hotel:          Dirección completa del hotel de destino.
        transport_type: Modo de transporte. Debe ser uno de los 17 valores
                        aceptados por Geoapify:
                          drive, light_truck, medium_truck, truck, heavy_truck,
                          truck_dangerous_goods, long_truck, bus, scooter,
                          motorcycle, bicycle, mountain_bike, road_bike,
                          walk, hike, transit, approximated_transit.

    Returns:
        Diccionario con:
          - distance_meters   (float)   distancia total en metros
          - distance_units    (str)     unidades ("meters")
          - duration_seconds  (float)   tiempo estimado en segundos
          - duration_formatted (str)    tiempo en formato legible "Xh Ym"
          - transport_type    (str)     modo de transporte usado
          - origin            (dict)    {iata, latitude, longitude}
          - destination       (dict)    {address, latitude, longitude}

    Raises:
        InvalidTransportModeError: si `transport_type` no es válido.
        AirportNotFoundError:      si el código IATA no se encuentra.
        HotelNotFoundError:        si la dirección no puede geocodificarse.
        RoutingError:              si la API de rutas no encuentra ruta.
        APIError:                  si ocurre cualquier otro error de API.
        ValueError:                si algún parámetro tiene formato incorrecto.
    """
    # --- Validar modo de transporte ---
    transport_type = transport_type.strip().lower()
    if transport_type not in VALID_TRANSPORT_MODES:
        valid_list = "\n  ".join(sorted(VALID_TRANSPORT_MODES))
        raise InvalidTransportModeError(
            f"'{transport_type}' no es un modo de transporte válido.\n"
            f"Modos aceptados:\n  {valid_list}"
        )

    # --- Obtener coordenadas del aeropuerto ---
    airport_coords = get_airport_coordinates(airport)

    # --- Obtener coordenadas del hotel ---
    hotel_coords = get_hotel_coordinates(hotel)

    # --- Llamar a la Routing API ---
    # Formato de waypoints: "lat,lon|lat,lon"
    origin_wp = f"{airport_coords['latitude']},{airport_coords['longitude']}"
    dest_wp = f"{hotel_coords['latitude']},{hotel_coords['longitude']}"
    waypoints = f"{origin_wp}|{dest_wp}"

    params = urllib.parse.urlencode(
        {
            "waypoints": waypoints,
            "mode": transport_type,
            "format": "json",
            "units": "metric",
            "apiKey": GEOAPIFY_API_KEY,
        }
    )
    url = f"https://api.geoapify.com/v1/routing?{params}"

    data = _get_json(url)

    # Detectar errores devueltos por Geoapify en el cuerpo JSON
    if "statusCode" in data or "error" in data:
        status = data.get("statusCode", "?")
        message = data.get("message", data.get("error", "Error desconocido"))
        raise RoutingError(
            f"Geoapify Routing devolvió un error (HTTP {status}): {message}"
        )

    results = data.get("results", [])
    if not results:
        raise RoutingError(
            f"Geoapify Routing no encontró ninguna ruta entre el aeropuerto "
            f"'{airport}' y el hotel '{hotel}' usando el modo '{transport_type}'."
        )

    route = results[0]

    distance = float(route.get("distance", 0))
    duration_s = float(route.get("time", 0))

    # Formatear la duración en horas y minutos
    hours = int(duration_s // 3600)
    minutes = int((duration_s % 3600) // 60)
    if hours > 0:
        duration_fmt = f"{hours}h {minutes}min"
    else:
        duration_fmt = f"{minutes}min"

    return {
        "distance_meters": distance,
        "distance_units": route.get("distance_units", "meters"),
        "duration_seconds": duration_s,
        "duration_formatted": duration_fmt,
        "transport_type": transport_type,
        "origin": {
            "iata": airport.upper(),
            "latitude": airport_coords["latitude"],
            "longitude": airport_coords["longitude"],
        },
        "destination": {
            "address": hotel,
            "latitude": hotel_coords["latitude"],
            "longitude": hotel_coords["longitude"],
        },
    }


# ---------------------------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=== Transport Tool — ejemplo de uso ===\n")

    # Ejemplo: vuelo a Madrid → hotel en el centro
    try:
        result = get_airport_to_hotel_transport(
            airport="MAD",
            hotel="Gran Vía 32, 28013 Madrid, España",
            transport_type="drive",
        )

        print(f"✈  Aeropuerto : {result['origin']['iata']}")
        print(f"   Coordenadas: {result['origin']['latitude']}, {result['origin']['longitude']}")
        print(f"🏨 Hotel      : {result['destination']['address']}")
        print(f"   Coordenadas: {result['destination']['latitude']}, {result['destination']['longitude']}")
        print(f"🚗 Transporte : {result['transport_type']}")
        print(f"📏 Distancia  : {result['distance_meters'] / 1000:.1f} km")
        print(f"⏱  Duración   : {result['duration_formatted']}")

    except InvalidTransportModeError as e:
        print(f"[ERROR] Modo de transporte inválido:\n{e}", file=sys.stderr)
    except AirportNotFoundError as e:
        print(f"[ERROR] Aeropuerto no encontrado:\n{e}", file=sys.stderr)
    except HotelNotFoundError as e:
        print(f"[ERROR] Hotel no encontrado:\n{e}", file=sys.stderr)
    except RoutingError as e:
        print(f"[ERROR] Error al calcular la ruta:\n{e}", file=sys.stderr)
    except APIError as e:
        print(f"[ERROR] Error de API:\n{e}", file=sys.stderr)
    except ValueError as e:
        print(f"[ERROR] Parámetro inválido:\n{e}", file=sys.stderr)