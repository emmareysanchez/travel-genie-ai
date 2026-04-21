"""
places_of_interest.py
---------------------
Tool para buscar lugares de interés cercanos a una ubicación.

API utilizada:
  - Geoapify Places API  →  búsqueda de POIs por categoría y radio
  - Geoapify Forward Geocoding  →  geocodificación de la ubicación (si es texto)

Variables de entorno requeridas:
  - GEOAPIFY_API_KEY
"""

import os
import httpx
from typing import Any
from dotenv import load_dotenv


# ─── Configuración ─────────────────────────────────────────────────────────────
load_dotenv()  # Carga variables de entorno desde .env si existe
GEOAPIFY_API_KEY: str = os.getenv("GEOAPIFY_API_KEY", "TU_API_KEY_AQUI")
PLACES_BASE_URL     = "https://api.geoapify.com/v2/places"
GEOCODING_BASE_URL  = "https://api.geoapify.com/v1/geocode/search"
REQUEST_TIMEOUT     = 10  # segundos


# ─── Errores personalizados ────────────────────────────────────────────────────

class PlacesAPIError(Exception):
    """Error genérico al llamar a la API de Places."""

class LocationNotFoundError(Exception):
    """No se han podido obtener coordenadas para la ubicación indicada."""

class InvalidInterestTypeError(Exception):
    """Uno o más interest_types no son reconocidos."""

class InvalidConditionError(Exception):
    """Una o más conditions no son reconocidas."""


# ─── Mapeo de tipos de interés → categorías Geoapify ──────────────────────────
#
# Los valores son strings con una o varias categorías separadas por coma,
# tal como los acepta el parámetro `categories` de la API.

INTEREST_TYPE_MAP: dict[str, str] = {
    # Turismo y cultura
    "monumentos":       "tourism.sights,tourism.attraction",
    "museos":           "entertainment.museum",
    "cultura":          "entertainment.culture",
    "patrimonio":       "heritage",
    "arte":             "tourism.attraction.artwork,entertainment.culture.gallery",
    "arqueologia":      "tourism.sights.archaeological_site",

    # Gastronomía
    "restaurantes":     "catering.restaurant",
    "cafes":            "catering.cafe",
    "bares":            "catering.bar,catering.pub",
    "comida_rapida":    "catering.fast_food",
    "tapas":            "catering.fast_food.tapas,catering.restaurant.tapas",

    # Naturaleza y espacios al aire libre
    "parques":          "leisure.park",
    "naturaleza":       "natural",
    "playas":           "beach",
    "montanas":         "natural.mountain",

    # Entretenimiento y ocio
    "ocio":             "entertainment",
    "vida_nocturna":    "adult.nightclub,catering.bar,adult.casino",
    "deporte":          "sport",
    "parques_acuaticos":"entertainment.water_park",
    "zoologico":        "entertainment.zoo",
    "cine":             "entertainment.cinema",
    "teatro":           "entertainment.culture.theatre",

    # Servicios para el viajero
    "transporte":       "public_transport",
    "farmacias":        "commercial.health_and_beauty.pharmacy",
    "supermercados":    "commercial.supermarket",
    "cajeros":          "service.financial.atm",
    "bancos":           "service.financial.bank",
    "aparcamiento":     "parking.cars",
    "alquiler_coches":  "rental.car",
    "alquiler_bici":    "rental.bicycle",
    "hospitales":       "healthcare.hospital",
    "oficina_turismo":  "tourism.information.office",
}

# Condiciones válidas admitidas por la API (subconjunto más útil para viajeros)
VALID_CONDITIONS: frozenset[str] = frozenset({
    "wheelchair",
    "wheelchair.yes",
    "wheelchair.limited",
    "internet_access",
    "internet_access.free",
    "internet_access.for_customers",
    "dogs",
    "dogs.yes",
    "dogs.leashed",
    "no-dogs",
    "access",
    "access.yes",
    "fee",
    "no_fee",
    "named",
    "vegetarian",
    "vegetarian.only",
    "vegan",
    "vegan.only",
    "halal",
    "halal.only",
    "kosher",
    "kosher.only",
    "organic",
    "organic.only",
    "gluten_free",
})


# ─── Helpers internos ──────────────────────────────────────────────────────────

def _parse_coordinates(location: str) -> tuple[float, float] | None:
    """
    Intenta interpretar `location` como 'lat,lon'.
    Devuelve (lat, lon) si tiene éxito, None si no.
    """
    parts = location.strip().split(",")
    if len(parts) != 2:
        return None
    try:
        lat, lon = float(parts[0].strip()), float(parts[1].strip())
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except ValueError:
        pass
    return None


def _geocode_location(location: str) -> tuple[float, float]:
    """
    Geocodifica una dirección o nombre de lugar con la Geoapify Forward Geocoding API.
    Devuelve (lat, lon).
    Lanza LocationNotFoundError si no se encuentra resultado.
    """
    try:
        response = httpx.get(
            GEOCODING_BASE_URL,
            params={
                "text":   location,
                "format": "json",
                "limit":  1,
                "apiKey": GEOAPIFY_API_KEY,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise PlacesAPIError(
            f"Error HTTP al geocodificar '{location}': {exc.response.status_code} {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:
        raise PlacesAPIError(f"Error de red al geocodificar: {exc}") from exc

    data = response.json()
    results = data.get("results", [])
    if not results:
        raise LocationNotFoundError(
            f"No se encontraron coordenadas para la ubicación: '{location}'"
        )

    return float(results[0]["lat"]), float(results[0]["lon"])


def _resolve_coordinates(location: str) -> tuple[float, float]:
    """
    Resuelve la ubicación a coordenadas (lat, lon):
      - Si es 'lat,lon' parseable → coordenadas directas (sin llamada de red).
      - Si es texto → geocodificación.
    """
    coords = _parse_coordinates(location)
    if coords is not None:
        return coords
    return _geocode_location(location)


def _build_categories(interest_types: list[str]) -> str:
    """
    Traduce la lista de interest_types al string de categorías de Geoapify.
    Acepta tanto claves del mapa (ej. 'museos') como categorías raw (ej. 'catering.restaurant').
    Lanza InvalidInterestTypeError si alguna clave no es reconocida ni parece una categoría válida.
    """
    categories: list[str] = []
    unknown: list[str] = []

    for itype in interest_types:
        normalized = itype.strip().lower()
        if normalized in INTEREST_TYPE_MAP:
            categories.append(INTEREST_TYPE_MAP[normalized])
        elif "." in normalized:
            # Asumir que es una categoría Geoapify pasada directamente (raw)
            categories.append(normalized)
        else:
            unknown.append(itype)

    if unknown:
        valid_keys = sorted(INTEREST_TYPE_MAP.keys())
        raise InvalidInterestTypeError(
            f"Tipos de interés no reconocidos: {unknown}. "
            f"Válidos: {valid_keys}. "
            "También puedes pasar categorías Geoapify directamente (p.ej. 'catering.restaurant')."
        )

    # Aplanar y deduplicar manteniendo orden
    seen: set[str] = set()
    flat: list[str] = []
    for cat_group in categories:
        for cat in cat_group.split(","):
            cat = cat.strip()
            if cat and cat not in seen:
                seen.add(cat)
                flat.append(cat)

    return ",".join(flat)


def _parse_feature(feature: dict) -> dict:
    """
    Extrae los campos relevantes de un feature GeoJSON de la respuesta de Places API.
    """
    props = feature.get("properties", {})
    geometry = feature.get("geometry", {})
    coordinates = geometry.get("coordinates", [None, None])

    return {
        "name":             props.get("name") or props.get("address_line1", "Sin nombre"),
        "categories":       props.get("categories", []),
        "address":          props.get("formatted", ""),
        "address_line1":    props.get("address_line1", ""),
        "address_line2":    props.get("address_line2", ""),
        "distance_meters":  round(props["distance"]) if "distance" in props else None,
        "lat":              props.get("lat") or (coordinates[1] if len(coordinates) > 1 else None),
        "lon":              props.get("lon") or (coordinates[0] if len(coordinates) > 0 else None),
        "place_id":         props.get("place_id"),  # útil para Place Details API
    }


# ─── Tool principal ────────────────────────────────────────────────────────────

def search_places_of_interest(
    location: str,
    interest_types: list[str],
    radius_meters: int = 2000,
    limit: int = 10,
    conditions: list[str] | None = None,
    lang: str = "es",
) -> list[dict]:
    """
    Busca lugares de interés cercanos a una ubicación usando la Geoapify Places API.

    Parámetros
    ----------
    location : str
        Dirección, nombre de lugar o coordenadas en formato 'lat,lon'.
        Si el agente ya dispone de coordenadas del hotel (del paso anterior),
        puede pasarlas directamente en formato 'lat,lon' para evitar una
        geocodificación innecesaria.

    interest_types : list[str]
        Lista de tipos de interés. Valores admitidos:
          Turismo/cultura : monumentos, museos, cultura, patrimonio, arte, arqueologia
          Gastronomía     : restaurantes, cafes, bares, comida_rapida, tapas
          Naturaleza      : parques, naturaleza, playas, montanas
          Entretenimiento : ocio, vida_nocturna, deporte, parques_acuaticos,
                            zoologico, cine, teatro
          Servicios       : transporte, farmacias, supermercados, cajeros, bancos,
                            aparcamiento, alquiler_coches, alquiler_bici,
                            hospitales, oficina_turismo
        También acepta categorías Geoapify directamente (p.ej. 'catering.taproom').
        Se pueden combinar varios: ['restaurantes', 'bares'].

    radius_meters : int, opcional
        Radio de búsqueda en metros alrededor de `location`. Por defecto 2000.
        Para búsquedas en toda una ciudad, usar valores más altos (5000-10000).

    limit : int, opcional
        Número máximo de resultados a devolver (máx. 500). Por defecto 10.

    conditions : list[str] | None, opcional
        Filtros adicionales de Geoapify. Ejemplos:
          - 'wheelchair.yes'       → accesible en silla de ruedas
          - 'internet_access.free' → WiFi gratuito
          - 'vegetarian'           → sirven comida vegetariana
          - 'no_fee'               → entrada gratuita
        Ver VALID_CONDITIONS para el listado completo.

    lang : str, opcional
        Idioma de los nombres y direcciones en la respuesta (ISO 639-1).
        Por defecto 'es' (español).

    Devuelve
    --------
    list[dict]
        Lista de lugares encontrados, ordenados por cercanía, cada uno con:
          - name            : nombre del lugar
          - categories      : lista de categorías Geoapify
          - address         : dirección formateada
          - address_line1   : primera línea de dirección
          - address_line2   : segunda línea de dirección
          - distance_meters : distancia en metros desde `location`
          - lat, lon        : coordenadas del lugar
          - place_id        : ID único de Geoapify (para Place Details API)

    Lanza
    -----
    InvalidInterestTypeError  : algún interest_type no es reconocido.
    InvalidConditionError     : alguna condition no está en VALID_CONDITIONS.
    LocationNotFoundError     : no se pudieron obtener coordenadas para `location`.
    PlacesAPIError            : error de red o respuesta inesperada de la API.
    ValueError                : parámetros fuera de rango.
    """

    # ── Validaciones básicas ───────────────────────────────────────────────────
    if not location or not location.strip():
        raise ValueError("El parámetro 'location' no puede estar vacío.")

    if not interest_types:
        raise ValueError("Debes indicar al menos un tipo de interés en 'interest_types'.")

    if not (1 <= limit <= 500):
        raise ValueError("El parámetro 'limit' debe estar entre 1 y 500.")

    if radius_meters <= 0:
        raise ValueError("El parámetro 'radius_meters' debe ser un valor positivo.")

    # ── Validar conditions ─────────────────────────────────────────────────────
    if conditions:
        invalid_conditions = [c for c in conditions if c not in VALID_CONDITIONS]
        if invalid_conditions:
            raise InvalidConditionError(
                f"Conditions no reconocidas: {invalid_conditions}. "
                f"Válidas: {sorted(VALID_CONDITIONS)}"
            )

    # ── Resolver coordenadas ───────────────────────────────────────────────────
    lat, lon = _resolve_coordinates(location)

    # ── Construir categorías ───────────────────────────────────────────────────
    categories_str = _build_categories(interest_types)

    # ── Construir parámetros de la request ────────────────────────────────────
    params: dict[str, Any] = {
        "categories": categories_str,
        "filter":     f"circle:{lon},{lat},{radius_meters}",
        "bias":       f"proximity:{lon},{lat}",
        "limit":      limit,
        "lang":       lang,
        "apiKey":     GEOAPIFY_API_KEY,
    }

    if conditions:
        params["conditions"] = ",".join(conditions)

    # ── Llamar a la API ────────────────────────────────────────────────────────
    try:
        response = httpx.get(PLACES_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise PlacesAPIError(
            f"Error HTTP de la Places API: {exc.response.status_code} {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:
        raise PlacesAPIError(f"Error de red al llamar a la Places API: {exc}") from exc

    # ── Parsear respuesta GeoJSON ──────────────────────────────────────────────
    data = response.json()
    features = data.get("features", [])

<<<<<<< HEAD
    return [_parse_feature(f) for f in features]

if __name__ == "__main__":
    # Ejemplo de uso rápido
    try:
        results = search_places_of_interest(
            location="Madrid, España",
            interest_types=["museos", "parques"],
            radius_meters=3000,
            limit=5,
            conditions=["wheelchair.yes", "internet_access.free"],
        )
        for place in results:
            print(f"{place['name']} - {place['address']} (Distancia: {place['distance_meters']} m)")
    except Exception as e:
        print(f"Error: {e}")
=======
    return [_parse_feature(f) for f in features]
>>>>>>> 82cb8607f81b80519d927374c509e97da2414c42
