def suggest_food_web(city: str) -> str:
    """Returns a TasteAtlas URL for the given city."""
    return f"https://www.tasteatlas.com/{city.lower()}"
