import requests
import json
from .config import settings


def get_weather(location: str) -> str:
    """
    Fetch current weather data using WeatherAPI.

    Args:
        location (str): City name, zip code, or coordinates (latitude,longitude).
        api_key (str): Your WeatherAPI API key.

    Returns:
        str: A formatted string containing the weather data.
    """
    # Base URL for WeatherAPI
    url = f"http://api.weatherapi.com/v1/current.json?key={settings.WEATHER_API_KEY}&q={location}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        data = response.json()

        # Extract relevant weather details
        location_name = data["location"]["name"]
        region = data["location"]["region"]
        country = data["location"]["country"]
        localtime = data["location"]["localtime"]
        temperature = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]
        wind_speed = data["current"]["wind_kph"]
        humidity = data["current"]["humidity"]
        feels_like = data["current"]["feelslike_c"]

        return (
            f"Weather in {location_name}, {region}, {country}:\n"
            f"Time: {localtime}\n"
            f"Condition: {condition}\n"
            f"Temperature: {temperature}°C (Feels like {feels_like}°C)\n"
            f"Humidity: {humidity}%\n"
            f"Wind Speed: {wind_speed} km/h\n"
        )
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except Exception as err:
        return f"An error occurred: {err}"
