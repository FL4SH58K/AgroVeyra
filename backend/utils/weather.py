"""Utility to fetch weather data and compute disease spread risk for AgroVeyra."""

import os

import requests


def get_weather_risk(lat: float, lon: float, disease: str) -> dict:
    """Fetch weather data from OpenWeatherMap and compute disease spread risk.

    Args:
        lat: Latitude coordinate.
        lon: Longitude coordinate.
        disease: Disease class name (e.g., "Early_blight").

    Returns:
        dict with keys:
            - spread_risk (str): "HIGH", "MODERATE", or "LOW".
            - message (str): Human readable advisory message.
            - avg_humidity (float): Average humidity percentage.
            - avg_temp (float): Average temperature in Celsius.
    """
    try:
        api_key = os.getenv("OPENWEATHERMAP_API_KEY", "")
        if not api_key:
            raise ValueError("OPENWEATHERMAP_API_KEY not found in environment")

        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric",
            "cnt": 8,
        }
        print(f"[weather.py] Fetching weather for lat={lat}, lon={lon}")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        humidity_sum = 0.0
        temp_sum = 0.0
        count = 0
        rain_days = 0

        for item in data.get("list", []):
            main = item.get("main", {})
            humidity = main.get("humidity", 0)
            temp = main.get("temp", 0)
            weather = item.get("weather", [{}])[0].get("main", "")
            humidity_sum += humidity
            temp_sum += temp
            count += 1
            if "Rain" in weather:
                rain_days += 1

        avg_humidity = round(humidity_sum / count, 1) if count > 0 else 0.0
        avg_temp = round(temp_sum / count, 1) if count > 0 else 0.0

        # Determine risk
        if "blight" in disease.lower() or "mold" in disease.lower() or "rust" in disease.lower():
            if avg_humidity > 75 and rain_days >= 2:
                spread_risk = "HIGH"
                message = f"Rain expected in {rain_days} days will accelerate {disease.replace('_', ' ')} spread"
            elif avg_humidity > 60:
                spread_risk = "MODERATE"
                message = f"Moderate humidity may support {disease.replace('_', ' ')} development"
            else:
                spread_risk = "LOW"
                message = f"Conditions are not favorable for {disease.replace('_', ' ')} spread"
        else:
            if avg_humidity > 80:
                spread_risk = "HIGH"
                message = f"High humidity creates favorable conditions for {disease.replace('_', ' ')}"
            elif avg_humidity > 60:
                spread_risk = "MODERATE"
                message = f"Monitor {disease.replace('_', ' ')} progression under current conditions"
            else:
                spread_risk = "LOW"
                message = f"Low risk environment for {disease.replace('_', ' ')}"

        print(f"[weather.py] Risk: {spread_risk}, Humidity: {avg_humidity}%, Temp: {avg_temp}°C")
        return {
            "spread_risk": spread_risk,
            "message": message,
            "avg_humidity": avg_humidity,
            "avg_temp": avg_temp,
        }
    except requests.RequestException as e:
        print(f"[weather.py] API request failed: {e}")
        return {"spread_risk": "UNKNOWN", "message": f"Weather data unavailable: {e}", "avg_humidity": 0.0, "avg_temp": 0.0}
    except Exception as e:
        print(f"[weather.py] Error: {e}")
        return {"spread_risk": "UNKNOWN", "message": f"Weather analysis failed: {e}", "avg_humidity": 0.0, "avg_temp": 0.0}


