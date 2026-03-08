import requests
import pandas as pd
from datetime import datetime
import time

def get_weather_realtime(lat=51.5074, lon=-0.1278):
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "minutely_15": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "Europe/London",
        "past_minutely_15": 1
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame({
        "timestamp": data["minutely_15"]["time"],
        "temperature": data["minutely_15"]["temperature_2m"],
        "humidity": data["minutely_15"]["relative_humidity_2m"],
        "precipitation": data["minutely_15"]["precipitation"],
        "wind_speed": data["minutely_15"]["wind_speed_10m"]
    })
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def collect_and_save(output_file="weather_log.csv", interval_minutes=15):
    while True:
        df = get_weather_realtime()
        latest = df[df["timestamp"] == df["timestamp"].max()]
        latest.to_csv(output_file, mode="a", header=not pd.io.common.file_exists(output_file), index=False)
        print(f"[{datetime.now()}] 已记录: {latest['timestamp'].values[0]}")
        time.sleep(interval_minutes * 60)

collect_and_save()