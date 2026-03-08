import requests
from datetime import datetime, timezone

API_KEY = "YOUR_API_KEY"
LAT = 51.5074
LON = -0.1278

hours = range(1, 9)  # 1:30 to 8:30

results = []
for hour in hours:
    dt = datetime(2026, 2, 26, hour, 30, tzinfo=timezone.utc)  # London in Feb = UTC+0
    timestamp = int(dt.timestamp())
    
    url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    params = {
        "lat": LAT,
        "lon": LON,
        "dt": timestamp,
        "appid": API_KEY,
        "units": "metric"
        # no lang param = English by default
    }
    
    resp = requests.get(url, params=params)
    data = resp.json()
    
    hourly = data.get("data", [{}])[0]
    results.append({
        "time": dt.strftime("%H:%M"),
        "temp_c": hourly.get("temp"),
        "feels_like_c": hourly.get("feels_like"),
        "humidity_%": hourly.get("humidity"),
        "weather": hourly.get("weather", [{}])[0].get("description"),
        "wind_speed_m/s": hourly.get("wind_speed"),
        "pressure_hPa": hourly.get("pressure"),
        "clouds_%": hourly.get("clouds"),
    })
    print(f"{dt.strftime('%H:%M')} - {hourly.get('temp')}°C, {hourly.get('weather', [{}])[0].get('description')}")

import pandas as pd
df = pd.DataFrame(results)
df.to_csv("london_weather_feb26.csv", index=False)
print(df)