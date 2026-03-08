import requests
import pandas as pd
from datetime import datetime
import time

API_KEY = "9bbe762eaa18f3c0533bd9851e6cefa0"

def get_weather_realtime(lat=51.5074, lon=-0.1278):
    url = "https://api.openweathermap.org/data/2.5/weather"
    
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"  # 摄氏度
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "precipitation": data.get("rain", {}).get("1h", 0.0),  # 过去1小时降雨mm
        "wind_speed": data["wind"]["speed"]
    }])
    
    return df

def collect_and_save(output_file="weather_log.csv", interval_minutes=15):
    while True:
        df = get_weather_realtime()
        df.to_csv(output_file, mode="a", header=not pd.io.common.file_exists(output_file), index=False)
        print(f"[{datetime.now()}] 已记录: {df['timestamp'].values[0]}")
        time.sleep(interval_minutes * 60)

collect_and_save()