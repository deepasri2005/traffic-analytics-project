import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_data(file_path="data/traffic_data.csv"):
    np.random.seed(42)
    random.seed(42)
    
    zones = ['North', 'South', 'East', 'West', 'Central']
    road_types = ['Highway', 'Arterial', 'Residential', 'Crossroad']
    weathers = ['Clear', 'Rain', 'Fog', 'Snow']
    signals = ['Green', 'Red', 'Yellow']
    road_conditions = ['Good', 'Fair', 'Poor']
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(3000):
        dt = start_date + timedelta(days=random.randint(0, 365), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        date_str = dt.strftime('%Y-%m-%d')
        time_str = dt.strftime('%H:%M:%S')
        day_of_week = dt.strftime('%A')
        
        zone = random.choice(zones)
        road = random.choice(road_types)
        weather = random.choice(weathers)
        signal = random.choice(signals)
        condition = random.choice(road_conditions)
        
        vehicle_count = int(np.random.normal(loc=50 if road != 'Highway' else 150, scale=20))
        vehicle_count = max(5, vehicle_count)
        
        avg_speed = np.random.normal(loc=40 if road != 'Highway' else 80, scale=10)
        if weather in ['Rain', 'Fog', 'Snow']:
            avg_speed *= 0.8
        avg_speed = max(10, avg_speed)
        
        visibility = random.randint(1, 10) if weather != 'Fog' else random.randint(1, 4)
        
        acc_prob = 0.05
        if weather in ['Rain', 'Snow', 'Fog']: acc_prob += 0.05
        if condition == 'Poor': acc_prob += 0.05
        if visibility < 3: acc_prob += 0.05
        
        accident_occurred = 1 if random.random() < acc_prob else 0
        if accident_occurred:
            accident_severity = random.choice(['Minor', 'Major', 'Fatal'])
        else:
            accident_severity = 'None'
            
        data.append({
            'traffic_id': i + 1,
            'date': date_str,
            'time': time_str,
            'city_zone': zone,
            'road_type': road,
            'vehicle_count': vehicle_count,
            'avg_speed': round(avg_speed, 2),
            'weather': weather,
            'visibility': visibility,
            'traffic_signal': signal,
            'accident_occurred': accident_occurred,
            'accident_severity': accident_severity,
            'road_condition': condition,
            'day_of_week': day_of_week
        })
        
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Dataset generated at {file_path}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/traffic_data.csv'
    generate_data(path)
