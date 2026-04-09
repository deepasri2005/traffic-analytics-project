import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def clean_data(input_path="data/traffic_data.csv", output_path="data/cleaned_traffic_data.csv"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found.")
        
    df = pd.read_csv(input_path)
    
    # Handle missing values
    df = df.ffill()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Correct data types
    df['date'] = pd.to_datetime(df['date'])
    
    # Detect outliers using IQR for vehicle_count and cap them
    Q1 = df['vehicle_count'].quantile(0.25)
    Q3 = df['vehicle_count'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['vehicle_count'] = df['vehicle_count'].clip(lower=lower_bound, upper=upper_bound)
    
    # Normalize numeric fields
    scaler = MinMaxScaler()
    df['vehicle_count_norm'] = scaler.fit_transform(df[['vehicle_count']])
    df['avg_speed_norm'] = scaler.fit_transform(df[['avg_speed']])
    df['visibility_norm'] = scaler.fit_transform(df[['visibility']])
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    return df

if __name__ == "__main__":
    clean_data()
