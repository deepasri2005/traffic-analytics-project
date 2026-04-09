import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def detect_hotspots(data_path="data/cleaned_traffic_data.csv", output_path="data/hotspots.csv", model_path="data/kmeans_model.pkl"):
    df = pd.read_csv(data_path)
    
    # Filter only accidents to find hotspots
    acc_df = df[df['accident_occurred'] == 1].copy()
    if len(acc_df) < 10:
        print("Not enough accident data for clustering.")
        return
        
    le_zone = LabelEncoder()
    le_road = LabelEncoder()
    acc_df['zone_encoded'] = le_zone.fit_transform(acc_df['city_zone'])
    acc_df['road_encoded'] = le_road.fit_transform(acc_df['road_type'])
    
    features = ['zone_encoded', 'road_encoded', 'vehicle_count_norm']
    X = acc_df[features]
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    acc_df['hotspot_cluster'] = kmeans.fit_predict(X)
    
    acc_df.to_csv(output_path, index=False)
    
    with open(model_path, "wb") as f:
        pickle.dump(kmeans, f)
        
    print(f"Hotspot detection completed. Clusters saved to {output_path}")
    return acc_df

if __name__ == "__main__":
    detect_hotspots()
