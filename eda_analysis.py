import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

def perform_eda(data_path="data/cleaned_traffic_data.csv", output_dir="data/plots"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    # Traffic volume by zone
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='city_zone', y='vehicle_count', estimator=sum)
    plt.title('Traffic Volume by Zone')
    plt.savefig(f"{output_dir}/traffic_by_zone.png")
    plt.close()
    
    # Accident frequency by road type
    plt.figure(figsize=(10, 6))
    acc_df = df[df['accident_occurred'] == 1]
    sns.countplot(data=acc_df, x='road_type')
    plt.title('Accident Frequency by Road Type')
    plt.savefig(f"{output_dir}/accidents_by_road_type.png")
    plt.close()
    
    # Accidents by weather condition
    plt.figure(figsize=(10, 6))
    sns.countplot(data=acc_df, x='weather')
    plt.title('Accident Frequency by Weather Condition')
    plt.savefig(f"{output_dir}/accidents_by_weather.png")
    plt.close()
    
    # Hourly traffic patterns
    df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='hour', y='vehicle_count', estimator='mean')
    plt.title('Hourly Traffic Patterns')
    plt.savefig(f"{output_dir}/hourly_traffic.png")
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns='traffic_id', errors='ignore')
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()
    
    print(f"EDA plots generated in {output_dir}.")

if __name__ == "__main__":
    perform_eda()
