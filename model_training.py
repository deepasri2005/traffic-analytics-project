import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import json
import os
import warnings

warnings.filterwarnings("ignore")

def train_model(data_path="data/cleaned_traffic_data.csv", model_dir="data"):
    df = pd.read_csv(data_path)
    
    features = ['vehicle_count', 'avg_speed', 'weather', 'road_condition', 'visibility', 'traffic_signal']
    target = 'accident_occurred'
    
    X = df[features]
    y = df[target]
    
    X = pd.get_dummies(X, columns=['weather', 'road_condition', 'traffic_signal'], drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec
    }
    
    print(f"Model Training Results: Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    
    with open(os.path.join(model_dir, "rf_model.pkl"), "wb") as f:
        pickle.dump(model, f)
        
    with open(os.path.join(model_dir, "model_columns.pkl"), "wb") as f:
        pickle.dump(X.columns.tolist(), f)
        
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
        
    print("Model and metrics saved.")

if __name__ == "__main__":
    train_model()
