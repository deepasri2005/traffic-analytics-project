import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import pickle
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(page_title="Smart City Traffic Analytics", page_icon="🚥", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Premium Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    .stSidebar {
        background-color: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-weight: 700;
        text-shadow: 0px 0px 20px rgba(56, 189, 248, 0.4);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(56, 189, 248, 0.4);
        border: 1px solid rgba(56, 189, 248, 0.5);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #38bdf8;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 1rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div.stButton > button {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        color: white;
        border: none;
        padding: 10px 25px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.4);
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.6);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚦 Smart City Traffic Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.2rem; color: #cbd5e1; margin-bottom: 2rem;'>AI-powered congestion & accident prediction platform.</p>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if os.path.exists('data/cleaned_traffic_data.csv'):
        df = pd.read_csv('data/cleaned_traffic_data.csv')
        if 'hour' not in df.columns:
            df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour
        return df
    return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Dataset not found. Please run the data pipeline processes first.")
    st.stop()

# Sidebar Navigation
with st.sidebar:
    st.markdown("<h3>Navigation Menu</h3>", unsafe_allow_html=True)
    st.markdown("---")
    selected_section = st.radio("", ["📊 Overview", "📈 Traffic Trends", "💥 Accident Analysis", "🎯 Hotspot Zones", "🤖 AI Risk Predictor"])
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>Smart City V1.0</p>", unsafe_allow_html=True)

if selected_section == "📊 Overview":
    st.markdown("<h2>Project Overview</h2>", unsafe_allow_html=True)
    
    # Custom Metric Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Records</div>
                <div class="metric-value">{len(df):,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        accidents = df['accident_occurred'].sum()
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Accidents Recorded</div>
                <div class="metric-value">{accidents:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_speed = df['avg_speed'].mean()
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg City Speed</div>
                <div class="metric-value">{avg_speed:.1f} km/h</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>Dataset Preview</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(15), use_container_width=True)

elif selected_section == "📈 Traffic Trends":
    st.markdown("<h2>Traffic Flow & Congestion Trends</h2>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Zone Distribution", "Hourly Patterns"])
    
    with tab1:
        vol_df = df.groupby('city_zone')['vehicle_count'].sum().reset_index()
        fig1 = px.bar(
            vol_df, x='city_zone', y='vehicle_count', 
            color='vehicle_count', color_continuous_scale="Blues",
            title="Total Traffic Volume by City Zone",
            template="plotly_dark"
        )
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, use_container_width=True)
        
    with tab2:
        hourly_df = df.groupby('hour')['vehicle_count'].mean().reset_index()
        fig2 = px.line(
            hourly_df, x='hour', y='vehicle_count', markers=True,
            title="Average Hourly Traffic Volume",
            template="plotly_dark", line_shape='spline'
        )
        fig2.update_traces(line=dict(color='#818cf8', width=4), marker=dict(size=8, color='#38bdf8'))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

elif selected_section == "💥 Accident Analysis":
    st.markdown("<h2>Accident Analytics</h2>", unsafe_allow_html=True)
    acc_df = df[df['accident_occurred'] == 1]
    
    col1, col2 = st.columns(2)
    with col1:
        fig3 = px.pie(
            acc_df, names='road_type', title="Accidents by Road Type",
            hole=0.4, template="plotly_dark", color_discrete_sequence=px.colors.sequential.Tealgrn
        )
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)
        
    with col2:
        fig4 = px.pie(
            acc_df, names='weather', title="Accidents by Weather Impact",
            hole=0.4, template="plotly_dark", color_discrete_sequence=px.colors.sequential.Sunset
        )
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

elif selected_section == "🎯 Hotspot Zones":
    st.markdown("<h2>Accident Hotspots (K-Means Clustering)</h2>", unsafe_allow_html=True)
    
    if os.path.exists("data/hotspots.csv"):
        hotspots = pd.read_csv("data/hotspots.csv")
        cluster_counts = hotspots['hotspot_cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Number of Accidents']
        cluster_counts['Cluster'] = cluster_counts['Cluster'].astype(str)
        
        fig5 = px.bar(
            cluster_counts, x='Cluster', y='Number of Accidents',
            color='Cluster', title="Accident Density per Identified Hotspot Cluster",
            template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig5, use_container_width=True)
        
        st.markdown("<h3>High-Risk Hotspot Data</h3>", unsafe_allow_html=True)
        st.dataframe(hotspots[['city_zone', 'road_type', 'vehicle_count', 'weather', 'hotspot_cluster']].head(15), use_container_width=True)
    else:
        st.warning("Hotspot data not found. Please run the hotspot detection script.")

elif selected_section == "🤖 AI Risk Predictor":
    st.markdown("<h2>Real-time AI Accident Risk Predictor</h2>", unsafe_allow_html=True)
    st.markdown("Use our deployed **Random Forest Machine Learning** model to simulate live conditions and gauge danger levels.")
    
    if os.path.exists("data/rf_model.pkl") and os.path.exists("data/model_columns.pkl"):
        # Display Metrics Beautifully
        if os.path.exists("data/metrics.json"):
            with open("data/metrics.json", "r") as f:
                metrics = json.load(f)
            
            st.markdown("### 🏆 Algorithm Confidence Metrics")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Accuracy</div><div class='metric-value'>{metrics.get('accuracy',0)*100:.1f}%</div></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Precision</div><div class='metric-value'>{metrics.get('precision',0)*100:.1f}%</div></div>", unsafe_allow_html=True)
            with m3:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Recall</div><div class='metric-value'>{metrics.get('recall',0)*100:.1f}%</div></div>", unsafe_allow_html=True)
                
        st.markdown("<br><hr>", unsafe_allow_html=True)
        
        # User Form
        with st.form("prediction_form"):
            st.markdown("### 🚦 Input Live Conditions")
            c1, c2, c3 = st.columns(3)
            with c1:
                vc = st.number_input("Vehicle Count", 1, 500, 50)
                speed = st.number_input("Average Speed (km/h)", 0, 150, 40)
            with c2:
                weather = st.selectbox("Weather", ['Clear', 'Rain', 'Fog', 'Snow'])
                road_cond = st.selectbox("Road Condition", ['Good', 'Fair', 'Poor'])
            with c3:
                visibility = st.slider("Visibility (1-10 km)", 1, 10, 5)
                signal = st.selectbox("Traffic Signal", ['Green', 'Red', 'Yellow'])
            
            submit_btn = st.form_submit_button("Predict Risk Level")
            
        if submit_btn:
            with st.spinner("Analyzing parameters..."):
                with open("data/rf_model.pkl", "rb") as f:
                    model = pickle.load(f)
                with open("data/model_columns.pkl", "rb") as f:
                    model_cols = pickle.load(f)
                    
                input_data = {'vehicle_count': vc, 'avg_speed': speed, 'visibility': visibility}
                input_df = pd.DataFrame([input_data])
                
                input_df[f'weather_{weather}'] = 1
                input_df[f'road_condition_{road_cond}'] = 1
                input_df[f'traffic_signal_{signal}'] = 1
                
                for c in model_cols:
                    if c not in input_df.columns:
                        input_df[c] = 0
                input_df = input_df[model_cols]
                
                pred = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0]
                
                st.markdown("<br>", unsafe_allow_html=True)
                if pred == 1:
                    chance = proba[1] * 100
                    st.error(f"🚨 **DANGER: High Risk of Accident!** The AI calculates a {chance:.1f}% probability of an incident under these conditions.")
                else:
                    chance = proba[0] * 100
                    st.success(f"✅ **SAFE: Low Risk of Accident.** The AI calculates a {chance:.1f}% probability of safe travel.")
    else:
        st.warning("Model files not found. Please run the model training script.")
