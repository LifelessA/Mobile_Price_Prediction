import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
try:
    pipeline = joblib.load('price_prediction_pipeline.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run the Jupyter Notebook first to train and save the model.")
    st.stop()

# Define the price range mapping for labels and estimated INR
price_map = {
    0: {'label': 'Low Cost', 'inr': '< ₹10,000'},
    1: {'label': 'Medium Cost', 'inr': '₹10,000 - ₹20,000'},
    2: {'label': 'High Cost', 'inr': '₹20,000 - ₹40,000'},
    3: {'label': 'Very High Cost', 'inr': '> ₹40,000'}
}

# Set up the Streamlit page
st.set_page_config(page_title="Mobile Phone Price Predictor", layout="wide")

# App title and description
st.title("📱 Mobile Phone Price Range Predictor")
st.markdown("Enter the mobile phone's specifications below to get an estimated price range.")

# --- User Input Section ---
st.header("Enter Device Specifications")

# Create columns for a structured layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Core Features")
    ram = st.slider("RAM (MB)", min_value=250, max_value=4000, value=2000, step=128)
    int_memory = st.slider("Internal Memory (GB)", min_value=2, max_value=64, value=32, step=2)
    battery_power = st.slider("Battery Power (mAh)", min_value=500, max_value=2000, value=1250, step=50)
    mobile_wt = st.slider("Mobile Weight (gm)", min_value=80, max_value=200, value=140, step=1)
    
with col2:
    st.subheader("Camera & Display")
    px_height = st.slider("Pixel Resolution Height", min_value=0, max_value=2000, value=650, step=10)
    px_width = st.slider("Pixel Resolution Width", min_value=500, max_value=2000, value=1250, step=10)
    pc = st.slider("Primary Camera (MP)", min_value=0, max_value=21, value=10, step=1)
    fc = st.slider("Front Camera (MP)", min_value=0, max_value=20, value=5, step=1)

with col3:
    st.subheader("Processor & Connectivity")
    n_cores = st.selectbox("Number of Cores", [1, 2, 3, 4, 5, 6, 7, 8], index=3)
    clock_speed = st.slider("Clock Speed (GHz)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
    four_g = st.selectbox("4G Support", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    three_g = st.selectbox("3G Support", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Collect all other features with default values
other_features = {
    'blue': 1, 'dual_sim': 1, 'm_dep': 0.5, 'sc_h': 12, 'sc_w': 7,
    'talk_time': 10, 'touch_screen': 1, 'wifi': 1
}

# Prediction Logic
if st.button("Predict Price Range", use_container_width=True):
    # Create a DataFrame from the user inputs
    input_data = {
        'battery_power': battery_power, 'blue': other_features['blue'], 'clock_speed': clock_speed,
        'dual_sim': other_features['dual_sim'], 'fc': fc, 'four_g': four_g, 'int_memory': int_memory,
        'm_dep': other_features['m_dep'], 'mobile_wt': mobile_wt, 'n_cores': n_cores, 'pc': pc,
        'px_height': px_height, 'px_width': px_width, 'ram': ram, 'sc_h': other_features['sc_h'],
        'sc_w': other_features['sc_w'], 'talk_time': other_features['talk_time'], 'three_g': three_g,
        'touch_screen': other_features['touch_screen'], 'wifi': other_features['wifi']
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    try:
        prediction_code = pipeline.predict(input_df)[0]
        prediction_info = price_map[prediction_code]
        
        # Display the result
        st.markdown(f"""
        <div style="background-color:#F0F2F6; padding: 20px; border-radius: 10px;">
            <h3 style='text-align: center; color: #1E88E5;'>Prediction Result</h3>
            <p style='text-align: center; font-size: 20px;'>
                Predicted Price Category: <strong>{prediction_info['label']}</strong>
            </p>
            <p style='text-align: center; font-size: 24px; color: #4CAF50;'>
                Estimated Price in India: <strong>{prediction_info['inr']}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        if prediction_code == 3:
            st.balloons()
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")