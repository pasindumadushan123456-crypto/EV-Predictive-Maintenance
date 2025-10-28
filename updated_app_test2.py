import os
import time
import random
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# --------------------------
# Paths
# -------------------------
# --------------------------
# Paths
# --------------------------
# Use the current directory ('.') which is the root of your repo
# where the script is running.
MODEL_DIR = "."  

# Or, simplify it completely:
RUL_MODEL_PATH = "RUL_pipeline.pkl"
FAILURE_MODEL_PATH = "Failure_Probability_pipeline.pkl"

# --------------------------
# Load Models Safely
# --------------------------
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        model = joblib.load(path)
        return model
    else:
        st.sidebar.error(f"❌ Model not found: {path}")
        return None

rul_model = load_model(RUL_MODEL_PATH)
failure_model = load_model(FAILURE_MODEL_PATH)

# --------------------------
# Streamlit Config
# --------------------------
st.set_page_config(page_title="EV Predictive Maintenance", page_icon="🚗", layout="wide")

st.title("🚗 EV Predictive Maintenance Dashboard")
st.markdown("### Predict vehicle health, maintenance, and warranty eligibility with live sensor data integration.")

# --------------------------
# Sidebar Manual Input
# --------------------------
st.sidebar.header("⚙️ Manual Vehicle Data Input")

soc = st.sidebar.slider("🔋 Battery SoC (%)", 0, 100, 80)
soh = st.sidebar.slider("📉 Battery SoH (%)", 0, 100, 95)
battery_voltage = st.sidebar.number_input("🔋 Battery Voltage (V)", 0, 500, 400)
battery_current = st.sidebar.number_input("🔌 Battery Current (A)", -500, 500, 0)
battery_temp = st.sidebar.slider("🌡️ Battery Temperature (°C)", 0, 100, 30)
charge_cycles = st.sidebar.number_input("🔄 Charge Cycles", 0, 5000, 150)

motor_temp = st.sidebar.slider("⚡ Motor Temperature (°C)", 0, 200, 60)
motor_vibration = st.sidebar.number_input("🛠️ Motor Vibration (units)", 0.0, 10.0, 0.5)
motor_torque = st.sidebar.number_input("⚡ Motor Torque (Nm)", 0.0, 500.0, 100.0)
motor_rpm = st.sidebar.number_input("⚙️ Motor RPM", 0, 10000, 2000)
power_consumption = st.sidebar.number_input("⚡ Power Consumption (kW)", 0.0, 500.0, 50.0)

brake_wear = st.sidebar.slider("🛑 Brake Pad Wear (%)", 0, 100, 20)
brake_pressure = st.sidebar.number_input("🛑 Brake Pressure", 0.0, 100.0, 40.0)
reg_brake_eff = st.sidebar.number_input("🔁 Regenerative Brake Efficiency", 0.0, 1.0, 0.8)

tire_pressure = st.sidebar.slider("🛞 Tire Pressure (PSI)", 20, 50, 32)
tire_temp = st.sidebar.slider("🌡️ Tire Temperature (°C)", 0, 100, 30)
suspension_load = st.sidebar.number_input("🛞 Suspension Load", 0.0, 500.0, 100.0)

ambient_temp = st.sidebar.slider("🌡️ Ambient Temperature (°C)", -10, 50, 25)
ambient_humidity = st.sidebar.slider("💧 Ambient Humidity (%)", 0, 100, 50)
load_weight = st.sidebar.number_input("🏋️ Load Weight (kg)", 0.0, 2000.0, 500.0)
driving_speed = st.sidebar.number_input("🚗 Driving Speed (km/h)", 0.0, 200.0, 60.0)
distance_traveled = st.sidebar.number_input("📍 Distance Traveled (km)", 0, 500000, 20000)
idle_time = st.sidebar.number_input("🕒 Idle Time (minutes)", 0, 5000, 60)
route_roughness = st.sidebar.number_input("🛣️ Route Roughness", 0.0, 10.0, 2.0)

# --------------------------
# Feature List
# --------------------------
features = [
    "SoC", "SoH", "Battery_Voltage", "Battery_Current", "Battery_Temperature", "Charge_Cycles",
    "Motor_Temperature", "Motor_Vibration", "Motor_Torque", "Motor_RPM", "Power_Consumption",
    "Brake_Pad_Wear", "Brake_Pressure", "Reg_Brake_Efficiency", "Tire_Pressure", "Tire_Temperature",
    "Suspension_Load", "Ambient_Temperature", "Ambient_Humidity", "Load_Weight", "Driving_Speed",
    "Distance_Traveled", "Idle_Time", "Route_Roughness"
]

manual_input = pd.DataFrame([[soc/100, soh/100, battery_voltage, battery_current, battery_temp, charge_cycles,
    motor_temp, motor_vibration, motor_torque, motor_rpm, power_consumption,
    brake_wear/100, brake_pressure, reg_brake_eff, tire_pressure, tire_temp,
    suspension_load, ambient_temp, ambient_humidity, load_weight, driving_speed,
    distance_traveled, idle_time, route_roughness]], columns=features)

# --------------------------
# Tabs for Navigation
# --------------------------
tab1, tab2, tab3 = st.tabs(["📁 Data Input", "📊 Prediction & Warranty", "📡 Live Sensor Console"])

# --------------------------
# TAB 1: CSV Upload
# --------------------------
with tab1:
    st.header("📁 Upload EV Sensor Data (CSV)")
    uploaded_file = st.file_uploader("Upload CSV file with all required columns", type=["csv"])

    if uploaded_file:
        try:
            csv_data = pd.read_csv(uploaded_file)
            st.success("✅ CSV Loaded Successfully")
            st.dataframe(csv_data.head())
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")
            csv_data = None
    else:
        csv_data = None

# --------------------------
# TAB 2: Predictions
# --------------------------
with tab2:
    st.header("🔮 Predict Maintenance Needs")
    if st.button("🚀 Run Prediction"):
        if rul_model and failure_model:
            try:
                input_data = pd.concat([manual_input, csv_data], ignore_index=True) if csv_data is not None else manual_input

                rul_predictions = rul_model.predict(input_data)
                failure_predictions = failure_model.predict_proba(input_data)[:, 1]

                results = input_data.copy()
                results["RUL_days"] = rul_predictions
                results["Failure_Probability"] = failure_predictions
                results["Vehicle_Health"] = 1 - failure_predictions
                results["Warranty_Claim_Accepted"] = results.apply(
                    lambda row: "✅ Accepted" if (row["RUL_days"] < 180 or row["Failure_Probability"] > 0.5)
                    else "❌ Rejected", axis=1
                )

                st.dataframe(results[["RUL_days", "Failure_Probability", "Vehicle_Health", "Warranty_Claim_Accepted"]])

                avg_rul = rul_predictions.mean()
                avg_failure = failure_predictions.mean()

                col1, col2 = st.columns(2)
                col1.metric("🕒 Avg Remaining Useful Life (days)", f"{int(avg_rul)}")
                col2.metric("⚠️ Avg Failure Probability", f"{avg_failure*100:.1f}%")

                st.markdown("### 🛠️ Recommended Maintenance")
                st.write(f"- Next **battery health check**: {int(avg_rul/3)} days")
                st.write(f"- **Brake service**: {int(avg_rul/2)} days")
                st.write(f"- **Tire rotation**: {int(distance_traveled/1000)} km")
                st.write(f"- **Cooling system inspection**: {int(avg_rul/4)} days")
                st.write(f"- **Motor vibration check**: {int(avg_rul/5)} days")

                fig = px.bar(
                    x=["Battery", "Brakes", "Tires", "Motor", "Cooling"],
                    y=[avg_rul/3, avg_rul/2, distance_traveled/1000, avg_rul/4, avg_rul/5],
                    labels={"x": "Component", "y": "Days to Service"},
                    color=["Battery", "Brakes", "Tires", "Motor", "Cooling"],
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

# --------------------------
# TAB 3: Live Sensor Console
# --------------------------
with tab3:
    st.header("📡 Live Sensor Data Console")

    st.write("Monitor live EV sensor readings in real time. (Currently simulating values.)")

    placeholder = st.empty()
    start = st.checkbox("▶️ Start Live Monitoring")

    while start:
        sensor_data = {
            "Battery_Voltage": random.uniform(380, 420),
            "Battery_Current": random.uniform(-50, 50),
            "Battery_Temperature": random.uniform(25, 40),
            "Motor_Temperature": random.uniform(50, 90),
            "Motor_Vibration": random.uniform(0.1, 2.0),
            "Brake_Pad_Wear": random.uniform(0.1, 0.9),
            "Tire_Pressure": random.uniform(28, 36),
            "Ambient_Temperature": random.uniform(20, 35),
        }

        live_df = pd.DataFrame(sensor_data, index=[0])
        with placeholder.container():
            st.metric("🔋 Battery Voltage (V)", f"{sensor_data['Battery_Voltage']:.2f}")
            st.metric("⚡ Battery Current (A)", f"{sensor_data['Battery_Current']:.2f}")
            st.metric("🌡️ Battery Temperature (°C)", f"{sensor_data['Battery_Temperature']:.2f}")
            st.metric("⚙️ Motor Temperature (°C)", f"{sensor_data['Motor_Temperature']:.2f}")
            st.metric("🛠️ Motor Vibration", f"{sensor_data['Motor_Vibration']:.2f}")
            st.metric("🛑 Brake Pad Wear (%)", f"{sensor_data['Brake_Pad_Wear']*100:.1f}")
            st.metric("🛞 Tire Pressure (PSI)", f"{sensor_data['Tire_Pressure']:.1f}")
            st.metric("🌡️ Ambient Temp (°C)", f"{sensor_data['Ambient_Temperature']:.1f}")

            chart = px.line(live_df.T, title="📈 Live Sensor Trends", markers=True)
            st.plotly_chart(chart, use_container_width=True)

        time.sleep(2)


