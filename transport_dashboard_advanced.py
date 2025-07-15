import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from geopy.distance import geodesic

# -------------------------
# 📊 Sample data for analysis
# -------------------------
def load_default_data():
    return pd.DataFrame({
        "City": ["Pretoria", "Johannesburg"] * 4,
        "Mode": ["Minibus Taxi", "Uber Go", "Gautrain", "MetroBus"] * 2,
        "Avg Cost per Trip (ZAR)": [13, 312/15, 92, 9, 13, 312/15, 92, 9],
        "Cost per km (ZAR/km)": [13/15, 312/56, 92/36, 9/10] * 2,
        "Avg Monthly Cost (ZAR)": [572, 312*22, 3254, 352] * 2,
        "Avg Travel Time (min)": [63, 50, 38, 84] * 2
    })

# -------------------------
# 📍 Key Transport Hubs
# -------------------------
hub_locations = {
    "Pretoria Station": (-25.7461, 28.1881),
    "Hatfield Gautrain": (-25.7479, 28.2346),
    "Joburg Park Station": (-26.2010, 28.0436),
    "Sandton Gautrain": (-26.1076, 28.0567),
    "OR Tambo Airport": (-26.1392, 28.2460)
}

# -------------------------
# 🚕 Simulate Uber Cost
# -------------------------
def simulate_uber_cost(distance_km, car_type):
    base = {"Uber Go": 10, "UberX": 15}
    per_km = {"Uber Go": 6, "UberX": 9}
    return round(base[car_type] + per_km[car_type] * distance_km, 2)

# -------------------------
# 🎨 Setup Streamlit Page
# -------------------------
st.set_page_config("Gauteng Transport Dashboard", layout="wide")
st.title("🚦 Gauteng Public Transport Dashboard")

# Load default data
df = load_default_data()

# -------------------------
# 📌 Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["📊 Transport Cost Analysis", "🗺️ Interactive Map", "🚕 Uber Estimator"])

# -------------------------
# 📊 TAB 1 – Analysis
# -------------------------
with tab1:
    st.subheader("📊 Transport Cost & Time Comparison")
    city = st.selectbox("Select City", df["City"].unique())
    df_city = df[df["City"] == city]

    def make_plot(column, title, xlabel):
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(y="Mode", x=column, data=df_city, palette="Set2", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        st.pyplot(fig)

    make_plot("Avg Cost per Trip (ZAR)", "Average Cost per Trip", "ZAR")
    make_plot("Avg Monthly Cost (ZAR)", "Monthly Cost", "ZAR/month")
    make_plot("Cost per km (ZAR/km)", "Cost per km", "ZAR/km")
    make_plot("Avg Travel Time (min)", "Travel Time", "Minutes")
    st.dataframe(df_city)

# -------------------------
# 🗺️ TAB 2 – Interactive Map
# -------------------------
with tab2:
    st.subheader("🗺️ Key Transport Locations in Gauteng")

    m = folium.Map(location=[-25.85, 28.1], zoom_start=9)

    for name, coords in hub_locations.items():
        folium.Marker(location=coords, tooltip=name).add_to(m)

    folium_static(m, width=900, height=500)

# -------------------------
# 🚕 TAB 3 – Uber Estimator
# -------------------------
with tab3:
    st.subheader("🚕 Simulated Uber Cost Estimator")

    origin = st.selectbox("Start Location", list(hub_locations.keys()))
    destination = st.selectbox("Destination", list(hub_locations.keys()))
    car_type = st.radio("Select Uber Type", ["Uber Go", "UberX"])

    if origin != destination:
        coord1 = hub_locations[origin]
        coord2 = hub_locations[destination]
        distance_km = round(geodesic(coord1, coord2).km, 2)
        est_price = simulate_uber_cost(distance_km, car_type)

        st.markdown(f"**Distance**: {distance_km} km")
        st.markdown(f"**Estimated {car_type} fare**: R{est_price}")
    else:
        st.warning("Please select different start and end locations.")
