import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style="whitegrid")

# Sample data for both cities
def load_default_data():
    return pd.DataFrame({
        "City": ["Pretoria", "Johannesburg"] * 4,
        "Mode": ["Minibus Taxi", "Uber Go", "Gautrain", "MetroBus"] * 2,
        "Avg Cost per Trip (ZAR)": [13, 312/15, 92, 9, 13, 312/15, 92, 9],
        "Cost per km (ZAR/km)": [13/15, 312/56, 92/36, 9/10] * 2,
        "Avg Monthly Cost (ZAR)": [572, 312*22, 3254, 352] * 2,
        "Avg Travel Time (min)": [63, 50, 38, 84] * 2
    })

# Streamlit App
st.set_page_config("Transport Cost Dashboard", layout="wide")
st.title("🚍 Public Transport Cost Dashboard – Gauteng (Pretoria & Johannesburg)")

# Load data
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV file", type=["csv", "xlsx"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("Custom data loaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = load_default_data()
else:
    df = load_default_data()

# Filters
st.sidebar.header("Filter Options")
selected_city = st.sidebar.selectbox("Select City", df["City"].unique())
filtered_df = df[df["City"] == selected_city]

sort_by = st.sidebar.selectbox("Sort Charts By", filtered_df.columns[2:])

# Trip Simulator
st.sidebar.header("Simulate Your Commute Cost")
trip_distance = st.sidebar.slider("Trip Distance (one way, in km)", 1, 60, 15)
days_per_month = st.sidebar.slider("Commuting Days per Month", 10, 30, 22)

# Simulate monthly cost based on trip distance
simulated_df = filtered_df.copy()
simulated_df["Simulated Monthly Cost (ZAR)"] = simulated_df["Cost per km (ZAR/km)"] * trip_distance * 2 * days_per_month

# Plotting function
def make_barplot(df, x, title, xlabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=x, y="Mode", data=df.sort_values(by=x, ascending=False), ax=ax, palette="crest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Transport Mode")
    st.pyplot(fig)

# Visualizations
st.subheader(f"Transport Costs in {selected_city}")
make_barplot(filtered_df, "Avg Cost per Trip (ZAR)", "Average Cost per Trip", "ZAR")
make_barplot(filtered_df, "Avg Monthly Cost (ZAR)", "Estimated Monthly Cost", "ZAR/month")
make_barplot(filtered_df, "Cost per km (ZAR/km)", "Cost per Kilometer", "ZAR/km")
make_barplot(filtered_df, "Avg Travel Time (min)", "Travel Time by Mode", "Minutes")

st.subheader("🧮 Simulated Monthly Costs Based on Your Trip")
make_barplot(simulated_df, "Simulated Monthly Cost (ZAR)", f"Simulated Monthly Cost – {trip_distance} km per trip × {days_per_month} days", "ZAR/month")

# Show Raw Data
with st.expander("📊 View Raw Data"):
    st.dataframe(filtered_df)

# Export to CSV
csv = simulated_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Simulated Data as CSV", data=csv, file_name="simulated_transport_costs.csv", mime="text/csv")
