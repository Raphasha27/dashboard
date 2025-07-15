import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style="whitegrid")

# Sample transport cost data (ZAR = South African Rand)
data = {
    "Mode": ["Minibus Taxi", "Uber Go", "UberX", "Gautrain", "Metrorail", "MetroBus", "Rea Vaya", "Walking"],
    "Avg Cost per Trip (ZAR)": [13, 312/15, 420/15, 92, 13, 9, 10.5, 0],
    "Cost per km (ZAR/km)": [13/15, 312/56, 420/56, 92/36, 13/15, 9/10, 10.5/12, 0],
    "Avg Monthly Cost (ZAR)": [572, 312*22, 420*22, 3254, 581, 352, 420, 0],
    "Avg Travel Time (min)": [63, 50, 48, 38, 107, 84, 75, 0]
}

df = pd.DataFrame(data)

# Streamlit App
st.title("🚍 Public Transport Cost Dashboard – Gauteng (Pretoria & Johannesburg)")

st.markdown("""
Explore average costs and travel times for various transport modes in South Africa's major urban centers.
""")

st.sidebar.header("Filter Options")
sort_by = st.sidebar.selectbox("Sort Charts By", df.columns[1:])

# Bar Plot Generator
def make_barplot(x, title, xlabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=x, y="Mode", data=df.sort_values(by=x, ascending=False), ax=ax, palette="Set2")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Transport Mode")
    st.pyplot(fig)

# Plot 1: Average Cost per Trip
make_barplot("Avg Cost per Trip (ZAR)", "Average Cost per Trip by Mode", "ZAR")

# Plot 2: Monthly Cost
make_barplot("Avg Monthly Cost (ZAR)", "Estimated Monthly Transport Cost", "ZAR (Monthly)")

# Plot 3: Cost per Kilometer
make_barplot("Cost per km (ZAR/km)", "Cost per Kilometer by Mode", "ZAR/km")

# Plot 4: Travel Time
make_barplot("Avg Travel Time (min)", "Average Travel Time by Mode", "Minutes")

# Show raw data
st.markdown("### Raw Data")
st.dataframe(df)

