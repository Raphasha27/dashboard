import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="Gauteng Transport Dashboard", layout="wide")

# -------------------------------
# 🎒 Intro
# -------------------------------
st.title("🚦 Gauteng Public Transport Cost & Time Dashboard")
st.markdown("""
Welcome! This dashboard helps you explore the cost and time involved in different public transport options across **Pretoria** and **Johannesburg**.

Use this tool to compare:
- 🚕 Average Cost per Trip
- 📅 Monthly Cost for Daily Commuting
- 📏 Cost per Kilometer
- ⏱️ Travel Time

_This data is based on typical fare and time estimates for common routes._
""")

# -------------------------------
# 📦 Transport Data
# -------------------------------
data = {
    "Mode": ["Minibus Taxi", "Uber Go", "UberX", "Gautrain", "Metrorail", "MetroBus", "Rea Vaya", "Walking"],
    "Avg Cost per Trip (ZAR)": [13, 312/15, 420/15, 92, 13, 9, 10.5, 0],
    "Cost per km (ZAR/km)": [13/15, 312/56, 420/56, 92/36, 13/15, 9/10, 10.5/12, 0],
    "Avg Monthly Cost (ZAR)": [572, 312*22, 420*22, 3254, 581, 352, 420, 0],
    "Avg Travel Time (min)": [63, 50, 48, 38, 107, 84, 75, 0]
}
df = pd.DataFrame(data)

# -------------------------------
# 🔍 Sidebar Filter
# -------------------------------
selected_metric = st.sidebar.selectbox(
    "Select a metric to compare:",
    ["Avg Cost per Trip (ZAR)", "Avg Monthly Cost (ZAR)", "Cost per km (ZAR/km)", "Avg Travel Time (min)"]
)

# -------------------------------
# 📊 Plotting Function
# -------------------------------
def make_plot(column, title, xlabel, palette="Set2"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=column, y="Mode", data=df, palette=palette, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Transport Mode")
    st.pyplot(fig)

# -------------------------------
# 📈 Show Plot Based on Selection
# -------------------------------
if selected_metric == "Avg Cost per Trip (ZAR)":
    make_plot(selected_metric, "Average Cost per Trip", "Cost (ZAR)", "viridis")
elif selected_metric == "Avg Monthly Cost (ZAR)":
    make_plot(selected_metric, "Estimated Monthly Transport Cost", "Monthly Cost (ZAR)", "magma")
elif selected_metric == "Cost per km (ZAR/km)":
    make_plot(selected_metric, "Cost per Kilometer", "Cost per km (ZAR/km)", "cubehelix")
elif selected_metric == "Avg Travel Time (min)":
    make_plot(selected_metric, "Average Travel Time", "Travel Time (Minutes)", "coolwarm")

# -------------------------------
# 🧾 Show Raw Data
# -------------------------------
with st.expander("📄 Show Raw Data"):
    st.dataframe(df.set_index("Mode"))
