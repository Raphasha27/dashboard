import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from fpdf import FPDF
import io
import numpy as np

sns.set(style="whitegrid")

# --- Data Setup ---
def load_data():
    data = {
        "Mode": ["Minibus Taxi", "Uber Go", "UberX", "Gautrain", "Metrorail", "MetroBus", "Rea Vaya", "Walking"],
        "Avg Cost per Trip (ZAR)": [13, 312/15, 420/15, 92, 13, 9, 10.5, 0],
        "Cost per km (ZAR/km)": [13/15, 312/56, 420/56, 92/36, 13/15, 9/10, 10.5/12, 0],
        "Avg Monthly Cost (ZAR)": [572, 312*22, 420*22, 3254, 581, 352, 420, 0],
        "Avg Travel Time (min)": [63, 50, 48, 38, 107, 84, 75, 0],
        "Distance (km)": [15, 56, 56, 36, 15, 10, 12, 0],
        "Frequency per Month": [44, 22, 22, 35, 45, 39, 40, 0],
        # Approximate coordinates for Gauteng transport hubs (dummy data)
        "Latitude": [-25.7449, -26.2041, -26.2041, -25.8880, -26.2041, -26.2023, -26.1966, -25.7461],
        "Longitude": [28.1870, 28.0473, 28.0473, 28.2023, 28.0473, 28.0456, 28.0734, 28.1883]
    }
    return pd.DataFrame(data)

df = load_data()

# --- Functions for plots ---
def plot_bar_cost(filtered_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Avg Cost per Trip (ZAR)", y="Mode", data=filtered_df, palette="viridis", ax=ax)
    ax.set_xlabel("Cost (ZAR)")
    ax.set_ylabel("")
    return fig

def plot_bar_monthly_cost(filtered_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Avg Monthly Cost (ZAR)", y="Mode", data=filtered_df, palette="magma", ax=ax)
    ax.set_xlabel("Monthly Cost (ZAR)")
    ax.set_ylabel("")
    return fig

def plot_pie_frequency(filtered_df):
    pie_data = filtered_df.set_index("Mode")["Frequency per Month"]
    fig, ax = plt.subplots()
    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    ax.axis("equal")
    return fig

def plot_scatter_cost_time(filtered_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x="Cost per km (ZAR/km)", y="Avg Travel Time (min)", hue="Mode", size="Frequency per Month",
                    sizes=(50, 300), data=filtered_df, ax=ax, palette="tab10", legend="brief")
    ax.set_xlabel("Cost per km (ZAR/km)")
    ax.set_ylabel("Average Travel Time (min)")
    return fig

def plot_line_freq_distance(filtered_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x="Distance (km)", y="Frequency per Month", hue="Mode", marker="o", data=filtered_df, ax=ax)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Frequency per Month")
    return fig

def plot_hist_travel_time(filtered_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(filtered_df["Avg Travel Time (min)"], bins=10, kde=True, color="skyblue", ax=ax)
    ax.set_xlabel("Travel Time (min)")
    return fig

def plot_corr_heatmap(filtered_df):
    corr = filtered_df.select_dtypes(include=["float", "int"]).corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    return fig

# --- Map visualization ---
def plot_map(filtered_df):
    m = folium.Map(location=[-26.2041, 28.0473], zoom_start=10, tiles="cartodbpositron")
    for _, row in filtered_df.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=f"{row['Mode']}<br>Avg Cost: {row['Avg Cost per Trip (ZAR)']:.2f} ZAR",
            tooltip=row["Mode"],
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)
    return m

# --- Prediction simulation ---
def predict_cost(base_cost, fuel_price_factor):
    # Simple linear simulation: cost increases with fuel price factor
    return base_cost * fuel_price_factor

# --- PDF report generation ---
def generate_pdf_report(filtered_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Gauteng Transport Report", 0, 1, 'C')

    pdf.set_font("Arial", '', 12)
    for i, row in filtered_df.iterrows():
        pdf.cell(0, 8, f"{row['Mode']}: Avg Trip Cost = {row['Avg Cost per Trip (ZAR)']:.2f} ZAR, "
                       f"Monthly Cost = {row['Avg Monthly Cost (ZAR)']:.0f} ZAR, "
                       f"Travel Time = {row['Avg Travel Time (min)']} min", 0, 1)
    return pdf.output(dest='S').encode('latin1')

# --- Main app ---
def main():
    st.title("🚦 Gauteng Transport Dashboard")

    st.markdown("""
    This dashboard provides detailed insights into transport modes in Gauteng, South Africa.
    Use filters and tabs to explore costs, travel times, maps, and predictions.
    """)

    # Sidebar filters
    st.sidebar.header("Filters")

    modes_selected = st.sidebar.multiselect(
        "Select Transport Modes",
        options=df["Mode"].unique(),
        default=df["Mode"].unique()
    )

    cost_min, cost_max = st.sidebar.slider(
        "Average Cost per Trip (ZAR)",
        float(df["Avg Cost per Trip (ZAR)"].min()),
        float(df["Avg Cost per Trip (ZAR)"].max()),
        (float(df["Avg Cost per Trip (ZAR)"].min()), float(df["Avg Cost per Trip (ZAR)"].max()))
    )

    time_min, time_max = st.sidebar.slider(
        "Average Travel Time (min)",
        float(df["Avg Travel Time (min)"].min()),
        float(df["Avg Travel Time (min)"].max()),
        (float(df["Avg Travel Time (min)"].min()), float(df["Avg Travel Time (min)"].max()))
    )

    # Filter data
    filtered_df = df[
        (df["Mode"].isin(modes_selected)) &
        (df["Avg Cost per Trip (ZAR)"] >= cost_min) &
        (df["Avg Cost per Trip (ZAR)"] <= cost_max) &
        (df["Avg Travel Time (min)"] >= time_min) &
        (df["Avg Travel Time (min)"] <= time_max)
    ]

    # Summary
    st.header("Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Cost per Trip (ZAR)", f"{filtered_df['Avg Cost per Trip (ZAR)'].mean():.2f}")
    col2.metric("Avg Monthly Cost (ZAR)", f"{filtered_df['Avg Monthly Cost (ZAR)'].mean():.2f}")
    col3.metric("Avg Travel Time (min)", f"{filtered_df['Avg Travel Time (min)'].mean():.1f}")

    # Tabs for organizing content
    tabs = st.tabs(["Bar Charts", "Scatter & Line Plots", "Distribution & Heatmap", "Map View", "Prediction", "Feedback", "Data & Report"])

    # Tab 1: Bar Charts
    with tabs[0]:
        st.subheader("Average Cost per Trip")
        st.pyplot(plot_bar_cost(filtered_df))

        st.subheader("Monthly Cost by Mode")
        st.pyplot(plot_bar_monthly_cost(filtered_df))

        st.subheader("Mode Share by Frequency")
        st.pyplot(plot_pie_frequency(filtered_df))

    # Tab 2: Scatter & Line Plots
    with tabs[1]:
        st.subheader("Cost per km vs Travel Time")
        st.pyplot(plot_scatter_cost_time(filtered_df))

        st.subheader("Frequency per Month vs Distance")
        st.pyplot(plot_line_freq_distance(filtered_df))

    # Tab 3: Distribution & Heatmap
    with tabs[2]:
        st.subheader("Travel Time Distribution")
        st.pyplot(plot_hist_travel_time(filtered_df))

        st.subheader("Correlation Heatmap")
        st.pyplot(plot_corr_heatmap(filtered_df))

    # Tab 4: Map View
    with tabs[3]:
        st.subheader("Transport Hubs Map")
        map_obj = plot_map(filtered_df)
        st_folium(map_obj, width=700, height=500)

    # Tab 5: Cost Prediction Simulation
    with tabs[4]:
        st.subheader("Cost Prediction Simulation")
        fuel_factor = st.slider("Fuel Price Multiplier", 0.5, 2.0, 1.0, 0.05)
        st.write("Adjust fuel price multiplier to simulate impact on transport cost.")

        # Show predicted cost table
        predicted_costs = filtered_df.copy()
        predicted_costs["Predicted Cost per Trip (ZAR)"] = predicted_costs["Avg Cost per Trip (ZAR)"].apply(lambda c: predict_cost(c, fuel_factor))
        st.dataframe(predicted_costs[["Mode", "Avg Cost per Trip (ZAR)", "Predicted Cost per Trip (ZAR)"]].style.format({
            "Avg Cost per Trip (ZAR)": "{:.2f}",
            "Predicted Cost per Trip (ZAR)": "{:.2f}"
        }))

    # Tab 6: User Feedback
    with tabs[5]:
        st.subheader("User Feedback")
        feedback = st.text_area("Let us know your thoughts or suggestions:")
        if st.button("Submit Feedback"):
            if feedback.strip() == "":
                st.warning("Please enter some feedback before submitting.")
            else:
                st.success("Thanks for your feedback! 👍")
                # Normally here you'd save feedback to a DB or file
                st.write("You submitted:")
                st.write(feedback)

    # Tab 7: Data Table & PDF Report
    with tabs[6]:
        st.subheader("Filtered Data Table")
        st.dataframe(filtered_df.style.format({
            "Avg Cost per Trip (ZAR)": "{:.2f}",
            "Cost per km (ZAR/km)": "{:.3f}",
            "Avg Monthly Cost (ZAR)": "{:.0f}",
            "Avg Travel Time (min)": "{:.0f}",
            "Distance (km)": "{:.0f}",
            "Frequency per Month": "{:.0f}"
        }))

        # Download filtered data CSV
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name='gauteng_transport_filtered.csv',
            mime='text/csv',
        )

        # Generate and download PDF report
        pdf_bytes = generate_pdf_report(filtered_df)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="gauteng_transport_report.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()
