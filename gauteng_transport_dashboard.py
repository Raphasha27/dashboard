import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Gauteng Transport Dashboard",
    page_icon="🚍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Polish
st.markdown("""
<style>
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #3E404D;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF4B4B;
    }
    .metric-label {
        font-size: 1rem;
        color: #A0A0A0;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #FAFAFA;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA GENERATION (MOCK) ---
@st.cache_data
def generate_data():
    """Generates realistic mock data for Gauteng transport."""
    # Stations/Hubs in Gauteng
    hubs = {
        "Johannesburg Park Station": [-26.1952, 28.0416],
        "Sandton Gautrain": [-26.1076, 28.0567],
        "Pretoria Station": [-25.7592, 28.1883],
        "Soweto (Bara)": [-26.2598, 27.9400],
        "Midrand": [-25.9964, 28.1278],
        "Rosebank": [-26.1458, 28.0419],
        "OR Tambo Airport": [-26.1367, 28.2411],
        "Centurion": [-25.8524, 28.1868],
        "Menlyn": [-25.7820, 28.2750],
        "Fourways": [-26.0227, 28.0076]
    }
    
    hub_names = list(hubs.keys())
    modes = ["Gautrain", "Metrobus", "Rea Vaya (BRT)", "Mini-Bus Taxi", "Uber/Bolt"]
    
    # Generate 5000 ride records for the last 30 days
    dates = pd.date_range(end=datetime.now(), periods=30*24, freq='H')
    
    data = []
    
    for _ in range(2000):
        start_hub = np.random.choice(hub_names)
        end_hub = np.random.choice(hub_names)
        while start_hub == end_hub:
            end_hub = np.random.choice(hub_names)
            
        mode = np.random.choice(modes, p=[0.15, 0.20, 0.20, 0.40, 0.05])
        
        # Base Fare
        fare_base = {"Gautrain": 80, "Metrobus": 20, "Rea Vaya (BRT)": 15, "Mini-Bus Taxi": 18, "Uber/Bolt": 120}
        fare = fare_base[mode] * np.random.uniform(0.8, 1.5)
        
        # Satisfaction
        satisfaction = np.random.randint(1, 6)
        if mode == "Gautrain": satisfaction = np.random.randint(4, 6)
        if mode == "Mini-Bus Taxi": satisfaction = np.random.randint(2, 5)
        
        # Status
        status = np.random.choice(["On Time", "Delayed", "Cancelled"], p=[0.85, 0.12, 0.03])
        
        data.append({
            "Date": np.random.choice(dates),
            "Start_Hub": start_hub,
            "End_Hub": end_hub,
            "Start_Lat": hubs[start_hub][0],
            "Start_Lon": hubs[start_hub][1],
            "Mode": mode,
            "Passengers": np.random.randint(1, 60), # per trip vehicle average
            "Fare_Total": fare,
            "Status": status,
            "Satisfaction": satisfaction
        })
        
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.day_name()
    return df, hubs

df, hubs = generate_data()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bus.png", width=60)
    st.title("Gauteng Transit")
    st.caption("v2.1.0 • Professional Edition")
    
    st.divider()
    
    # Filters
    st.subheader("🛠️ Filters")
    selected_modes = st.multiselect("Select Transport Modes", df['Mode'].unique(), default=df['Mode'].unique())
    
    date_range = st.date_input(
        "Select Date Range",
        value=(df['Date'].min(), df['Date'].max()),
        min_value=df['Date'].min(),
        max_value=df['Date'].max()
    )
    
    st.divider()
    st.info("💡 **Pro Tip**: Toggle dark mode in settings for best viewing experience.")

    # Filter Data Logic (Moved here for Sidebar access)
    filtered_df = df[df['Mode'].isin(selected_modes)]
    if len(date_range) == 2:
        filtered_df = filtered_df[(filtered_df['Date'].dt.date >= date_range[0]) & (filtered_df['Date'].dt.date <= date_range[1])]

    # Data Export (New Feature)
    st.subheader("💾 Export Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Filtered Data (CSV)",
        data=csv,
        file_name="gauteng_transport_data.csv",
        mime="text/csv",
    )

# --- 4. MAIN DASHBOARD ---
st.title("🚍 Gauteng Transport Intelligence Dashboard")
st.markdown(f"**Real-time analytics for {', '.join(selected_modes)}**")

# Top Level Metrics
col1, col2, col3, col4 = st.columns(4)

if not filtered_df.empty:
    total_px = filtered_df['Passengers'].sum() * 15 # Simulated total
    avg_satisfaction = filtered_df['Satisfaction'].mean()
    on_time_pct = (filtered_df[filtered_df['Status'] == 'On Time'].shape[0] / filtered_df.shape[0]) * 100
    revenue = filtered_df['Fare_Total'].sum() * 15 # Simulated
else:
    total_px = 0
    avg_satisfaction = 0
    on_time_pct = 0
    revenue = 0

def metric_card(label, value, prefix="", suffix=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{prefix}{value}{suffix}</div>
    </div>
    """

with col1: st.markdown(metric_card("Total Passengers", f"{total_px:,.0f}"), unsafe_allow_html=True)
with col2: st.markdown(metric_card("Avg. Satisfaction", f"{avg_satisfaction:.1f}", suffix="/5.0"), unsafe_allow_html=True)
with col3: st.markdown(metric_card("On-Time Performance", f"{on_time_pct:.1f}", suffix="%"), unsafe_allow_html=True)
with col4: st.markdown(metric_card("Est. Revenue (ZAR)", f"{revenue/1000000:.1f}", prefix="R", suffix="M"), unsafe_allow_html=True)

st.write("") # Spacer

# Tabs for detailed views
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Geospatial Ops", "📊 Analytics & Trends", "📉 Route Diagnostics", "🤖 AI Companion"])

# --- TAB 1: GEOSPATIAL ---
with tab1:
    st.subheader("📍 Live Network Activity")
    
    # Standard Streamlit Map (More robust)
    # Prepare data: Needs lat/lon columns
    map_data = filtered_df[['Start_Lat', 'Start_Lon']].rename(columns={'Start_Lat': 'lat', 'Start_Lon': 'lon'})
    st.map(map_data, zoom=9, use_container_width=True)
    
    col_map1, col_map2 = st.columns([1, 1])
    with col_map1:
        st.markdown("### 🚨 Active Alerts")
        st.warning("⚠️ **N1 Highway Delay**: Major congestion near Midrand (35 min delay).")
        st.error("🛑 **Rosebank Route Suspended**: Due to ongoing maintenance.")
        st.info("ℹ️ **Weather**: Heavy rain predicted in Pretoria this afternoon.")
        
    with col_map2:
        st.markdown("### 🏆 Top Hubs by Traffic")
        top_hubs = filtered_df['Start_Hub'].value_counts().head(5)
        st.bar_chart(top_hubs, color="#FF4B4B")

# --- TAB 2: ANALYTICS ---
with tab2:
    st.subheader("📈 Operational Trends")
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown("**Passenger Volume by Hour**")
        hourly_counts = filtered_df.groupby('Hour')['Passengers'].sum().reset_index()
        fig_hourly = px.area(hourly_counts, x='Hour', y='Passengers', color_discrete_sequence=['#FF4B4B'])
        fig_hourly.update_layout(xaxis_title="Hour of Day", yaxis_title="Passengers", template="plotly_dark")
        st.plotly_chart(fig_hourly, use_container_width=True)
        
    with col_a2:
        st.markdown("**Market Share by Mode**")
        mode_counts = filtered_df['Mode'].value_counts().reset_index()
        mode_counts.columns = ['Mode', 'Trips']
        fig_pie = px.pie(mode_counts, values='Trips', names='Mode', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.markdown("**Revenue vs On-Time Performance Trend**")
    # Resample by day
    daily_stats = filtered_df.set_index('Date').resample('D').agg({'Fare_Total': 'sum', 'Status': lambda x: (x=='On Time').mean()}).reset_index()
    daily_stats['On_Time_Pct'] = daily_stats['Status'] * 100
    
    fig_dual = go.Figure()
    fig_dual.add_trace(go.Bar(x=daily_stats['Date'], y=daily_stats['Fare_Total'], name="Revenue (ZAR)", marker_color='#262730'))
    fig_dual.add_trace(go.Scatter(x=daily_stats['Date'], y=daily_stats['On_Time_Pct'], name="On-Time %", yaxis='y2', line=dict(color='#FF4B4B', width=3)))
    
    fig_dual.update_layout(
        template="plotly_dark",
        yaxis=dict(title="Revenue"),
        yaxis2=dict(title="On-Time %", overlaying='y', side='right', range=[0, 100]),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_dual, use_container_width=True)

# --- TAB 3: ROUTE DIAGNOSTICS ---
with tab3:
    st.subheader("📉 Route Efficiency Analysis")
    
    col_d1, col_d2 = st.columns([2, 1])
    
    with col_d1:
        st.markdown("**Route Performance Scatter**")
        # Aggregation
        route_stats = filtered_df.groupby(['Start_Hub', 'End_Hub', 'Mode']).agg({
            'Passengers': 'mean',
            'Satisfaction': 'mean',
            'Status': lambda x: (x == 'On Time').mean() * 100
        }).reset_index()
        route_stats['Route_Name'] = route_stats['Start_Hub'] + " ➝ " + route_stats['End_Hub']
        
        fig_scatter = px.scatter(
            route_stats, 
            x="Satisfaction", 
            y="Status", 
            size="Passengers", 
            color="Mode",
            hover_name="Route_Name",
            title="Satisfaction vs Reliability (Size = Volume)",
            template="plotly_dark"
        )
        fig_scatter.update_layout(xaxis_title="Avg Satisfaction (1-5)", yaxis_title="On-Time %")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_d2:
        st.markdown("**🔍 High Priority Issues**")
        # Filter for low satisfaction routes
        problem_routes = route_stats[route_stats['Satisfaction'] < 3.0].sort_values('Satisfaction')
        st.dataframe(
            problem_routes[['Route_Name', 'Mode', 'Satisfaction']].head(10),
            hide_index=True,
            column_config={
                "Satisfaction": st.column_config.ProgressColumn(
                    "Rating",
                    help="User satisfaction score",
                    format="%.1f",
                    min_value=1,
                    max_value=5,
                ),
            }
        )

# --- TAB 4: AI ASSISTANT ---
with tab4:
    st.subheader("🤖 Smart Transport Assistant")
    
    col_ai1, col_ai2 = st.columns([1, 1])
    
    with col_ai1:
        st.markdown("""
        **Ask me anything about the transport network!**
        *Examples:*
        - "What is the busiest route?"
        - "How is Gautrain performing?"
        - "Show me predicted delays."
        """)
        
        # Simple Rule-Based AI Chatbot
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Type your query here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # AI Logic (Simulated)
            response = ""
            prompt_lower = prompt.lower()
            
            if "busiest" in prompt_lower or "popular" in prompt_lower:
                top_route = filtered_df['End_Hub'].mode()[0]
                response = f"Based on current data, the busiest destination hub is **{top_route}**. Peak traffic is observed between 07:00 and 09:00."
            elif "gautrain" in prompt_lower:
                g_stats = filtered_df[filtered_df['Mode'] == 'Gautrain']
                g_sat = g_stats['Satisfaction'].mean()
                response = f"**Gautrain Status**: Operating normally with an average satisfaction score of **{g_sat:.1f}/5.0**. On-time performance is at 94%."
            elif "delay" in prompt_lower or "predict" in prompt_lower:
                response = "⚠️ **Prediction Alert**: My models indicate a **high probability of delays (65%)** on the **N1 Midrand corridor** tomorrow morning due to expected heavy rains. Suggest rerouting Metrobus services."
            elif "revenue" in prompt_lower:
                rev = filtered_df['Fare_Total'].sum() * 15
                response = f"The estimated total revenue for the selected period is **R{rev:,.2f}**. This is 12% higher than the previous period."
            else:
                response = "I'm analyzing that query... For now, I can tell you that the network is operating at **88% efficiency**. Try asking about specific modes or routes!"
            
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    with col_ai2:
        st.markdown("### 🔮 Predictive Insights")
        st.info("These projections are generated by the NexusSys ML Engine.")
        
        # Simulated Forecast Chart
        future_dates = pd.date_range(start=df['Date'].max(), periods=7, freq='D')
        future_demand = np.random.normal(loc=total_px/30, scale=total_px/100, size=7) * [1.1, 1.05, 1.2, 1.25, 0.9, 0.8, 1.1] # Trend
        
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Demand': future_demand})
        
        fig_forecast = px.line(forecast_df, x='Date', y='Predicted Demand', markers=True, title="7-Day Passenger Demand Forecast")
        fig_forecast.update_traces(line_color='#00CC96')
        fig_forecast.update_layout(template="plotly_dark")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        st.success("✅ **Optimization**: Capacity should be increased by **15%** on Friday to meet expected demand spikes.")

# Footer
st.markdown("---")
st.caption("© 2025 Gauteng Transport/NexusSys | Developed by Raphasha27")
