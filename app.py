import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Weather Analysis Dashboard", layout="wide")

# ------------------------------------
# LOAD DATA
# ------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("weather_classification_data.csv")

df = load_data()

# ------------------------------------
# SIDEBAR
# ------------------------------------
st.sidebar.title("⚙️ Filters")

weather_types = ["All"] + sorted(df["Weather Type"].unique())
selected_weather = st.sidebar.selectbox("Select Weather Type:", weather_types)

locations = ["All"] + sorted(df["Location"].unique())
selected_location = st.sidebar.selectbox("Select Location:", locations)

filtered_df = df.copy()

if selected_weather != "All":
    filtered_df = filtered_df[filtered_df["Weather Type"] == selected_weather]

if selected_location != "All":
    filtered_df = filtered_df[filtered_df["Location"] == selected_location]

st.sidebar.markdown("---")
st.sidebar.write("🔽 Download filtered data")
st.sidebar.download_button(
    label="Download CSV",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_weather _classification_data.csv",
    mime="text/csv"
)

# ------------------------------------
# HEADER
# ------------------------------------
st.title("🌤 **Weather Analysis Dashboard**")
st.markdown("An interactive dashboard built using **Python, Data Science, and Streamlit**.")

st.markdown("---")

# ------------------------------------
# SUMMARY KPIs
# ------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("🌡 Avg Temperature", f"{filtered_df['Temperature'].mean():.1f}°C")
col2.metric("💧 Avg Humidity", f"{filtered_df['Humidity'].mean():.1f}%")
col3.metric("🌬 Avg Wind Speed", f"{filtered_df['Wind Speed'].mean():.1f} km/h")
col4.metric("☀ Avg UV Index", f"{filtered_df['UV Index'].mean():.1f}")

st.markdown("---")

# ------------------------------------
# TABS (Dataset | Visuals | Insights)
# ------------------------------------
tab1, tab2, tab3 = st.tabs(["📄 Dataset", "📊 Visual Analysis", "🧠 Insights"])

# ===============================
# TAB 1: DATASET
# ===============================
with tab1:
    st.subheader("📄 Dataset Preview")
    st.dataframe(filtered_df.head(20))

    st.subheader("🔢 Data Summary")
    st.write(filtered_df.describe())

# ===============================
# TAB 2: VISUAL ANALYSIS
# ===============================
with tab2:

    # Temperature Distribution
    st.subheader("🌡 Temperature Distribution")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    sns.histplot(filtered_df["Temperature"], kde=True, color="orange", ax=ax1)
    st.pyplot(fig1)

    # Weather Type Count
    st.subheader("☁ Weather Type Count")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.countplot(data=filtered_df, x="Weather Type", palette="viridis", ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Humidity vs Temperature
    st.subheader("💧 Humidity vs Temperature Scatter Plot")
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    sns.scatterplot(
        data=filtered_df,
        x="Temperature",
        y="Humidity",
        hue="Weather Type",
        palette="tab10",
        s=80,
        ax=ax3
    )
    st.pyplot(fig3)

    # Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")
    num_df = filtered_df.select_dtypes(include=["float64", "int64"])
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax4)
    st.pyplot(fig4)

# ===============================
# TAB 3: INSIGHTS
# ===============================
with tab3:

    st.subheader("🧠 Key Insights from the Data")

    st.write("""
    ### 📌 **1. Weather Pattern Trends**
    - High humidity often corresponds to Rainy or Cloudy weather.
    - Sunny days are generally associated with higher temperatures and lower humidity.

    ### 🌡 **2. Temperature Behavior**
    - Temperature shows a normal distribution with slight positive skew.
    - Some regions show significantly higher temperature spikes.

    ### 💧 **3. Humidity Observations**
    - Locations with high humidity (>80%) frequently experience rainy conditions.

    ### 🔥 **4. Correlation Analysis**
    - Strong negative correlation between **Temperature** and **Humidity**.
    - UV Index has a moderate positive correlation with temperature.

    ### 🌍 **5. Location Impact**
    - Some locations have consistent weather patterns while others fluctuate strongly.
    """)

st.markdown("---")
st.success("Dashboard loaded successfully ✨")

