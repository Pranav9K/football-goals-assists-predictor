import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="2026/2027 Football Performance Predictor",
    layout="wide"
)

st.title("Football Player Future Performance Predictor (2026-2027)")
st.markdown("This system applies a trained machine learning model across rolling multi-season consistency and momentum metrics to forecast goal contributions per 90 minutes for the upcoming season in their respective domestic leagues.")

@st.cache_resource
def load_prediction_assets():
    model = joblib.load('streamlit_assets/champion_predictor_model.pkl')
    return model

@st.cache_data
def load_player_features():
    df = pd.read_csv('streamlit_assets/future_forecast_features.csv')
    return df

try:
    forecasting_model = load_prediction_assets()
    features_df = load_player_features()
except Exception as e:
    st.error("system error: missing core prediction matrices. ensure you generated and saved the feature schemas into your streamlit assets folder.")
    st.stop()


st.sidebar.header("Scouting search criteria")


available_positions = sorted([str(x) for x in features_df['Pos_24'].dropna().unique().tolist()])
selected_pos = st.sidebar.multiselect("filter by player position:", available_positions, default=available_positions)

filtered_df = features_df[features_df['Pos_24'].isin(selected_pos)]

if filtered_df.empty:
    st.warning("no players found matching the selected position criteria. please select at least one position in the sidebar filter.")
    st.stop()


selected_player = st.sidebar.selectbox("select player to scout:", sorted(filtered_df['Player'].tolist()))

player_rows = features_df[features_df['Player'] == selected_player]
if player_rows.empty:
    st.error("selected player data could not be parsed safely.")
    st.stop()

player_data = player_rows.iloc[0]

st.sidebar.markdown("---")
st.sidebar.subheader("2026-2027 playing time simulator")

simulated_90s = st.sidebar.slider("simulate projected volume (90s played):", 1.0, 38.0, 25.0, step=0.5)

features_schema = [
    'Age_current', 'Pos_encoded', '90s_24', '90s_25', 'GA_per90_24',
    'GA_per90_25', 'GA_consistency', 'GA_volatility', 'GA_trend', 
    'PrgC_24', 'PrgP_24', 'PrgR_24' 
]

model_features = [
    'Age_current', 'Pos_encoded', '90s_23', '90s_24', 'GA_per90_23',
    'GA_per90_24', 'GA_consistency', 'GA_volatility', 'GA_trend', 
    'PrgC_24', 'PrgP_24', 'PrgR_24'
]

input_payload = pd.DataFrame([player_data[features_schema].values], columns=model_features)


predicted_ga_per90 = forecasting_model.predict(input_payload)[0]


predicted_ga_per90 = max(0.0, float(predicted_ga_per90))
projected_total_ga = predicted_ga_per90 * simulated_90s

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="scouted player", value=selected_player)
with col2:
    st.metric(label="projected age (26/27)", value=f"{int(player_data['Age_current'])} years")
with col3:
    st.metric(label="predicted G+A per 90", value=f"{predicted_ga_per90:.2f}")
with col4:
    st.metric(label="total projected G+A output", value=f"{projected_total_ga:.1f}", delta=f"{predicted_ga_per90 * 38:.1f} max potential season ceiling", delta_color="normal")

st.markdown("---")

graph, = st.columns(1)

with graph:
    st.subheader("Historical trajectory vs Machine learning projection")
    
    seasons = ['24-25 season', '25-26 season', '26-27 prediction']
    historical_rates = [player_data['GA_per90_24'], player_data['GA_per90_25'], predicted_ga_per90]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(x=seasons, y=historical_rates, marker='o', sort=False, color='#1f77b4', linewidth=3, ax=ax)
    ax.set_ylabel("goals + assists (per 90 minutes)")
    ax.set_ylim(0, max(historical_rates) * 1.3 if max(historical_rates) > 0 else 1.0)
    
    for i, v in enumerate(historical_rates):
        ax.text(i, v + (max(historical_rates) * 0.05 if max(historical_rates) > 0 else 0.05), f"{v:.2f}", ha='center', fontweight='bold')
        
    sns.despine()
    st.pyplot(fig)
