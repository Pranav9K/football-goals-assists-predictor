import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# configure basic window layouts
st.set_page_config(
    page_title="2026/2027 Football Performance Predictor",
    layout="wide"
)

st.title("⚽ Football Player Future Performance Predictor (2026-2027)")
st.markdown("this system applies a trained machine learning model across rolling multi-season consistency and momentum metrics to forecast goal contributions per 90 minutes for the upcoming season.")

# secure resource loading from cache arrays
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

# sidebar tracking controls
st.sidebar.header("scouting search criteria")

# extract unique positions and make sure they are strings
available_positions = sorted([str(x) for x in features_df['Pos_24'].dropna().unique().tolist()])
selected_pos = st.sidebar.multiselect("filter by player position:", available_positions, default=available_positions)

# filter operational array down to selected positions
filtered_df = features_df[features_df['Pos_24'].isin(selected_pos)]

if filtered_df.empty:
    st.warning("no players found matching the selected position criteria. please select at least one position in the sidebar filter.")
    st.stop()

# user targeted player search selection
selected_player = st.sidebar.selectbox("select player to scout:", sorted(filtered_df['Player'].tolist()))

# extract base record row for the active selected entity safely
player_rows = features_df[features_df['Player'] == selected_player]
if player_rows.empty:
    st.error("selected player data could not be parsed safely.")
    st.stop()

player_data = player_rows.iloc[0]

st.sidebar.markdown("---")
st.sidebar.subheader("2026-2027 playing time simulator")
# dynamic slider allowing users to toggle simulated minutes for the upcoming season
simulated_90s = st.sidebar.slider("simulate projected volume (90s played):", 1.0, 38.0, 25.0, step=0.5)

# compile machine learning input vectors mapped to expected training features
features_schema = [
    'Age_current', 'Pos_encoded', '90s_24', '90s_25', 'GA_per90_24', 'xG_per90_24', 'xAG_per90_24',
    'GA_per90_25', 'xG_per90_25', 'xAG_per90_25', 'GA_consistency', 'GA_volatility', 'GA_trend', 
    'xG_consistency', 'xG_volatility', 'xG_trend', 'xAG_consistency', 'xAG_volatility', 'xAG_trend',
    'PrgC_25', 'PrgP_25', 'PrgR_25'
]

model_features = [
    'Age_current', 'Pos_encoded', '90s_23', '90s_24', 'GA_per90_23', 'xG_per90_23', 'xAG_per90_23',
    'GA_per90_24', 'xG_per90_24', 'xAG_per90_24', 'GA_consistency', 'GA_volatility', 'GA_trend', 
    'xG_consistency', 'xG_volatility', 'xG_trend', 'xAG_consistency', 'xAG_volatility', 'xAG_trend',
    'PrgC_24', 'PrgP_24', 'PrgR_24'
]

input_payload = pd.DataFrame([player_data[features_schema].values], columns=model_features)

# process live forward calculation engine predictions
predicted_ga_per90 = forecasting_model.predict(input_payload)[0]
# safeguard against anomalies dropping model predictions below mathematical zero floors
predicted_ga_per90 = max(0.0, predicted_ga_per90)
projected_total_ga = predicted_ga_per90 * simulated_90s

# visual grid section 1: key scorecards
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

# visual grid section 2: graphics and performance context curves
graph_col1, graph_col2 = st.columns(2)

with graph_col1:
    st.subheader("📊 historical trajectory vs machine learning projection")
    
    # build a clean timeline mapping values across rolling intervals
    seasons = ['24-25 season', '25-26 season', '26-27 prediction']
    historical_rates = [player_data['GA_per90_24'], player_data['GA_per90_25'], predicted_ga_per90]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(x=seasons, y=historical_rates, marker='o', sort=False, color='#1f77b4', linewidth=3, ax=ax)
    ax.set_ylabel("goals + assists (per 90 minutes)")
    ax.set_ylim(0, max(historical_rates) * 1.3 if max(historical_rates) > 0 else 1.0)
    
    # inject value labels above individual markers to maximize design layout readability
    for i, v in enumerate(historical_rates):
        ax.text(i, v + (max(historical_rates) * 0.05 if max(historical_rates) > 0 else 0.05), f"{v:.2f}", ha='center', fontweight='bold')
        
    sns.despine()
    st.pyplot(fig)

with graph_col2:
    st.subheader("🎯 performance profiles and regression indicators")
    
    st.write(f"**primary tactical grouping:** `{player_data['Pos_24']}`")
    st.write(f"**rolling goal momentum metric (trend):** `{player_data['GA_trend']}:+.2f`")
    st.write(f"**historical volatility parameter:** `{player_data['GA_volatility']:.2f}`")
    
    st.markdown("##### expected threat profiles (25-26 season baseline)")
    st.write(f"expected goals (xG) per 90: `{player_data['xG_per90_25']:.2f}`")
    st.write(f"expected assisted goals (xAG) per 90: `{player_data['xAG_per90_25']:.2f}`")
    
    # dynamic insight generation based on tactical momentum vectors
    if player_data['GA_trend'] > 0.05:
        st.success("scouting brief: player demonstrates a clear ascending performance curve. upward projections are heavily backed by positive growth trajectories.")
    elif player_data['GA_trend'] < -0.05:
        st.warning("scouting brief: baseline analysis shows downward statistical contraction. upcoming model predictions are tuned defensively to adapt to this variance regression.")
    else:
        st.info("scouting brief: player maintains a stable, highly consistent production profile across multi-year data distributions.")