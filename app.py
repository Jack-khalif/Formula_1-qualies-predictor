import streamlit as st
import fastf1 as ff1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
import os

st.set_page_config(layout="wide")

st.title('F1 Grand Prix Race Predictor')

# --- 1. Enable FastF1 Cache ---
st.header('Configuration')
cache_path = 'fastf1_cache'
if not os.path.exists(cache_path):
    os.makedirs(cache_path)
ff1.Cache.enable_cache(cache_path)
st.success(f'FastF1 cache enabled at: {os.path.abspath(cache_path)}')

# --- 2. Load 2024 GP Race Session and Extract Lap Times ---
st.header('Data Loading and Preparation')

@st.cache_data
def load_race_data(year, gp_name):
    cache_file = f'data_cache/race_data_{year}_{gp_name.replace(" ", "_")}.csv'
    required_cols = ['Driver', 'LapTime']

    if os.path.exists(cache_file):
        try:
            data = pd.read_csv(cache_file)
            if all(col in data.columns for col in required_cols):
                data['LapTime'] = pd.to_timedelta(data['LapTime'])
                return data
        except Exception:
            pass # Ignore errors in reading cache, just refetch

    with st.spinner(f'Loading {year} {gp_name} race data from FastF1...'):
        try:
            session = ff1.get_session(year, gp_name, 'R')
            session.load(laps=True, telemetry=False, weather=False)
            laps = session.laps
            if laps.empty:
                st.warning(f'No race lap data found for {year} {gp_name}.')
                return pd.DataFrame()
            laps.to_csv(cache_file, index=False)
            return laps
        except Exception as e:
            st.error(f'Could not load race data for {year} {gp_name}: {e}')
            return pd.DataFrame()

# --- UI Sidebar and Controls ---
st.sidebar.header('Settings')

use_synthetic_data = st.sidebar.checkbox('Use Synthetic Qualifying Data', value=True)

# List of 2024 Grand Prix
gp_list_2024 = [
    'Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China', 'Miami',
    'Emilia Romagna', 'Monaco', 'Canada', 'Spain', 'Austria', 'Great Britain',
    'Hungary', 'Belgium', 'Netherlands', 'Italy', 'Azerbaijan', 'Singapore',
    'United States', 'Mexico', 'Brazil', 'Las Vegas', 'Qatar', 'Abu Dhabi'
]

# Dropdown to select the Grand Prix
gp_name = st.sidebar.selectbox(
    'Select a Grand Prix for Historical Data (2024 Season):',
    gp_list_2024,
    index=gp_list_2024.index('Great Britain')  # Default to Great Britain
)
year = 2024

# --- Data Loading Functions ---

def load_synthetic_qualifying_data():
    st.info("Using synthetic qualifying data for Silverstone 2025.")
    data = {
        'Driver': ['VER', 'LEC', 'PER', 'SAI', 'HAM', 'RUS', 'NOR', 'PIA', 'ALO', 'OCO', 'BOT', 'ZHO', 'MAG', 'HUL', 'GAS', 'TSU', 'VET', 'STR', 'ALB', 'SAR'],
        'QualifyingTime': [88.2, 88.4, 88.5, 88.6, 88.8, 88.9, 89.0, 89.1, 89.2, 89.3, 89.4, 89.5, 89.6, 89.7, 89.8, 89.9, 90.0, 90.1, 90.2, 90.3],
        'DriverName': ['Max Verstappen', 'Charles Leclerc', 'Sergio Pérez', 'Carlos Sainz', 'Lewis Hamilton', 'George Russell', 'Lando Norris', 'Oscar Piastri', 'Fernando Alonso', 'Esteban Ocon', 'Valtteri Bottas', 'Guanyu Zhou', 'Kevin Magnussen', 'Nico Hülkenberg', 'Pierre Gasly', 'Yuki Tsunoda', 'Sebastian Vettel', 'Lance Stroll', 'Alexander Albon', 'Logan Sargeant']
    }
    return pd.DataFrame(data)

@st.cache_data
def load_qualifying_data(year, gp_name):
    cache_file = f'data_cache/qualifying_data_{year}_{gp_name.replace(" ", "_")}.csv'
    required_cols = ['DriverName', 'Driver', 'QualifyingTime']

    if os.path.exists(cache_file):
        try:
            data = pd.read_csv(cache_file)
            if all(col in data.columns for col in required_cols):
                return data
        except Exception:
            pass # Ignore errors, just refetch

    with st.spinner(f'Loading {year} {gp_name} qualifying data from FastF1...'):
        try:
            session = ff1.get_session(year, gp_name, 'Q')
            session.load(laps=True, telemetry=False, weather=False)
            laps = session.laps
            if laps.empty:
                st.warning(f'No qualifying lap data found for {year} {gp_name}.')
                return pd.DataFrame()

            fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()].copy()
            fastest_laps['QualifyingTime'] = fastest_laps['LapTime'].dt.total_seconds()
            driver_map = {drv['Abbreviation']: drv['FullName'] for drv in session.drivers.values()}
            fastest_laps['DriverName'] = fastest_laps['Driver'].map(driver_map)

            final_data = fastest_laps[['DriverName', 'Driver', 'QualifyingTime']]
            final_data.to_csv(cache_file, index=False)
            return final_data
        except Exception as e:
            st.error(f'Could not load qualifying data for {year} {gp_name}: {e}')
            return pd.DataFrame()

# --- Main Application Logic ---
if use_synthetic_data:
    qualifying_df_to_predict = load_synthetic_qualifying_data()
else:
    qualifying_df_to_predict = load_qualifying_data(year, gp_name)

if qualifying_df_to_predict is not None and not qualifying_df_to_predict.empty:
    race_df_2024 = load_race_data(year, gp_name)

    if race_df_2024 is not None and not race_df_2024.empty:
        race_df_2024['RaceTime'] = race_df_2024['LapTime'].dt.total_seconds()
        avg_race_times_2024 = race_df_2024.groupby('Driver')['RaceTime'].mean().reset_index()
        
        qualifying_df_2024 = load_qualifying_data(year, gp_name)

        if qualifying_df_2024 is not None and not qualifying_df_2024.empty:
            merged_df_2024 = pd.merge(qualifying_df_2024, avg_race_times_2024, on='Driver')

            if not merged_df_2024.empty:
                X_train = merged_df_2024[['QualifyingTime']]
                y_train = merged_df_2024['RaceTime']
                model = LinearRegression()
                model.fit(X_train, y_train)

                X_pred = qualifying_df_to_predict[['QualifyingTime']]
                predicted_race_times = model.predict(X_pred)
                qualifying_df_to_predict['PredictedRaceTime'] = predicted_race_times

                ranked_drivers = qualifying_df_to_predict.sort_values(by='PredictedRaceTime').reset_index(drop=True)
                ranked_drivers.index += 1

                st.subheader('Predicted Race Performance Ranking')
                
                # --- Visualization ---
                viz_df = ranked_drivers.set_index('DriverName')
                st.bar_chart(viz_df[['PredictedRaceTime']])

                with st.expander("View Detailed Ranking Data"):
                    st.dataframe(ranked_drivers[['DriverName', 'PredictedRaceTime']])

                mae = mean_absolute_error(y_train, model.predict(X_train))
                st.sidebar.info(f'Model trained on {year} {gp_name} data.\nMAE: {mae:.2f}s')
            else:
                st.error('Could not merge 2024 qualifying and race data for training.')
        else:
            st.warning('Could not load 2024 qualifying data for model training.')
    else:
        st.warning('Could not load 2024 race data for model training.')
else:
    st.warning('Prediction will be available once qualifying data is loaded.')
