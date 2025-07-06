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

# --- User Inputs for Prediction ---
st.header('Weekly Prediction Inputs')
st.info('Select the GP for historical data and enter the latest qualifying results below.')

year = 2024 # We'll use 2024 as the year for historical data

# List of GPs from the 2024 season for selection
gp_list = ['Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China', 'Miami', 'Emilia Romagna', 'Monaco', 'Canada', 'Spain', 'Austria', 'Great Britain', 'Hungary', 'Belgium', 'Netherlands', 'Italy', 'Azerbaijan', 'Singapore', 'United States', 'Mexico', 'Brazil', 'Las Vegas', 'Qatar', 'Abu Dhabi']

# Set default to Great Britain (Silverstone)
try:
    default_ix = gp_list.index('Great Britain')
except ValueError:
    default_ix = 0

gp_name = st.selectbox('Select Grand Prix for Historical Data:', gp_list, index=default_ix)

race_laps = load_race_data(year, gp_name)

if not race_laps.empty:
    # Get average lap times for each driver from the 2024 race
    avg_race_laps = race_laps.groupby('Driver')['LapTime'].mean().reset_index()
    avg_race_laps.rename(columns={'LapTime': 'AvgRaceLapTime'}, inplace=True)
    st.write(f'Historical Average Race Lap Times from {year} {gp_name}:')
    st.dataframe(avg_race_laps)

    # --- 3. Load Actual Qualifying Data ---
    st.header('Loading Actual Qualifying Data')

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

                # Correctly find fastest lap and map driver names
                fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()].copy()
                fastest_laps['QualifyingTime'] = fastest_laps['LapTime'].dt.total_seconds()

                # Correctly build the driver name map (Abbreviation -> FullName)
                driver_map = {drv['Abbreviation']: drv['FullName'] for drv in session.drivers.values()}
                fastest_laps['DriverName'] = fastest_laps['Driver'].map(driver_map)

                final_data = fastest_laps[['DriverName', 'Driver', 'QualifyingTime']]
                final_data.to_csv(cache_file, index=False)
                return final_data
            except Exception as e:
                st.error(f'Could not load qualifying data for {year} {gp_name}: {e}')
                return pd.DataFrame()

    # We use the same year for qualifying as for the race data for our model
    qualifying_df_2025 = load_qualifying_data(year, gp_name)

    if not qualifying_df_2025.empty:
        st.write(f'Actual Qualifying Results from {year} {gp_name}:')
        st.dataframe(qualifying_df_2025[['DriverName', 'QualifyingTime']])

        # --- 4. Merge qualifying and race data ---
        merged_data = pd.merge(qualifying_df_2025, avg_race_laps, on='Driver', how='inner')
        
        if not merged_data.empty:
            merged_data['AvgRaceLapTime'] = merged_data['AvgRaceLapTime'].dt.total_seconds()
            st.write('Merged Data for Model Training:')
            st.dataframe(merged_data[['DriverName', 'QualifyingTime', 'AvgRaceLapTime']])

            if len(merged_data) > 1:
                # --- 5. Train a model ---
                st.header('Model Training')
                X = merged_data[['QualifyingTime']]
                y = merged_data['AvgRaceLapTime']

                model = LinearRegression()
                model.fit(X, y)
                st.success('Model trained successfully!')

                # --- 6. Predict race performance ---
                st.header('Race Performance Prediction')
                predicted_race_times = model.predict(qualifying_df_2025[['QualifyingTime']])
                qualifying_df_2025['PredictedRaceTime'] = predicted_race_times

                # --- 7. Rank drivers ---
                st.header('Predicted Race Rankings')
                ranked_drivers = qualifying_df_2025.sort_values(by='PredictedRaceTime').reset_index(drop=True)
                ranked_drivers['Rank'] = ranked_drivers.index + 1
                st.dataframe(ranked_drivers[['Rank', 'DriverName', 'QualifyingTime', 'PredictedRaceTime']])

                # --- 8. Evaluate the model ---
                st.header('Model Evaluation')
                mae = mean_absolute_error(y, model.predict(X))
                st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f} seconds")
                st.info('This MAE is calculated on the training data (drivers present in both datasets).')
            else:
                st.warning('Not enough common drivers between qualifying and historical race data to train the model. Need at least two.')
        else:
            st.warning('No common drivers found between the qualifying session and the historical race data. Cannot build a model.')
    else:
        # This part is executed if qualifying_df_2025 is empty
        st.info("Prediction will be available once the qualifying session data is loaded.")
else:
    st.error('Could not load initial race data. The application cannot proceed.')
