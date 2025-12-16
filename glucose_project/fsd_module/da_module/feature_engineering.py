import pandas as pd
import numpy as np

# --- Configuration ---
DATA_PATH = '../data/ohio_t1dm_simulated.csv'
TIME_STEP = 5 # Data interval in minutes
SEQUENCE_LENGTH = 12 # 12 steps of 5 min = 1 hour of history (Input Window)
PREDICTION_HORIZON = 6 # 6 steps of 5 min = 30 minutes into the future (Target)

def create_time_series_features(df_group):
    """
    Applies feature engineering (Lagged Variables and Rate of Change)
    to a single patient's time-series data.
    """
    # 1. Create Lagged BG Features (Historical Sequence)
    # BG values from t-5, t-10, ..., t-60 minutes
    # These form the core sequence input features for the LSTM
    for i in range(1, SEQUENCE_LENGTH + 1):
        df_group[f'BG_lag_{i}'] = df_group['bg_cgm'].shift(i)
    
    # 2. Create Rate of Change (Velocity) Features
    # Difference between current BG and BG 15, 30 min ago
    df_group['BG_ROC_15'] = df_group['bg_cgm'] - df_group['bg_cgm'].shift(3) # 15 min change
    df_group['BG_ROC_30'] = df_group['bg_cgm'] - df_group['bg_cgm'].shift(6) # 30 min change
    
    # 3. Target Variable (BG at t+30)
    # The output label is the BG value 'PREDICTION_HORIZON' steps ahead
    df_group['BG_target_t+30'] = df_group['bg_cgm'].shift(-PREDICTION_HORIZON)

    return df_group

def prepare_data_for_lstm():
    """Reads data, engineers features, and formats for training."""
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}. Please ensure it exists.")
        return None, None, None

    # Ensure timestamp is datetime type and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['patient_id', 'timestamp'])

    # Apply feature creation grouped by patient
    df = df.groupby('patient_id', group_keys=False).apply(create_time_series_features)

    # --- Select Input Features (X) and Target (y) ---
    feature_cols = [col for col in df.columns if col.startswith('BG_lag_') or col.startswith('BG_ROC_')]
    
    # Add other contextual and static features
    context_cols = ['meal_carbs', 'meal_fat', 'activity_steps', 'activity_intensity', 'basal_rate', 'age', 'bmi']
    X_cols = feature_cols + context_cols
    y_col = 'BG_target_t+30'

    # Drop rows where NaNs were introduced by shifting (first 1 hour and last 30 min)
    df_clean = df.dropna(subset=X_cols + [y_col])
    
    # Select final arrays for LSTM training
    X = df_clean[X_cols].values
    y = df_clean[y_col].values
    
    # Store patient IDs and timestamps for splitting/validation (optional, but good practice)
    meta = df_clean[['patient_id', 'timestamp']]
    
    print(f"Total processed samples: {len(df_clean)}")
    print(f"Number of input features (per sample): {X.shape[1]}")
    
    return X, y, meta

if __name__ == '__main__':
    X_features, y_target, meta_df = prepare_data_for_lstm()
    if X_features is not None:
        print("\nFeature Preparation Complete.")
        print(f"Input Shape (X): {X_features.shape}")
        print(f"Target Shape (y): {y_target.shape}")