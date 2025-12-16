import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os # Need to import os for file path management

# --- Configuration ---
FILE_PATH = 'data/ohio_t1dm_simulated.csv'
TIME_INTERVAL_MINUTES = 5
# CHANGED: 14 days per patient (was 0.5 days)
NUM_DAYS_PER_PATIENT = 14  
PATIENT_CONFIG = {
    # ID: (Age, BMI, Basal Rate U/hr)
    559: (42, 25.5, 0.9),
    600: (30, 22.1, 0.8),
    # ADDED: Two more patients for better model generalization
    701: (55, 28.0, 1.1),
    810: (25, 21.0, 0.7),
}

def generate_patient_data(patient_id, age, bmi, basal_rate):
    """Generates synthetic time-series data for a single patient."""
    
    start_time = datetime(2025, 1, 1, 8, 0, 0)
    end_time = start_time + timedelta(days=NUM_DAYS_PER_PATIENT)
    
    # Generate timestamps at 5-minute intervals
    timestamps = []
    current_time = start_time
    while current_time < end_time:
        timestamps.append(current_time)
        current_time += timedelta(minutes=TIME_INTERVAL_MINUTES)
    
    num_samples = len(timestamps)
    
    # Initialize features
    df = pd.DataFrame({
        'patient_id': patient_id,
        'timestamp': timestamps,
        'meal_carbs': 0.0,
        'meal_fat': 0.0,
        'activity_steps': 0,
        'activity_intensity': 0,
        'basal_rate': basal_rate,
        'age': age,
        'bmi': bmi,
    })
    
    # 1. Simulate Baseline BG
    df['bg_cgm'] = np.random.normal(loc=120, scale=5, size=num_samples)

    # 2. Inject Random Events (More complex simulation for 14 days)
    
    # Define event probability and typical values
    MEAL_PROB = 0.05  # 5% chance of a meal start at any 5-min interval
    ACTIVITY_PROB = 0.08 
    
    for i in range(num_samples):
        # Apply random activity
        if np.random.rand() < ACTIVITY_PROB:
            df.loc[i, 'activity_steps'] = np.random.randint(200, 800)
            df.loc[i, 'activity_intensity'] = 1
        
        # Apply random meal
        if np.random.rand() < MEAL_PROB:
            df.loc[i, 'meal_carbs'] = np.random.randint(30, 80)
            df.loc[i, 'meal_fat'] = np.random.randint(5, 25)
            
    # 3. Apply BG Response Simulation (Smoothing and reacting to events)
    bg_data = df['bg_cgm'].copy()
    
    for i in range(1, num_samples):
        # General inertia: BG tends to follow the previous value
        bg_data.iloc[i] = 0.8 * bg_data.iloc[i-1] + 0.2 * bg_data.iloc[i]
        
        # Meal Effect (Spike starts 4 steps later, lasts 2 hours)
        if df.loc[i, 'meal_carbs'] > 0:
            meal_carbs = df.loc[i, 'meal_carbs']
            for j in range(4, 24): # 4 steps (20 min delay) to 24 steps (2 hours)
                if i + j < num_samples:
                    spike_magnitude = (meal_carbs / 5.0) * (j / 24.0) 
                    bg_data.iloc[i + j] += spike_magnitude * (1 - j / 24.0) # Spike decays over time
                    
        # Activity Effect (Drop starts immediately, lasts 1 hour)
        if df.loc[i, 'activity_steps'] > 0:
            for j in range(1, 12): # 12 steps = 1 hour
                if i + j < num_samples:
                    bg_data.iloc[i + j] -= 2 # Constant drop during activity
                    
    df['bg_cgm'] = bg_data.round(1)
    
    # Apply reasonable BG limits
    df['bg_cgm'] = df['bg_cgm'].clip(lower=60, upper=300)

    return df

def generate_ohio_t1dm_simulated_csv():
    """Generates data for all patients and saves the final CSV."""
    
    all_data = []
    for patient_id, (age, bmi, basal_rate) in PATIENT_CONFIG.items():
        print(f"Generating data for Patient ID: {patient_id}")
        df = generate_patient_data(patient_id, age, bmi, basal_rate)
        all_data.append(df)
        
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure the 'data' directory exists
    os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)
    
    # Save the CSV
    final_df.to_csv(FILE_PATH, index=False)
    print(f"\nSuccessfully created and saved data to: {FILE_PATH}")
    print(f"Total samples generated: {len(final_df)}")

if __name__ == '__main__':
    generate_ohio_t1dm_simulated_csv()