import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from feature_engineering import prepare_data_for_lstm # Import data preparation function

# --- Configuration ---
BACKEND_MODEL_PATH = 'fsd_module/backend/prediction_model.h5'
TIMESTEPS = 1 # We use a single input timestep of all combined features

def reshape_data(X, y):
    """Reshapes 2D feature array into 3D LSTM input shape (samples, timesteps, features)."""
    
    num_features = X.shape[1]
    # Reshape the data for the LSTM: (samples, timesteps, features)
    X_reshaped = X.reshape((X.shape[0], TIMESTEPS, num_features))
    
    # --- Scaling ---
    # Normalization of features is critical for Deep Learning stability
    scaler = StandardScaler()
    # Scale X data: requires fitting on a 2D view and then reshaping back to 3D
    X_scaled = scaler.fit_transform(X_reshaped.reshape(-1, num_features)).reshape(X_reshaped.shape)
    
    # Scale Y data (target)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    print(f"Reshaped X shape for LSTM: {X_scaled.shape}")
    return X_scaled, y_scaled, scaler, y_scaler

def build_lstm_model(input_shape):
    """Defines the Bi-LSTM model architecture."""
    # The architecture is designed for sequential data captured by the feature engineering
    model = Sequential([
        # Bi-LSTM for better context capture in time-series
        Bidirectional(LSTM(units=100, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(units=50)),
        Dropout(0.3),
        Dense(units=1) # Output: single predicted BG value (t+30)
    ])
    
    # Using 'mae' (Mean Absolute Error) loss, as it directly relates to the project's numerical goal (~6.5 mg/dL)
    model.compile(optimizer='adam', loss='mae', metrics=['mae', 'mse'])
    return model

def train_and_save_model():
    """Main function to run the DA module pipeline."""
    
    # 1. Prepare Data
    X, y, meta = prepare_data_for_lstm()
    
    if X is None:
        return
        
    # 2. Reshape and Scale
    X_scaled, y_scaled, X_scaler, y_scaler = reshape_data(X, y)

    # 3. Train-Validation Split (80/20 split, using shuffle=False for time series)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, shuffle=False
    )

    # 4. Build Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    model.summary()

    # 5. Define Callbacks
    # Early Stopping prevents overfitting and ModelCheckpoint saves the best model version
    callbacks = [
        EarlyStopping(patience=10, monitor='val_mae', mode='min', restore_best_weights=True),
        ModelCheckpoint(BACKEND_MODEL_PATH, save_best_only=True, monitor='val_mae', mode='min')
    ]

    # 6. Training
    print("\nStarting model training...")
    # NOTE: Training will take some time with 16k samples and 100 epochs.
    history = model.fit(
        X_train, y_train,
        epochs=100, 
        batch_size=32, 
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # 7. Final Confirmation
    print(f"\nModel training complete. Best model saved to: {BACKEND_MODEL_PATH}")

if __name__ == '__main__':
    train_and_save_model()