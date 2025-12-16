from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import json
import time
import asyncio
import numpy as np
from model_service import prediction_service

# --- 1. FastAPI Setup ---
app = FastAPI()

# Mount the static directory (frontend files)
# The path is relative to where main.py is executed
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# --- 2. Data Models (Pydantic) ---
class PredictionRequest(BaseModel):
    user_id: int
    current_bg: float
    time_since_meal: float = 0.0
    current_activity: bool = False

class MealSearch(BaseModel):
    user_id: int
    query: str

# --- 3. Dummy Meal Data for Hyper-Personalization ---
# Simulates a database table: {meal_tag: [(user_id, max_delta_bg)]}
PERSONALIZED_MEAL_DATA = {
    "pasta": [
        (1, 45.5),  # User 1: High spike with pasta (45.5 mg/dL)
        (2, 22.0),  # User 2: Medium spike with pasta (22.0 mg/dL)
    ],
    "chicken salad": [
        (1, 10.2),  # User 1: Low spike with salad
        (2, 15.0),
    ],
    "pizza": [
        (1, 55.0),
    ]
}

# --- 4. API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves the main HTML dashboard file."""
    # Ensure this path is correct relative to the main.py execution location
    with open("../frontend/index.html", 'r') as f:
        return f.read()

@app.post("/api/predict", tags=["Prediction"])
async def get_glucose_prediction(data: PredictionRequest):
    """Predicts future BG and returns risk level."""
    predicted_bg, risk_level, risk_message = prediction_service.get_prediction(
        data.current_bg,
        data.time_since_meal,
        data.current_activity
    )
    
    return {
        "predicted_bg_t+30": round(predicted_bg, 2),
        "risk_level": risk_level,
        "risk_message": risk_message
    }

@app.post("/api/recommend_meal", tags=["Recommendation"])
async def recommend_meal(data: MealSearch):
    """Hyper-personalized meal recommender logic."""
    
    query_lower = data.query.lower()
    
    if query_lower in PERSONALIZED_MEAL_DATA:
        historical_spikes = PERSONALIZED_MEAL_DATA[query_lower]
        
        # 1. Filter for the specific user (Hyper-Personalization)
        user_spikes = [
            (spike, user) for user, spike in historical_spikes 
            if user == data.user_id
        ]
        
        # 2. Rank based on lowest spike (Min Max_Delta_BG)
        if user_spikes:
            best_spike = user_spikes[0][0] # Since there's only one entry per meal per user in this dummy data
            
            # Calculate average spike across ALL users for comparison
            all_spikes = [spike for spike, user in historical_spikes]
            avg_spike = np.mean(all_spikes)
            
            recommendation = f"Meal: **{data.query.capitalize()}**. "
            
            if best_spike <= avg_spike * 1.1: # If personal spike is near or below average
                 recommendation += f"Your historical spike ({best_spike:.1f} mg/dL) is acceptable. Recommended."
            else:
                 recommendation += f"⚠️ Your historical spike ({best_spike:.1f} mg/dL) is **significantly higher** than the group average ({avg_spike:.1f} mg/dL). Consider a modification."

        else:
            # Fallback logic
            recommendation = f"No personal history found for '{data.query}'. Average spike for others: {np.mean([spike for spike, user in historical_spikes]):.1f} mg/dL. Proceed with caution."
            
        return {"recommendation": recommendation}
        
    return {"recommendation": f"Sorry, no specific meal data for '{data.query}' yet. Try 'pasta' or 'chicken salad'."}


# --- 5. Real-Time Nudge Engine (WebSockets) ---

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    await websocket.accept()
    print(f"User {user_id} connected to WebSocket.")
    
    try:
        current_bg = 120.0 # Starting BG
        while True:
            # 1. Get Prediction
            predicted_bg, risk_level, risk_message = prediction_service.get_prediction(
                current_bg, time_since_meal=1.0, current_activity=False
            )
            
            # 2. Check for HIGH Risk (The Nudge)
            if risk_level == "HIGH":
                alert_message = f"🚨 NUDGE: High spike predicted! Your BG could reach {predicted_bg:.0f} in 30 min. Intervention: 15 min light walk!"
                await websocket.send_json({"type": "alert", "message": alert_message})
                
            elif risk_level == "MEDIUM":
                 await websocket.send_json({"type": "info", "message": f"🔺 Predicted BG: {predicted_bg:.0f} mg/dL. Watch closely."})
                 
            # 3. Simulate new BG reading (for the next cycle)
            current_bg += np.random.uniform(-5, 10) # Simple BG drift simulation
            current_bg = max(80.0, min(current_bg, 250.0)) # Clamp BG
            
            # Send current/predicted data for charting
            await websocket.send_json({
                "type": "data",
                "current_bg": round(current_bg, 1),
                "predicted_bg": round(predicted_bg, 1)
            })

            await asyncio.sleep(5) # Push data every 5 seconds
            
    except WebSocketDisconnect:
        print(f"User {user_id} disconnected.")
    except Exception as e:
        print(f"Error in WebSocket: {e}")