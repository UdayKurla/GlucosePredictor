// --- Global State ---
const USER_ID = 1; // Simulated user ID for personalization
const API_URL = "http://127.0.0.1:8000"; 
let glucoseChart; // Chart.js object
let chartData = {
    labels: [],
    actual: [],
    predicted: []
};

// --- 1. Real-Time Chart Initialization ---
function initChart() {
    const ctx = document.getElementById('glucoseChart').getContext('2d');
    
    // Initialize labels for a 1-hour window (11 readings) + prediction point (t+30)
    chartData.labels = Array.from({ length: 12 }, (_, i) => `t-${(11-i)*5}`);
    chartData.labels[11] = 't+30'; // Label the prediction horizon

    glucoseChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: [
                {
                    label: 'BG (mg/dL)',
                    data: chartData.actual,
                    borderColor: '#1e88e5',
                    tension: 0.2,
                    borderWidth: 3,
                    pointRadius: 3,
                    // Use a dashed line for the predicted part to clearly separate it
                    segment: {
                        borderColor: ctx => ctx.p0DataIndex === chartData.actual.length - 2 ? '#00c853' : '#1e88e5', // Color for the line segment leading to prediction
                        borderDash: ctx => ctx.p0DataIndex === chartData.actual.length - 2 ? [6, 6] : []
                    }
                },
            ]
        },
        options: {
            animation: false, // Turn off animation for real-time feel
            scales: {
                y: {
                    title: { display: true, text: 'Glucose (mg/dL)' },
                    min: 60,
                    max: 200
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// --- 2. WebSocket Nudge Engine ---
function connectWebSocket() {
    // Note: Use 'ws' for http and 'wss' for https
    const socket = new WebSocket(`ws://127.0.0.1:8000/ws/${USER_ID}`);
    const alertBox = document.getElementById('alert-box');
    const bgReading = document.getElementById('bg-reading');
    
    socket.onopen = () => {
        console.log("WebSocket connected.");
        alertBox.textContent = "✅ Real-Time Connection Established.";
        alertBox.className = 'panel';
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Handle Alerts/Nudges
        if (data.type === 'alert' || data.type === 'info') {
            alertBox.innerHTML = data.message;
            if (data.message.includes('High spike')) {
                alertBox.className = 'panel high-risk';
            } else {
                alertBox.className = 'panel';
            }
        }
        
        // Handle Data Updates (for Charting)
        if (data.type === 'data') {
            bgReading.textContent = `${data.current_bg} mg/dL`;
            
            // --- Update Chart Data ---
            chartData.actual.push(data.current_bg);
            
            // Keep the chart window size constant (11 historical points)
            if (chartData.actual.length > 11) {
                chartData.actual.shift();
            }

            // The predicted point is plotted at the last index ('t+30')
            let plotData = [...chartData.actual];
            plotData[plotData.length] = data.predicted_bg; 

            glucoseChart.data.datasets[0].data = plotData;
            glucoseChart.update();
        }
    };

    socket.onclose = () => {
        console.log('WebSocket disconnected. Attempting to reconnect...');
        alertBox.textContent = "❌ Connection lost. Attempting to reconnect...";
        setTimeout(connectWebSocket, 5000); 
    };
    
    socket.onerror = (error) => {
        console.error('WebSocket Error:', error);
    };
}


// --- 3. Meal Recommender Function ---
async function getMealRecommendation() {
    const query = document.getElementById('meal-query').value.trim();
    const resultElement = document.getElementById('recommendation-result');
    resultElement.innerHTML = "Searching...";

    if (!query) {
        resultElement.textContent = "Please enter a meal name to search.";
        return;
    }

    try {
        const response = await fetch(`${API_URL}/api/recommend_meal`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: USER_ID, query: query })
        });

        const data = await response.json();
        // Use innerHTML to render bold markdown from the backend
        resultElement.innerHTML = data.recommendation;
        
    } catch (error) {
        console.error('Error fetching recommendation:', error);
        resultElement.textContent = "Error fetching meal recommendation.";
    }
}


// --- Execute on page load ---
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    connectWebSocket();
});