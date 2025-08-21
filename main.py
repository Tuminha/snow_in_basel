from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime, timedelta
from meteostat import Point, Daily
import numpy as np
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Snow Predictor API is running!"

@app.route('/test')
def test():
    return {"status": "ok", "message": "API is working"}

@app.route('/api/predict-snow', methods=['GET'])
def predict_snow():
    try:
        # Your existing code...
        return jsonify({
            "prediction": "test",
            "probability": 0.5,
            "confidence": "medium",
            "weather_summary": "Test response",
            "prediction_date": "21/08/2025",
            "forecast_date": "28/08/2025"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)