# Change the last line from:
# app.run(debug=True, port=5000)
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime, timedelta
from meteostat import Point, Daily
import numpy as np
import os  # ← ADD THIS LINE!

app = Flask(__name__)
CORS(app)

# To:
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)