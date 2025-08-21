# Project settings
# File paths
DATA_DIR = "data"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"
NOTEBOOKS_DIR = "notebooks"
PIPELINE_DIR = "pipeline"
TEST_DIR = "test"
UTILS_DIR = "utils"

# Specific file paths
WEATHER_DATA_FILE = "data/weather_data.py"
WEATHER_MODEL_FILE = "models/weather_model.py"
PREDICTION_PIPELINE_FILE = "pipeline/prediction_pipeline.py"
MAIN_FILE = "main.py"
README_FILE = "README.md"
REQUIREMENTS_FILE = "requirements.txt"

# API keys

# Basel coordinates
BASEL_LATITUDE = 47.5596
BASEL_LONGITUDE = 7.5886

# Model parameters
# Training settings

# Not prediction thresholds


# Data sources
# Meteostat configuration
METEOSTAT_CONFIG = {
    'start_date': '2000-01-01',  # Start collecting from 2000
    'end_date': '2025-12-31',    # Collect up to current year
    'update_frequency': 'daily',  # How often to update data
    'weather_parameters': [
        'temp',      # Surface temperature
        'tmax',      # Maximum temperature
        'tmin',      # Minimum temperature
        'dwpt',      # Dew point temperature
        'rhum',      # Relative humidity
        'prcp',      # Precipitation amount
        'snow',      # Snow depth
        'pres',      # Atmospheric pressure
        'wspd',      # Wind speed
        'wdir',      # Wind direction
        'wpgt'       # Peak wind gust
    ]  # Comprehensive weather parameters for snow prediction
}

# Data collection settings
DATA_COLLECTION = {
    'min_data_quality': 0.8,     # Minimum data quality threshold
    'max_missing_days': 30,      # Maximum consecutive missing days
    'refresh_interval': '1d'     # Refresh data every day
}




