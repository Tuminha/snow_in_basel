import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import METEOSTAT_CONFIG, BASEL_LATITUDE, BASEL_LONGITUDE
from meteostat import Point, Daily
from datetime import datetime




class WeatherData:
    def __init__(self, start_date=None, end_date=None, update_frequency=None, weather_parameters=None):
        # Use config values as defaults if not provided
        self.start_date = start_date or METEOSTAT_CONFIG['start_date']
        self.end_date = end_date or METEOSTAT_CONFIG['end_date']
        self.update_frequency = update_frequency or METEOSTAT_CONFIG['update_frequency']
        self.weather_parameters = weather_parameters or METEOSTAT_CONFIG['weather_parameters']
        

    def get_weather_data(self):
        # Fetch real data from Meteostat
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')
        try:
            basel = Point(BASEL_LATITUDE, BASEL_LONGITUDE)
            data = Daily(basel, start=start_dt, end=end_dt)
            df = data.fetch()
            return df
        except Exception as e:
            print(f"❌ Error fetching weather data: {e}")
            return None

    def data_cleaning(self):
        # Handle missing values(<NA> or NaN), convert data types and add derived features like temperatures ranges, seasonal indicators
        df = self.get_weather_data()
        if df is not None:
            # Convert data types for available columns from Meteostat
            if 'tavg' in df.columns:
                df['tavg'] = df['tavg'].astype(float)
            if 'tmax' in df.columns:
                df['tmax'] = df['tmax'].astype(float)
            if 'tmin' in df.columns:
                df['tmin'] = df['tmin'].astype(float)
            if 'prcp' in df.columns:
                df['prcp'] = df['prcp'].astype(float)
            if 'snow' in df.columns:
                df['snow'] = df['snow'].astype(float)
            if 'wdir' in df.columns:
                df['wdir'] = df['wdir'].astype(float)
            if 'wspd' in df.columns:
                df['wspd'] = df['wspd'].astype(float)
            if 'wpgt' in df.columns:
                df['wpgt'] = df['wpgt'].astype(float)
            if 'pres' in df.columns:
                df['pres'] = df['pres'].astype(float)
            if 'tsun' in df.columns:
                df['tsun'] = df['tsun'].astype(float)

            # Add derived features
            if 'tmax' in df.columns and 'tmin' in df.columns:
                df['temp_range'] = df['tmax'] - df['tmin']
            df['season'] = df.index.month.map(self.get_season)

            # Handle missing values
            # Only drop rows where critical columns are missing
            critical_columns = ['tavg', 'tmin', 'tmax', 'prcp', 'snow']
            df = df.dropna(subset=critical_columns)

            # Fill other missing values with reasonable defaults
            df['wdir'] = df['wdir'].fillna(0)  # Wind direction
            df['wpgt'] = df['wpgt'].fillna(0)  # Wind gust
            # Filter out temperature inconsistencies
            temp_consistent = (df['tmin'] <= df['tavg']) & (df['tavg'] <= df['tmax'])
            df = df[temp_consistent]
            print(f"Removed {len(temp_consistent) - temp_consistent.sum()} rows with temperature inconsistencies")

            return df
        else:
            print("❌ No weather data to clean")
            return None
        
   

    def get_season(self, month):
        # Define seasons based on Swiss calendar
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def prepare_for_modeling(self):
        """
        Prepare cleaned weather data for snow prediction modeling with 7-day ahead prediction horizon
        """
        # Get cleaned data
        df = self.data_cleaning()
        
        if df is None:
            return None
        
        # Handle date gaps with forward-fill (reasonable for weather data)
        df = df.resample('D').ffill()
        
        # Create snow target variable shifted 7 days into the future (predict snow 7 days ahead)
        df['snow_binary_7d'] = (df['snow'] > 0).shift(-7).astype('Int64')
        
        # Current weather features that might predict future snow
        df['temp_below_freezing'] = (df['tavg'] < 0).astype(int)
        df['high_precipitation'] = (df['prcp'] > 5).astype(int)
        df['windy_day'] = (df['wspd'] > 15).astype(int)
        
        # Seasonal and weather pattern features for 7-day ahead prediction
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        df['is_winter_season'] = df['season'].apply(lambda x: 1 if x == 'winter' else 0)
        
        # Rolling weather patterns (7-day windows leading up to prediction)
        df['temp_trend_7d'] = df['tavg'].rolling(window=7, min_periods=1).mean()
        df['temp_std_7d'] = df['tavg'].rolling(window=7, min_periods=1).std().fillna(0)
        df['precip_sum_7d'] = df['prcp'].rolling(window=7, min_periods=1).sum()
        df['pressure_trend_7d'] = df['pres'].rolling(window=7, min_periods=1).mean()
        df['cold_days_7d'] = df['temp_below_freezing'].rolling(window=7, min_periods=1).sum()
        
        # Weather volatility indicators
        df['temp_volatility'] = df['temp_range'].rolling(window=7, min_periods=1).std().fillna(0)
        df['pressure_change'] = df['pres'].diff().fillna(0)
        df['temp_drop_rate'] = df['tavg'].diff().fillna(0)
        
        # Remove rows where we can't predict (last 7 days)
        df = df.dropna(subset=['snow_binary_7d'])
        
        # Fill any remaining NaN values with reasonable defaults
        df = df.fillna(0)
        
        return df

    def build_snow_model(self):
        """
        Build and train a logistic regression model for 7-day ahead snow prediction
        """
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np
        
        # Get prepared data
        df = self.prepare_for_modeling()
        if df is None:
            return None
        
        # Define features and target (updated for 7-day ahead prediction)
        feature_columns = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'wpgt', 'pres', 
                        'temp_range', 'temp_below_freezing', 'high_precipitation', 
                        'windy_day', 'month', 'day_of_year', 'is_winter_season',
                        'temp_trend_7d', 'temp_std_7d', 'precip_sum_7d', 
                        'pressure_trend_7d', 'cold_days_7d', 'temp_volatility', 
                        'pressure_change', 'temp_drop_rate']
        
        X = df[feature_columns]
        y = df['snow_binary_7d']  # Updated target variable
        
        # Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features (important for logistic regression)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression with class weights (handle imbalance)
        model = LogisticRegression(
            random_state=42, 
            class_weight='balanced',  # Handle class imbalance
            max_iter=1000
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Store results
        self.model_results = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_columns,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_importance': dict(zip(feature_columns, model.coef_[0]))
        }
        
        return self.model_results

    def save_model(self, filepath='models/snow_predictor.joblib'):
        """
        Save the trained model using joblib (better than pickle)
        """
        import joblib
        import os
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model and scaler
        model_data = {
            'model': self.model_results['model'],
            'scaler': self.model_results['scaler'],
            'feature_names': self.model_results['feature_names'],
            'model_info': {
                'accuracy': 0.774,  # Your actual accuracy
                'recall': 0.84,     # Your actual recall
                'precision': 0.164, # Your actual precision
                'f1_score': 0.274, # Your actual f1-score
                'roc_auc': 0.894,  # Your actual ROC AUC
                'training_date': datetime.now().strftime('%Y-%m-%d'),
                'data_range': f"{self.start_date} to {self.end_date}",
                'features_count': len(self.model_results['feature_names']),
                'prediction_horizon': '7 days ahead'
            }
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
        return filepath

    def load_model(self, filepath='models/snow_predictor.joblib'):
        """
        Load a previously trained model
        """
        import joblib
        
        if not os.path.exists(filepath):
            print(f"❌ Model file not found: {filepath}")
            return None
        
        model_data = joblib.load(filepath)
        self.model_results = model_data
        print(f"✅ Model loaded from {filepath}")
        return model_data
        
