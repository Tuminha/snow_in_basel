# Test script to verify WeatherData class works
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from data.weather_data import WeatherData

# Test 1: Create WeatherData instance
try:
    weather = WeatherData()
    print("✅ WeatherData class created successfully!")
    print(f"Start date: {weather.start_date}")
    print(f"End date: {weather.end_date}")
    print(f"Weather parameters: {weather.weather_parameters}")
except Exception as e:
    print(f"❌ Error creating WeatherData: {e}")

# Test 3: Fetch actual weather dat# Test 3: Fetch actual weather data
try:
    print("\n🌨️ Testing weather data collection...")
    weather_data = weather.get_weather_data()
    
    if weather_data is not None:
        print("✅ Weather data fetched successfully!")
        print(f"Data shape: {weather_data.shape}")
        print(f"Columns: {list(weather_data.columns)}")
        print(f"Date range: {weather_data.index.min()} to {weather_data.index.max()}")
        print("\nFirst few rows:")
        print(weather_data.head())
    else:
        print("❌ No weather data returned")
        
except Exception as e:
    print(f"❌ Error testing weather data: {e}")

# Test 4: Test data cleaning
try:
    print("\n🧹 Testing data cleaning...")
    cleaned_data = weather.data_cleaning()
    
    if cleaned_data is not None:
        print("✅ Data cleaning successful!")
        print(f"Cleaned data shape: {cleaned_data.shape}")
        print(f"New columns: {list(cleaned_data.columns)}")
        print(f"Sample seasons: {cleaned_data['season'].value_counts().head()}")
        print(f"Temperature range stats: {cleaned_data['temp_range'].describe()}")
        
        # Add validation checks here
        print("\n🔍 Data Quality Validation:")
        
        # 1. Snow data integrity
        print("Snow data sample:")
        print(cleaned_data['snow'].value_counts().head(10))
        
        # 2. Temperature consistency
        print("\nTemperature validation:")
        temp_consistent = (cleaned_data['tmin'] <= cleaned_data['tavg']).all() and (cleaned_data['tavg'] <= cleaned_data['tmax']).all()
        print(f"tmin <= tavg <= tmax: {temp_consistent}")
        
        # 3. Date continuity
        print("\nDate gaps:")
        date_diff = cleaned_data.index.to_series().diff()
        print(f"Max gap: {date_diff.max()}")
        # Find problematic temperature rows
        print("\n🚨 Problematic Temperature Rows:")
        temp_issues = cleaned_data[
            (cleaned_data['tmin'] > cleaned_data['tavg']) | 
            (cleaned_data['tavg'] > cleaned_data['tmax'])
        ]
        print(f"Found {len(temp_issues)} rows with temperature inconsistencies")
        print(temp_issues[['tmin', 'tavg', 'tmax']].head(10))
        # Check for missing dates
        print("\n📅 Date Continuity Check:")
        expected_dates = pd.date_range(start=cleaned_data.index.min(), end=cleaned_data.index.max(), freq='D')
        missing_dates = expected_dates.difference(cleaned_data.index)
        print(f"Missing dates: {len(missing_dates)}")
        if len(missing_dates) > 0:
            print(f"Sample missing dates: {missing_dates[:5].tolist()}")
        
    else:
        print("❌ Data cleaning failed")
        
except Exception as e:
    print(f"❌ Error in data cleaning: {e}")

# Test 5: Test model preparation
try:
    print("\n🤖 Testing model preparation...")
    model_data = weather.prepare_for_modeling()
    
    if model_data is not None:
        print("✅ Model preparation successful!")
        print(f"Final data shape: {model_data.shape}")
        print(f"Target variable distribution:")
        print(model_data['snow_binary_7d'].value_counts()) 
        print(f"Features: {list(model_data.columns)}")
    else:
        print("❌ Model preparation failed")
        
except Exception as e:
    print(f"❌ Error in model preparation: {e}")

# Test 6: Test snow prediction model
try:
    print("\n🌨️ Testing snow prediction model...")
    model_results = weather.build_snow_model()
    
    if model_results is not None:
        print("✅ Snow prediction model built successfully!")
        print(f"Model type: {type(model_results['model']).__name__}")
        print(f"Features used: {len(model_results['feature_names'])}")
        
        # Show feature importance
        print("\n🔍 Top 5 Most Important Features for Snow Prediction:")
        sorted_features = sorted(model_results['feature_importance'].items(), 
                               key=lambda x: abs(x[1]), reverse=True)
        for feature, importance in sorted_features[:5]:
            print(f"  {feature}: {importance:.4f}")
            
        # Show prediction distribution
        print(f"\n📊 Prediction Results:")
        print(f"  Test set size: {len(model_results['y_test'])}")
        print(f"  Snow predictions (7-day ahead): {sum(model_results['y_pred'])}")
        print(f"  No-snow predictions: {len(model_results['y_pred']) - sum(model_results['y_pred'])}")
        
        # Show 7-day ahead prediction target distribution
        print(f"\n📈 Target Variable Distribution (7-day ahead snow):")
        target_counts = model_results['y_test'].value_counts()
        print(f"  Days with snow (7 days later): {target_counts.get(1, 0)}")
        print(f"  Days without snow (7 days later): {target_counts.get(0, 0)}")
        
    else:
        print("❌ Snow prediction model failed")
        
except Exception as e:
    print(f"❌ Error in snow prediction model: {e}")

# Test 7: Evaluate model performance
try:
    print("\n📊 Evaluating Model Performance...")
    
    if hasattr(weather, 'model_results') and weather.model_results is not None:
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        import numpy as np
        
        # Get model results
        y_test = weather.model_results['y_test']
        y_pred = weather.model_results['y_pred']
        y_pred_proba = weather.model_results['y_pred_proba']
        
        # 1. Confusion Matrix
        print("\n🔍 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print("                Predicted")
        print("                No Snow  Snow")
        print(f"Actual No Snow   {cm[0,0]:>6}  {cm[0,1]:>4}")
        print(f"Actual Snow      {cm[1,0]:>6}  {cm[1,1]:>4}")
        
        # 2. Classification Report
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Snow', 'Snow']))
        
        # 3. Key Metrics
        print("\n🎯 Key Performance Metrics:")
        
        # Calculate metrics manually for better understanding
        tp = cm[1, 1]  # True Positives: Correctly predicted snow
        tn = cm[0, 0]  # True Negatives: Correctly predicted no snow
        fp = cm[0, 1]  # False Positives: Predicted snow but no snow
        fn = cm[1, 0]  # False Negatives: Predicted no snow but snow
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"  Precision: {precision:.3f} ({precision*100:.1f}%)")
        print(f"  Recall:    {recall:.3f} ({recall*100:.1f}%)")
        print(f"  F1-Score:  {f1_score:.3f}")
        
        # 4. ROC AUC Score
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"  ROC AUC:   {roc_auc:.3f}")
        except:
            print("  ROC AUC:   Could not calculate")
        
        # 5. Snow Prediction Analysis
        print(f"\n🌨️ Snow Prediction Analysis:")
        print(f"  Total snow days in test set: {sum(y_test)}")
        print(f"  Correctly predicted snow: {tp}")
        print(f"  Missed snow days: {fn}")
        print(f"  False snow alarms: {fp}")
        
        # 6. Model Confidence
        print(f"\n💪 Model Confidence:")
        snow_proba = y_pred_proba[y_pred == 1]
        no_snow_proba = y_pred_proba[y_pred == 0]
        if len(snow_proba) > 0:
            print(f"  Average confidence for snow predictions: {np.mean(snow_proba):.3f}")
        if len(no_snow_proba) > 0:
            print(f"  Average confidence for no-snow predictions: {np.mean(no_snow_proba):.3f}")
        
    else:
        print("❌ No model results available for evaluation")
        
except Exception as e:
    print(f"❌ Error in model evaluation: {e}")

# Test 8: Test model saving and loading
try:
    print("\n💾 Testing model saving and loading...")
    
    if hasattr(weather, 'model_results') and weather.model_results is not None:
        # Save the model
        saved_path = weather.save_model()
        print(f"✅ Model saved to: {saved_path}")
        
        # Load the model
        loaded_model = weather.load_model(saved_path)
        if loaded_model is not None:
            print("✅ Model loading successful!")
            print(f"Loaded features: {len(loaded_model['feature_names'])}")
        else:
            print("❌ Model loading failed")
    else:
        print("❌ No model to save - train the model first!")
        
except Exception as e:
    print(f"❌ Error in model saving/loading: {e}")