---
title: Snow Predictor Basel
emoji: 🌨️
colorFrom: blue
colorTo: white
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
---

# 🌨️ Snow Predictor Basel - My First ML Model! 🚀

Welcome to my first machine learning project! This repository contains a **7-day ahead snow prediction model** for Basel, Switzerland that I built from scratch during my Python learning journey.

## 📚 Repository Information

**This project is available in multiple locations:**

- **🌨️ Hugging Face Model:** [https://huggingface.co/Tuminha/snow-predictor-basel](https://huggingface.co/Tuminha/snow-predictor-basel)
- **🐙 GitHub Repository:** [https://github.com/Tuminha/snow_in_basel](https://github.com/Tuminha/snow_in_basel)

**Choose your preferred platform:**
- **Hugging Face:** For downloading the trained model and viewing documentation
- **GitHub:** For source code, development, and contributing to the project

## 🎯 What This Model Does

**Predicts snow in Basel 7 days in advance** using weather data patterns. Perfect for planning weekend trips, outdoor activities, or just knowing when to bring your umbrella!

## 🏆 Model Performance

After training on **25 years of Basel weather data**, here's how well it performs:

- **🎯 Accuracy:** 77.4% - Overall prediction accuracy
- **❄️ Recall:** 84.0% - Catches most snow events (prioritizes safety!)
- **⚠️ Precision:** 16.4% - Some false alarms, but better than missing snow
- **🔥 ROC AUC:** 89.4% - Excellent model discrimination

## 🚀 Key Features

- **⏰ 7-day ahead prediction** - Plan your week with confidence
- **🌡️ 22 weather features** - Temperature trends, precipitation patterns, seasonal indicators
- **🛡️ High recall design** - Built to catch snow events rather than avoid false alarms
- **📊 25 years of data** - Trained on comprehensive Basel weather history (2000-2025)

## 🏗️ How I Built This

### **Data Collection & Processing**
- **Source:** Meteostat API for real Basel weather data
- **Location:** Basel, Switzerland (47.5584° N, 7.5733° E)
- **Processing:** Handled missing values, temperature inconsistencies, and date gaps
- **Features:** Engineered rolling weather patterns, seasonal indicators, and volatility measures

### **Model Architecture**
- **Algorithm:** Logistic Regression (chosen for interpretability and reliability)
- **Training:** 80% of data for training, 20% for testing
- **Class Balancing:** Used balanced class weights to handle snow/no-snow imbalance
- **Feature Scaling:** Standardized all features for optimal performance

### **Feature Engineering**
The model uses sophisticated weather patterns:
- **Temperature trends** over 7-day windows
- **Precipitation accumulation** patterns
- **Atmospheric pressure** changes
- **Seasonal indicators** and day-of-year patterns
- **Weather volatility** measures

## 🔧 How to Use This Model

### **Quick Start**
