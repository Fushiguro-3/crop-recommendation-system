# 🌱 AI-Based Crop Recommendation System

A machine learning web application that recommends the most suitable crop based on soil and climate parameters.

## Features
- **Crop Prediction** — recommends best crop from 7 soil inputs with 99.55% accuracy
- **Soil Gap Analysis** — tells farmer what amendments needed to grow a desired crop

## Tech Stack
- Python, Flask, scikit-learn, Random Forest
- HTML5, CSS3, JavaScript ES6
- joblib, pandas, numpy

## How to Run
```bash
pip install flask scikit-learn joblib numpy pandas
python train_baseline.py   # generates model files
python app.py              # starts web server
```
Open http://127.0.0.1:5000

## Dataset
Crop_recommendation.csv — 2,200 records, 22 crop types, 7 features (N, P, K, temperature, humidity, pH, rainfall)

## Model Performance
- Algorithm: Random Forest (200 trees)
- Test Accuracy: 99.55%
- 5-Fold CV: 99.1% ± 0.4%

## Team
- Kiran Aravadi
- Nisarg Shettar  
- Sanjana Shidramani
- Vandana Guruwodeyar

**Guide:** Prof. Suman Yaligar  
**Dept:** AI & ML, S.D.M. College of Engineering & Technology, Dharwad  
**Year:** 2025–26