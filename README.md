# 🌱 CropSense — AI-Based Crop Recommendation System

A machine learning web application that recommends the most suitable crop based on soil and climate parameters, with a unique Soil Gap Analysis feature that tells farmers exactly what amendments are needed to grow their preferred crop.

## Features
- **Crop Prediction** — recommends best crop from 7 soil & climate inputs with 99.55% accuracy
- **Soil Gap Analysis** — compares current soil against ideal conditions for any desired crop
- **Amendment Advice** — specific fertilizer recommendations (Urea, SSP, MOP, Agricultural Lime)
- **Top 3 Alternatives** — ranked crop matches with confidence scores
- **Confidence Score** — model probability shown for every prediction

## Tech Stack
- **Backend:** Python, Flask
- **ML:** scikit-learn, Random Forest, XGBoost, SHAP
- **Data:** pandas, NumPy, joblib
- **Frontend:** HTML5, CSS3, JavaScript ES6

## How to Run
```bash
pip install flask scikit-learn joblib numpy pandas
python train_baseline.py   # trains model and saves pkl files
python app.py              # starts web server
```
Open `http://127.0.0.1:5000`

## Dataset
`Crop_recommendation.csv` — 2,200 records, 22 crop types, perfectly balanced (100 samples each)

**Input features (7):** Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall  
**Output:** Recommended crop name

## Model Performance
| Metric | Value |
|---|---|
| Algorithm | Random Forest (200 trees) |
| Test Accuracy | 99.55% |
| 5-Fold CV | 99.1% ± 0.4% |
| Classes | 22 crop types |

## Project Structure
crop_website/
├── app.py                  ← Flask backend
├── train_baseline.py       ← Model training script
├── Crop_recommendation.csv ← Dataset
├── requirements.txt        ← Dependencies
└── templates/
└── index.html          ← Frontend

## Screenshot
<img width="511" height="655" alt="image" src="https://github.com/user-attachments/assets/06e540c6-cde7-410c-b1b8-60d10ce3f44b" />


## Note
This was developed as a minor project for academic purposes.

