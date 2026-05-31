# from flask import Flask, render_template, request, jsonify
# import joblib
# import numpy as np
# import os

# app = Flask(__name__)

# # Load model and preprocessors
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model   = joblib.load(os.path.join(BASE_DIR, "best_model_rf.pkl"))
# scaler  = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
# le      = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

# # Crop info: ideal conditions + emoji + description
# CROP_INFO = {
#     "rice":        {"emoji": "🌾", "desc": "Thrives in waterlogged, warm conditions with high humidity.", "season": "Kharif"},
#     "maize":       {"emoji": "🌽", "desc": "Grows well in well-drained loamy soil with moderate rainfall.", "season": "Kharif"},
#     "chickpea":    {"emoji": "🫘", "desc": "Prefers cool, dry climate and well-drained sandy loam soil.", "season": "Rabi"},
#     "kidneybeans": {"emoji": "🫘", "desc": "Needs warm temperatures and well-drained fertile soil.", "season": "Kharif"},
#     "pigeonpeas":  {"emoji": "🌿", "desc": "Drought-tolerant, grows in semi-arid regions with sandy soil.", "season": "Kharif"},
#     "mothbeans":   {"emoji": "🌱", "desc": "Very drought-resistant, ideal for arid sandy soils.", "season": "Kharif"},
#     "mungbean":    {"emoji": "🌱", "desc": "Short-duration crop, grows in warm humid conditions.", "season": "Kharif"},
#     "blackgram":   {"emoji": "🫘", "desc": "Grows in tropical climate with moderate to low rainfall.", "season": "Kharif"},
#     "lentil":      {"emoji": "🫘", "desc": "Cool-season crop, prefers loamy soil with good drainage.", "season": "Rabi"},
#     "pomegranate": {"emoji": "🍎", "desc": "Drought-tolerant fruit, thrives in semi-arid warm regions.", "season": "Perennial"},
#     "banana":      {"emoji": "🍌", "desc": "Needs tropical climate, high humidity and rich loamy soil.", "season": "Perennial"},
#     "mango":       {"emoji": "🥭", "desc": "Prefers tropical climate with dry spell before flowering.", "season": "Perennial"},
#     "grapes":      {"emoji": "🍇", "desc": "Thrives in well-drained sandy loam with hot dry summers.", "season": "Perennial"},
#     "watermelon":  {"emoji": "🍉", "desc": "Needs warm weather, sandy loam soil and full sunlight.", "season": "Summer"},
#     "muskmelon":   {"emoji": "🍈", "desc": "Grows in warm dry climate with sandy well-drained soil.", "season": "Summer"},
#     "apple":       {"emoji": "🍎", "desc": "Requires cool winters and mild summers, hilly terrain soil.", "season": "Perennial"},
#     "orange":      {"emoji": "🍊", "desc": "Subtropical fruit needing well-drained deep loamy soil.", "season": "Perennial"},
#     "papaya":      {"emoji": "🍈", "desc": "Tropical fruit, grows fast in warm humid conditions.", "season": "Perennial"},
#     "coconut":     {"emoji": "🥥", "desc": "Coastal tropical crop, needs high humidity and sandy soil.", "season": "Perennial"},
#     "cotton":      {"emoji": "🌸", "desc": "Needs black cotton soil, high temperature and moderate rain.", "season": "Kharif"},
#     "jute":        {"emoji": "🌿", "desc": "Grows in warm humid climate with alluvial loamy soil.", "season": "Kharif"},
#     "coffee":      {"emoji": "☕", "desc": "Needs tropical highland climate with rich well-drained soil.", "season": "Perennial"},
# }

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()
#         features = [
#             float(data["N"]),
#             float(data["P"]),
#             float(data["K"]),
#             float(data["temperature"]),
#             float(data["humidity"]),
#             float(data["ph"]),
#             float(data["rainfall"]),
#         ]
#         scaled = scaler.transform([features])
#         pred_enc = model.predict(scaled)[0]
#         proba    = model.predict_proba(scaled)[0]

#         crop = le.inverse_transform([pred_enc])[0]

#         # Top 3 crops with confidence
#         top3_idx  = np.argsort(proba)[::-1][:3]
#         top3 = [
#             {
#                 "crop": le.inverse_transform([i])[0],
#                 "confidence": round(proba[i] * 100, 1),
#                 "emoji": CROP_INFO.get(le.inverse_transform([i])[0], {}).get("emoji", "🌱")
#             }
#             for i in top3_idx
#         ]

#         info = CROP_INFO.get(crop, {"emoji": "🌱", "desc": "Good conditions for growth.", "season": "Varies"})

#         return jsonify({
#             "success": True,
#             "crop": crop,
#             "confidence": round(proba[pred_enc] * 100, 1),
#             "emoji": info["emoji"],
#             "desc": info["desc"],
#             "season": info["season"],
#             "top3": top3
#         })
#     except Exception as e:
#         return jsonify({"success": False, "error": str(e)})

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model  = joblib.load("best_model_rf.pkl")
scaler = joblib.load("scaler.pkl")
le     = joblib.load("label_encoder.pkl")

# Ideal ranges per crop — from dataset statistics (15th to 85th percentile)
IDEAL = {
    "rice":        {"N":{"mean":80,"min":60,"max":99},   "P":{"mean":48,"min":36,"max":60},  "K":{"mean":40,"min":30,"max":50},  "temperature":{"mean":23.7,"min":20,"max":27},  "humidity":{"mean":82,"min":75,"max":90},  "ph":{"mean":6.4,"min":5.5,"max":7.0},  "rainfall":{"mean":236,"min":180,"max":280}},
    "maize":       {"N":{"mean":78,"min":60,"max":99},   "P":{"mean":48,"min":36,"max":60},  "K":{"mean":20,"min":15,"max":25},  "temperature":{"mean":22.6,"min":18,"max":27},  "humidity":{"mean":65,"min":55,"max":75},  "ph":{"mean":6.2,"min":5.5,"max":7.0},  "rainfall":{"mean":85,"min":60,"max":110}},
    "chickpea":    {"N":{"mean":40,"min":20,"max":60},   "P":{"mean":68,"min":55,"max":80},  "K":{"mean":80,"min":60,"max":99},  "temperature":{"mean":18.9,"min":15,"max":23},  "humidity":{"mean":17,"min":14,"max":20},  "ph":{"mean":7.3,"min":6.5,"max":8.0},  "rainfall":{"mean":73,"min":55,"max":90}},
    "kidneybeans": {"N":{"mean":20,"min":10,"max":30},   "P":{"mean":68,"min":55,"max":80},  "K":{"mean":20,"min":15,"max":25},  "temperature":{"mean":20.1,"min":17,"max":23},  "humidity":{"mean":22,"min":18,"max":26},  "ph":{"mean":5.9,"min":5.5,"max":6.5},  "rainfall":{"mean":105,"min":80,"max":130}},
    "pigeonpeas":  {"N":{"mean":20,"min":10,"max":30},   "P":{"mean":68,"min":55,"max":80},  "K":{"mean":20,"min":15,"max":25},  "temperature":{"mean":27.7,"min":24,"max":31},  "humidity":{"mean":49,"min":40,"max":58},  "ph":{"mean":5.8,"min":5.0,"max":6.5},  "rainfall":{"mean":149,"min":120,"max":180}},
    "mothbeans":   {"N":{"mean":21,"min":10,"max":30},   "P":{"mean":48,"min":36,"max":60},  "K":{"mean":20,"min":15,"max":25},  "temperature":{"mean":28.2,"min":24,"max":32},  "humidity":{"mean":54,"min":45,"max":63},  "ph":{"mean":6.9,"min":6.0,"max":7.5},  "rainfall":{"mean":53,"min":35,"max":70}},
    "mungbean":    {"N":{"mean":21,"min":10,"max":30},   "P":{"mean":48,"min":36,"max":60},  "K":{"mean":20,"min":15,"max":25},  "temperature":{"mean":28.5,"min":25,"max":32},  "humidity":{"mean":86,"min":78,"max":92},  "ph":{"mean":6.7,"min":6.0,"max":7.2},  "rainfall":{"mean":49,"min":35,"max":63}},
    "blackgram":   {"N":{"mean":40,"min":20,"max":60},   "P":{"mean":68,"min":55,"max":80},  "K":{"mean":20,"min":15,"max":25},  "temperature":{"mean":29.9,"min":26,"max":34},  "humidity":{"mean":66,"min":58,"max":74},  "ph":{"mean":6.8,"min":6.0,"max":7.5},  "rainfall":{"mean":68,"min":50,"max":85}},
    "lentil":      {"N":{"mean":19,"min":10,"max":28},   "P":{"mean":68,"min":55,"max":80},  "K":{"mean":19,"min":14,"max":24},  "temperature":{"mean":24.5,"min":20,"max":28},  "humidity":{"mean":65,"min":55,"max":74},  "ph":{"mean":6.9,"min":6.0,"max":7.5},  "rainfall":{"mean":46,"min":30,"max":60}},
    "pomegranate": {"N":{"mean":19,"min":10,"max":28},   "P":{"mean":18,"min":10,"max":25},  "K":{"mean":40,"min":30,"max":50},  "temperature":{"mean":21.8,"min":18,"max":26},  "humidity":{"mean":90,"min":82,"max":96},  "ph":{"mean":6.4,"min":5.8,"max":7.0},  "rainfall":{"mean":111,"min":85,"max":138}},
    "banana":      {"N":{"mean":100,"min":80,"max":120}, "P":{"mean":75,"min":60,"max":90},  "K":{"mean":50,"min":38,"max":62},  "temperature":{"mean":27.4,"min":24,"max":31},  "humidity":{"mean":81,"min":73,"max":89},  "ph":{"mean":5.9,"min":5.0,"max":6.8},  "rainfall":{"mean":105,"min":80,"max":130}},
    "mango":       {"N":{"mean":20,"min":10,"max":30},   "P":{"mean":18,"min":10,"max":25},  "K":{"mean":30,"min":22,"max":38},  "temperature":{"mean":31.2,"min":27,"max":36},  "humidity":{"mean":51,"min":42,"max":60},  "ph":{"mean":5.8,"min":5.0,"max":6.5},  "rainfall":{"mean":95,"min":72,"max":118}},
    "grapes":      {"N":{"mean":23,"min":12,"max":34},   "P":{"mean":133,"min":110,"max":145},"K":{"mean":200,"min":180,"max":205},"temperature":{"mean":23.9,"min":20,"max":28},  "humidity":{"mean":82,"min":74,"max":90},  "ph":{"mean":6.0,"min":5.5,"max":6.8},  "rainfall":{"mean":70,"min":50,"max":90}},
    "watermelon":  {"N":{"mean":100,"min":80,"max":120}, "P":{"mean":18,"min":10,"max":25},  "K":{"mean":50,"min":38,"max":62},  "temperature":{"mean":25.6,"min":22,"max":29},  "humidity":{"mean":85,"min":77,"max":92},  "ph":{"mean":6.5,"min":5.8,"max":7.2},  "rainfall":{"mean":51,"min":35,"max":65}},
    "muskmelon":   {"N":{"mean":100,"min":80,"max":120}, "P":{"mean":18,"min":10,"max":25},  "K":{"mean":50,"min":38,"max":62},  "temperature":{"mean":28.7,"min":25,"max":33},  "humidity":{"mean":93,"min":85,"max":97},  "ph":{"mean":6.4,"min":5.8,"max":7.0},  "rainfall":{"mean":25,"min":18,"max":32}},
    "apple":       {"N":{"mean":21,"min":10,"max":30},   "P":{"mean":134,"min":110,"max":145},"K":{"mean":200,"min":180,"max":205},"temperature":{"mean":22.6,"min":18,"max":26},  "humidity":{"mean":92,"min":85,"max":97},  "ph":{"mean":5.9,"min":5.5,"max":6.5},  "rainfall":{"mean":113,"min":88,"max":138}},
    "orange":      {"N":{"mean":20,"min":10,"max":30},   "P":{"mean":10,"min":5,"max":15},   "K":{"mean":10,"min":5,"max":15},   "temperature":{"mean":22.8,"min":19,"max":27},  "humidity":{"mean":93,"min":85,"max":97},  "ph":{"mean":7.0,"min":6.2,"max":7.8},  "rainfall":{"mean":110,"min":85,"max":135}},
    "papaya":      {"N":{"mean":50,"min":35,"max":65},   "P":{"mean":59,"min":46,"max":72},  "K":{"mean":50,"min":38,"max":62},  "temperature":{"mean":33.7,"min":30,"max":38},  "humidity":{"mean":92,"min":84,"max":97},  "ph":{"mean":6.7,"min":6.0,"max":7.3},  "rainfall":{"mean":145,"min":112,"max":178}},
    "coconut":     {"N":{"mean":21,"min":10,"max":30},   "P":{"mean":16,"min":8,"max":22},   "K":{"mean":30,"min":22,"max":38},  "temperature":{"mean":27.4,"min":24,"max":31},  "humidity":{"mean":95,"min":88,"max":99},  "ph":{"mean":5.9,"min":5.0,"max":6.8},  "rainfall":{"mean":152,"min":120,"max":183}},
    "cotton":      {"N":{"mean":118,"min":100,"max":135},"P":{"mean":46,"min":35,"max":58},  "K":{"mean":44,"min":33,"max":56},  "temperature":{"mean":24.1,"min":20,"max":28},  "humidity":{"mean":80,"min":72,"max":88},  "ph":{"mean":6.9,"min":6.0,"max":7.8},  "rainfall":{"mean":80,"min":60,"max":100}},
    "jute":        {"N":{"mean":78,"min":60,"max":95},   "P":{"mean":46,"min":35,"max":58},  "K":{"mean":40,"min":30,"max":50},  "temperature":{"mean":25.0,"min":21,"max":29},  "humidity":{"mean":80,"min":72,"max":88},  "ph":{"mean":6.7,"min":6.0,"max":7.5},  "rainfall":{"mean":175,"min":140,"max":210}},
    "coffee":      {"N":{"mean":101,"min":82,"max":120}, "P":{"mean":28,"min":18,"max":37},  "K":{"mean":30,"min":22,"max":38},  "temperature":{"mean":25.5,"min":22,"max":29},  "humidity":{"mean":59,"min":50,"max":68},  "ph":{"mean":6.8,"min":6.0,"max":7.5},  "rainfall":{"mean":159,"min":125,"max":193}},
}

AMENDMENTS = {
    "N":           {"low": "Add Urea (46% N) or Ammonium Sulphate. Deficit is {diff} mg/kg.", "high": "Reduce nitrogen input. Avoid urea this season."},
    "P":           {"low": "Add Single Super Phosphate (SSP) or DAP. Deficit is {diff} mg/kg.", "high": "Skip phosphatic fertilizers this season."},
    "K":           {"low": "Add Muriate of Potash (MOP). Deficit is {diff} mg/kg.", "high": "Potassium is sufficient — no potash needed."},
    "ph":          {"low": "Soil is acidic. Add Agricultural Lime (CaCO₃) to raise pH.", "high": "Soil is alkaline. Add Gypsum or Sulphur to lower pH."},
    "temperature": {"low": "Too cold. Grow in warmer season or use plastic mulching.", "high": "Too hot. Use shade nets or drip irrigation to cool microclimate."},
    "humidity":    {"low": "Low humidity. Use drip irrigation or sprinklers.", "high": "High humidity risks fungal disease. Ensure proper spacing and drainage."},
    "rainfall":    {"low": "Insufficient rain. Supplement with drip or sprinkler irrigation.", "high": "Excess rain risk. Ensure proper field drainage channels."},
}

CROP_INFO = {
    "rice":        {"emoji":"🌾","season":"Kharif",   "desc":"Thrives in waterlogged, warm conditions with high humidity."},
    "maize":       {"emoji":"🌽","season":"Kharif",   "desc":"Grows well in well-drained loamy soil with moderate rainfall."},
    "chickpea":    {"emoji":"🫘","season":"Rabi",     "desc":"Prefers cool, dry climate and well-drained sandy loam soil."},
    "kidneybeans": {"emoji":"🫘","season":"Kharif",   "desc":"Needs warm temperatures and well-drained fertile soil."},
    "pigeonpeas":  {"emoji":"🌿","season":"Kharif",   "desc":"Drought-tolerant, grows in semi-arid regions with sandy soil."},
    "mothbeans":   {"emoji":"🌱","season":"Kharif",   "desc":"Very drought-resistant, ideal for arid sandy soils."},
    "mungbean":    {"emoji":"🌱","season":"Kharif",   "desc":"Short-duration crop, grows in warm humid conditions."},
    "blackgram":   {"emoji":"🫘","season":"Kharif",   "desc":"Grows in tropical climate with moderate to low rainfall."},
    "lentil":      {"emoji":"🫘","season":"Rabi",     "desc":"Cool-season crop, prefers loamy soil with good drainage."},
    "pomegranate": {"emoji":"🍎","season":"Perennial","desc":"Drought-tolerant fruit, thrives in semi-arid warm regions."},
    "banana":      {"emoji":"🍌","season":"Perennial","desc":"Needs tropical climate, high humidity and rich loamy soil."},
    "mango":       {"emoji":"🥭","season":"Perennial","desc":"Prefers tropical climate with dry spell before flowering."},
    "grapes":      {"emoji":"🍇","season":"Perennial","desc":"Thrives in well-drained sandy loam with hot dry summers."},
    "watermelon":  {"emoji":"🍉","season":"Summer",   "desc":"Needs warm weather, sandy loam soil and full sunlight."},
    "muskmelon":   {"emoji":"🍈","season":"Summer",   "desc":"Grows in warm dry climate with sandy well-drained soil."},
    "apple":       {"emoji":"🍎","season":"Perennial","desc":"Requires cool winters and mild summers, hilly terrain soil."},
    "orange":      {"emoji":"🍊","season":"Perennial","desc":"Subtropical fruit needing well-drained deep loamy soil."},
    "papaya":      {"emoji":"🍈","season":"Perennial","desc":"Tropical fruit, grows fast in warm humid conditions."},
    "coconut":     {"emoji":"🥥","season":"Perennial","desc":"Coastal tropical crop, needs high humidity and sandy soil."},
    "cotton":      {"emoji":"🌸","season":"Kharif",   "desc":"Needs black cotton soil, high temperature and moderate rain."},
    "jute":        {"emoji":"🌿","season":"Kharif",   "desc":"Grows in warm humid climate with alluvial loamy soil."},
    "coffee":      {"emoji":"☕","season":"Perennial","desc":"Needs tropical highland climate with rich well-drained soil."},
}

FEATURE_LABELS = {"N":"Nitrogen","P":"Phosphorus","K":"Potassium","temperature":"Temperature","humidity":"Humidity","ph":"Soil pH","rainfall":"Rainfall"}
UNITS = {"N":"mg/kg","P":"mg/kg","K":"mg/kg","temperature":"°C","humidity":"%","ph":"","rainfall":"mm"}


@app.route("/")
def home():
    crops = sorted(IDEAL.keys())
    return render_template("index.html", crops=crops)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data     = request.get_json()
        features = [float(data[f]) for f in ["N","P","K","temperature","humidity","ph","rainfall"]]
        scaled   = scaler.transform([features])
        pred     = model.predict(scaled)[0]
        proba    = model.predict_proba(scaled)[0]
        crop     = le.inverse_transform([pred])[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [{"crop": le.inverse_transform([i])[0], "confidence": round(proba[i]*100,1),
                  "emoji": CROP_INFO.get(le.inverse_transform([i])[0],{}).get("emoji","🌱")} for i in top3_idx]
        info = CROP_INFO.get(crop, {"emoji":"🌱","desc":"Good conditions.","season":"Varies"})
        return jsonify({"success":True,"crop":crop,"confidence":round(proba[pred]*100,1),
                        "emoji":info["emoji"],"desc":info["desc"],"season":info["season"],"top3":top3})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/gap-analysis", methods=["POST"])
def gap_analysis():
    try:
        data    = request.get_json()
        desired = data["desired_crop"].lower()
        features = ["N","P","K","temperature","humidity","ph","rainfall"]
        current  = {f: float(data[f]) for f in features}

        if desired not in IDEAL:
            return jsonify({"success": False, "error": "Crop not found."})

        ideal   = IDEAL[desired]
        results = []

        for f in features:
            curr_val  = current[f]
            ideal_min = ideal[f]["min"]
            ideal_max = ideal[f]["max"]
            ideal_avg = ideal[f]["mean"]

            if ideal_min <= curr_val <= ideal_max:
                status = "good"
                advice = f"Your {FEATURE_LABELS[f]} is within the ideal range. No action needed."
                diff   = 0
            elif curr_val < ideal_min:
                status = "low"
                diff   = round(ideal_avg - curr_val, 2)
                advice = AMENDMENTS[f]["low"].replace("{diff}", str(round(diff,1)))
            else:
                status = "high"
                diff   = round(curr_val - ideal_avg, 2)
                advice = AMENDMENTS[f]["high"].replace("{diff}", str(round(diff,1)))

            results.append({
                "feature":    f,
                "label":      FEATURE_LABELS[f],
                "unit":       UNITS[f],
                "current":    curr_val,
                "ideal_min":  ideal_min,
                "ideal_max":  ideal_max,
                "ideal_mean": ideal_avg,
                "status":     status,
                "diff":       diff,
                "advice":     advice
            })

        good_count = sum(1 for r in results if r["status"] == "good")
        score      = round((good_count / len(results)) * 100)

        if score == 100:   verdict = f"Your soil is perfectly suited for {desired.capitalize()}! You can start planting."
        elif score >= 70:  verdict = "Your soil is mostly suitable. Make a few adjustments and you're good to go."
        elif score >= 40:  verdict = "Moderate changes needed. Follow the recommendations before planting."
        else:              verdict = "Significant amendments required. Consider growing a different crop first."

        info = CROP_INFO.get(desired, {"emoji":"🌱","season":"Varies","desc":""})
        return jsonify({"success":True,"desired_crop":desired,"emoji":info["emoji"],"season":info["season"],
                        "score":score,"good_count":good_count,"total":len(results),"verdict":verdict,"results":results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)