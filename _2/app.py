
from __future__ import annotations
import os, json, math, random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from flask import Flask, render_template, request, session, jsonify, redirect, url_for
import pandas as pd
import numpy as np

app = Flask(__name__)
# NOTE: replace with a strong secret key in production
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")

# --- Load dataset ---
ENRICHED = "C:/Users/rajes/OneDrive/Pictures/_2/Disease_symptom_and_patient_profile_dataset_enriched.csv"
ORIGINAL = "C:/Users/rajes/OneDrive/Pictures/_2/Disease_symptom_and_patient_profile_dataset.csv"
# Fallback to local files in the repo
LOCAL_ENRICHED = os.path.join(os.path.dirname(__file__), "Disease_symptom_and_patient_profile_dataset_enriched.csv")
LOCAL_ORIGINAL = os.path.join(os.path.dirname(__file__), "Disease_symptom_and_patient_profile_dataset.csv")

csv_path = next((p for p in [ENRICHED, ORIGINAL, LOCAL_ENRICHED, LOCAL_ORIGINAL] if os.path.exists(p)), None)
if not csv_path:
    raise RuntimeError("Dataset CSV not found. Place the CSV next to app.py or update the path.")
df = pd.read_csv(csv_path)

# --- Load disease details from CSV ---
CSV_PATH = "C:/Users/rajes/OneDrive/Pictures/_2/disease_descriptions.csv"
LOCAL_DETAILS = os.path.join(os.path.dirname(__file__), "disease_descriptions.csv")
CSV_PATH = CSV_PATH if os.path.exists(CSV_PATH) else LOCAL_DETAILS
if os.path.exists(CSV_PATH):
    disease_details_df = pd.read_csv(CSV_PATH)
    disease_details_df.columns = [c.strip().lower() for c in disease_details_df.columns]
    if not all(col in disease_details_df.columns for col in ["disease", "description", "preventive_measures"]):
        raise RuntimeError("CSV file must contain 'Disease', 'Description', and 'Preventive_Measures' columns.")
else:
    raise RuntimeError(f"CSV file not found at {CSV_PATH}")

# Normalize column names (strip & lower for internal use)
orig_cols = list(df.columns)
df.columns = [c.strip().lower() for c in df.columns]

# We expect a 'disease' column to exist
if "disease" not in df.columns:
    raise RuntimeError("Dataset must contain a 'Disease' column.")

# Identify candidate features to ask about
CATEGORICAL_CANDIDATES = [
    "fever", "cough", "fatigue", "difficulty breathing", "gender", "blood pressure",
    "cholesterol level", "outcome variable", "smoking_status", "alcohol_consumption",
    "physical_activity_level"
]
NUMERIC_CANDIDATES = [
    "age", "bmi", "systolic_bp", "diastolic_bp", "temperature_f", "heart_rate",
    "respiratory_rate", "blood_sugar_mg_dl", "cholesterol_mg_dl"
]

# Keep only those that actually exist in the dataset
cat_features = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
num_features = [c for c in NUMERIC_CANDIDATES if c in df.columns]

# Build per-disease statistics to evaluate answers
disease_stats: Dict[str, Dict[str, Any]] = {}
for disease, group in df.groupby("disease"):
    # Categorical: mode and value counts (for options)
    cat_modes = {}
    cat_options = {}
    cat_value_counts = {}
    for c in cat_features:
        vc = group[c].value_counts(dropna=False)
        cat_value_counts[c] = {str(k): v for k, v in vc.to_dict().items()}
        mode_val = vc.index[0] if not vc.empty else None
        cat_modes[c] = mode_val
        cat_options[c] = vc.index.astype(str).tolist()
    # Numeric: mean & std for scoring
    num_stats = {}
    for c in num_features:
        mean = float(group[c].astype(float).mean(skipna=True))
        if math.isnan(mean):
            mean = None
        std = float(group[c].astype(float).std(ddof=0, skipna=True)) or 1.0
        num_stats[c] = {"mean": mean, "std": std}
    # Add disease details from CSV
    matching_details = disease_details_df[disease_details_df["disease"].str.lower() == disease.lower()]
    if not matching_details.empty:
        details = matching_details.iloc[0]
    else:
        details = {"description": "No description available.", "preventive_measures": "No preventive measures available."}
    disease_stats[disease] = {
        "cat_modes": cat_modes,
        "cat_options": cat_options,
        "cat_value_counts": cat_value_counts,
        "num_stats": num_stats,
        "description": details["description"],
        "preventive_measures": details["preventive_measures"]
    }

# Build available questions (global)
available_questions: List[Dict[str, Any]] = []
q_id = 0

special_phrases = {
    "fever": "have fever",
    "cough": "have cough",
    "fatigue": "have fatigue",
    "difficulty breathing": "have difficulty breathing",
}

for c in cat_features:
    options = set()
    for d in disease_stats.values():
        options.update([str(o) for o in d["cat_options"][c]])
    options = options - {"nan", "None", "NaN"}
    for val in options:
        if c in special_phrases:
            if val == "Yes":
                text = f"Does the patient {special_phrases[c]}?"
            elif val == "No":
                continue  # Skip negative for binary symptoms
            else:
                continue
        elif c == "gender":
            text = f"Is the patient {val.lower()}?"
        elif c == "outcome variable":
            text = f"Is the outcome {val.lower()}?"
        else:
            text = f"Is the {c.replace('_', ' ')} {val.lower()}?"
        available_questions.append({
            "id": str(q_id),
            "text": text,
            "feature": c,
            "value": val,
            "type": "cat"
        })
        q_id += 1

# Numeric questions
num_questions = {
    "age": [
        {"min": 0, "max": 20, "desc": "under 20 years old"},
        {"min": 20, "max": 40, "desc": "between 20 and 40 years old"},
        {"min": 40, "max": 60, "desc": "between 40 and 60 years old"},
        {"min": 60, "max": 80, "desc": "between 60 and 80 years old"},
        {"min": 80, "max": 999, "desc": "over 80 years old"},
    ],
    "bmi": [
        {"min": 0, "max": 18.5, "desc": "underweight (BMI under 18.5)"},
        {"min": 18.5, "max": 25, "desc": "normal weight (BMI 18.5 to 25)"},
        {"min": 25, "max": 30, "desc": "overweight (BMI 25 to 30)"},
        {"min": 30, "max": 999, "desc": "obese (BMI over 30)"},
    ],
}

for feature, bins in num_questions.items():
    if feature not in num_features:
        continue
    for b in bins:
        text = f"Is the patient {b['desc']}?"
        available_questions.append({
            "id": str(q_id),
            "text": text,
            "feature": feature,
            "type": "num",
            "min": b["min"],
            "max": b["max"]
        })
        q_id += 1

# Answer function
def get_answer(disease: str, q: Dict[str, Any]) -> str:
    stats = disease_stats[disease]
    feature = q["feature"]
    if q.get("type") == "num":
        st = stats["num_stats"].get(feature)
        if not st or st["mean"] is None:
            return "don't know"
        mean = st["mean"]
        std = st["std"]
        minv = q["min"]
        maxv = math.inf if q["max"] == 999 else q["max"]
        if minv <= mean <= maxv:
            return "yes"
        # Distance to the range
        if mean < minv:
            dist = minv - mean
        else:
            dist = mean - maxv
        if dist <= std:
            return "maybe"
        else:
            return "no"
    else:  # cat
        value = q["value"]
        vc = stats["cat_value_counts"].get(feature, {})
        total = sum(vc.values())
        if total == 0:
            return "don't know"
        prop = vc.get(value, 0) / total
        if prop >= 0.7:
            return "yes"
        elif prop <= 0.3:
            return "no"
        else:
            return "maybe"

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    disease = random.choice(list(disease_stats.keys()))
    session['game'] = {
        "disease": disease,
        "asked": []
    }
    return redirect(url_for('play'))

@app.route('/play')
def play():
    g = session.get('game')
    if not g:
        return redirect(url_for('index'))
    return render_template('game.html', questions=available_questions, asked=g['asked'])

@app.route('/ask', methods=['POST'])
def ask():
    g = session.get('game')
    if not g:
        return jsonify({"error": "No active game."}), 400
    data = request.get_json(force=True)
    q_id = data.get("q_id")
    q = next((qq for qq in available_questions if qq["id"] == q_id), None)
    if not q:
        return jsonify({"error": "Invalid question."}), 400
    answer = get_answer(g["disease"], q)
    g["asked"].append({"text": q["text"], "answer": answer})
    session['game'] = g
    return jsonify({"answer": answer, "text": q["text"]})

@app.route('/guess', methods=['POST'])
def guess():
    g = session.get('game')
    if not g:
        return jsonify({"error": "No active game."}), 400
    # Use request.form to get the guess from the form data
    user_guess = request.form.get("guess")
    if not user_guess:
        return jsonify({"error": "No guess provided."}), 400
    target = g["disease"]
    correct = (user_guess.strip().lower() == target.lower())
    payload = {
        "correct": correct,
        "target": target,
        "user_guess": user_guess
    }
    # Store the result in the session and redirect to result page
    session['result'] = payload
    session.pop('game', None)
    return redirect(url_for('result'))

@app.route('/result')
def result():
    result = session.get('result')
    if not result:
        return redirect(url_for('index'))
    # Get disease details for the target disease
    target_disease = result["target"]
    disease_info = disease_stats.get(target_disease, {})
    description = disease_info.get("description", "No description available.")
    preventive_measures = disease_info.get("preventive_measures", "No preventive measures available.")
    return render_template('result.html', result=result, description=description, preventive_measures=preventive_measures)

if __name__ == "__main__":
    app.run(debug=True)