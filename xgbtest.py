from flask import Flask, request, jsonify
import numpy as np
import joblib
import re

app = Flask(__name__)

# Load models once when server starts
disease_model = joblib.load("disease_model.pkl")
doctor_model = joblib.load("doctor_model.pkl")

disease_encoder = joblib.load("disease_encoder.pkl")
doctor_encoder = joblib.load("doctor_encoder.pkl")
symptom_columns = joblib.load("symptom_columns.pkl")

symptom_map = {
    "fever": "fever",
    "temperature": "fever",
    "cold": "cold",
    "runny nose": "cold",
    "cough": "cough",
    "breath": "breathing_problem",
    "breathing": "breathing_problem",
    "breathless": "breathing_problem",
    "chest pain": "chest_pain",
    "tired": "tiredness",
    "weak": "tiredness",
    "fatigue": "tiredness",
    "weight loss": "weight_loss",
    "no appetite": "no_appetite",
    "loss of appetite": "no_appetite",
    "stomach pain": "stomach_pain",
    "abdominal pain": "stomach_pain",
    "nausea": "nausea",
    "vomit": "vomiting",
    "vomiting": "vomiting",
    "loose motion": "loose_stool",
    "diarrhea": "loose_stool",
    "constipation": "constipation",
    "gas": "gas",
    "joint pain": "joint_pain",
    "muscle pain": "muscle_pain",
    "back pain": "back_pain",
    "headache": "headache",
    "night sweat": "night_sweat",
    "rash": "skin_rash",
    "itch": "itching",
    "frequent urination": "frequent_urination",
    "thirst": "excess_thirst",
    "burning urine": "burning_urine",
    "less urine": "less_urine",
    "leg swelling": "leg_swelling",
    "face swelling": "face_swelling",
    "yellow eyes": "yellow_eyes",
    "pale": "pale_skin",
    "fits": "fits",
    "seizure": "fits",
    "dizzy": "dizziness",
    "sad": "sadness",
    "sadness": "sadness",
    "anxiety": "fear",
    "fear": "fear",
    "irregular periods": "irregular_periods",
    "infertility": "unable_to_conceive",
    "poor growth": "poor_growth_child"
}


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "symptoms" not in data:
        return jsonify({"error": "No symptoms provided"}), 400

    user_text = data["symptoms"].lower()

    X_new = np.zeros(len(symptom_columns))

    for phrase, mapped_symptom in symptom_map.items():
        if re.search(rf"\b{re.escape(phrase)}\b", user_text):
            if mapped_symptom in symptom_columns:
                idx = symptom_columns.index(mapped_symptom)
                X_new[idx] = 1

    if X_new.sum() == 0:
        return jsonify({"error": "No recognizable symptoms detected"}), 400

    X_new = X_new.reshape(1, -1)

    disease_probs = disease_model.predict_proba(X_new)[0]
    best_idx = np.argmax(disease_probs)
    best_disease = disease_encoder.inverse_transform([best_idx])[0]
    best_prob = float(disease_probs[best_idx])

    doctor_pred = doctor_model.predict(X_new)
    doctor_name = doctor_encoder.inverse_transform(doctor_pred)[0]

    return jsonify({
        "disease": best_disease,
        "confidence": round(best_prob * 100, 2),
        "recommended_doctor": doctor_name
    })


if __name__ == "__main__":
    app.run(debug=True)
