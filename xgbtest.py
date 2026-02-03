import numpy as np
import joblib
import re

disease_model = joblib.load("disease_model.pkl")
doctor_model = joblib.load("doctor_model.pkl")

disease_encoder = joblib.load("disease_encoder.pkl")
doctor_encoder = joblib.load("doctor_encoder.pkl")
symptom_columns = joblib.load("symptom_columns.pkl")

# Mapping of user input phrases to symptom column names because user input can vary widely
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

print("\nSymptom-Based Disease Prediction System")
print("Type 'exit' to quit\n")

while True:
    user_text = input("Enter symptoms: ").strip().lower()

    if user_text in ["exit", "quit"]:
        break

    X_new = np.zeros(len(symptom_columns))

    for phrase, mapped_symptom in symptom_map.items():
        if re.search(rf"\b{re.escape(phrase)}\b", user_text):
            if mapped_symptom in symptom_columns:
                idx = symptom_columns.index(mapped_symptom)
                X_new[idx] = 1

    if X_new.sum() == 0:
        print("\nNo recognizable symptoms detected.\n")
        continue

    X_new = X_new.reshape(1, -1)

    disease_probs = disease_model.predict_proba(X_new)[0]
    top3_idx = np.argsort(disease_probs)[-3:][::-1]
    top3_diseases = disease_encoder.inverse_transform(top3_idx)
    top3_probs = disease_probs[top3_idx]

    doctor_pred = doctor_model.predict(X_new)
    doctor_name = doctor_encoder.inverse_transform(doctor_pred)[0]

    print("\n========================================")
    print("SYMPTOM-BASED HEALTH ASSESSMENT")
    print("========================================\n")

    print("Possible Conditions (Top 3):")
    print("----------------------------------------")
    for i, (d, p) in enumerate(zip(top3_diseases, top3_probs), start=1):
        print(f"{i}. {d:<25} : {p * 100:6.2f}%")

    print("\nRecommended Specialist:")
    print("----------------------------------------")
    print(doctor_name)

    print("\n----------------------------------------")
    print("This result is based on symptom patterns only.")
    print("It is not a medical diagnosis.")
    print("Please consult a qualified doctor for confirmation.")
    print("========================================\n")
