import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def build_fuzzy_system():
    """
    Builds and returns the Fuzzy Logic Control System.
    It takes the ML Probability and the Patient's Cholesterol as inputs.
    """
    # 1. Define Antecedents (Inputs) and Consequent (Output)
    # ML Probability ranges from 0.0 to 1.0
    ml_prob = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "ml_prob")
    # Raw Cholesterol usually ranges from ~100 to ~600
    cholesterol = ctrl.Antecedent(np.arange(0, 601, 1), "cholesterol")
    # Our final Fuzzy Risk Score will be a 1-10 scale
    risk_score = ctrl.Consequent(np.arange(0, 10.1, 0.1), "risk_score")

    # 2. Define Membership Functions (The exact clinical coordinates!)

    # ML Probability Mappings
    ml_prob["low"] = fuzz.trapmf(ml_prob.universe, [0, 0, 0.3, 0.5])
    ml_prob["medium"] = fuzz.trimf(ml_prob.universe, [0.3, 0.5, 0.7])
    ml_prob["high"] = fuzz.trapmf(ml_prob.universe, [0.5, 0.7, 1.0, 1.0])

    # Cholesterol Mappings (Using standard clinical thresholds)
    cholesterol["normal"] = fuzz.trapmf(cholesterol.universe, [0, 0, 200, 240])
    cholesterol["borderline"] = fuzz.trimf(cholesterol.universe, [200, 240, 280])
    cholesterol["high"] = fuzz.trapmf(cholesterol.universe, [240, 280, 600, 600])

    # Output Risk Score Mappings
    risk_score["safe"] = fuzz.trapmf(risk_score.universe, [0, 0, 3, 5])
    risk_score["warning"] = fuzz.trimf(risk_score.universe, [3, 5, 8])
    risk_score["critical"] = fuzz.trapmf(risk_score.universe, [6, 8, 10, 10])

    # 3. Define the Fuzzy Rules (The "Brain")
    # If the ML says low risk, and cholesterol is normal, the patient is safe.
    rule1 = ctrl.Rule(ml_prob["low"] & cholesterol["normal"], risk_score["safe"])

    # If the ML says low risk, but cholesterol is high, raise a warning.
    rule2 = ctrl.Rule(ml_prob["low"] & cholesterol["high"], risk_score["warning"])

    # If the ML is unsure (medium), play it safe and issue a warning.
    rule3 = ctrl.Rule(ml_prob["medium"], risk_score["warning"])

    # If the ML says high risk, but cholesterol is normal, it's a severe warning.
    rule4 = ctrl.Rule(ml_prob["high"] & cholesterol["normal"], risk_score["warning"])

    # If BOTH the ML and Cholesterol agree it's bad, trigger a critical alert.
    rule5 = ctrl.Rule(
        ml_prob["high"] & (cholesterol["high"] | cholesterol["borderline"]),
        risk_score["critical"],
    )

    # 4. Build and return the simulation engine
    risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    return ctrl.ControlSystemSimulation(risk_ctrl)


def generate_linguistic_inference(
    ml_probability: float, raw_cholesterol: float
) -> dict:
    """
    Takes the raw ML output and patient data, runs it through the Fuzzy engine,
    and returns a structured JSON dictionary ready for the LLM API.
    """
    # 1. Initialize the system
    fuzzy_sim = build_fuzzy_system()

    # 2. Input the values
    fuzzy_sim.input["ml_prob"] = float(ml_probability)
    fuzzy_sim.input["cholesterol"] = float(raw_cholesterol)

    # 3. Crunch the fuzzy math
    try:
        fuzzy_sim.compute()
        score = fuzzy_sim.output["risk_score"]
    except ValueError:
        # Fallback safeguard in case of bizarre outlier data
        score = ml_probability * 10

    # 4. Translate the 1-10 score into a Linguistic Verdict
    if score <= 4.0:
        verdict = "LOW RISK (Routine Cardiology Checkup)"
    elif score <= 7.0:
        verdict = "MODERATE RISK (Warning - Monitor Vitals and Diet)"
    else:
        verdict = "HIGH RISK (Danger - Immediate Cardiology Consult Required)"

    # 5. Package the final payload
    return {
        "status": "success",
        "ml_probability": round(ml_probability, 3),
        "fuzzy_risk_score": round(score, 1),
        "fuzzy_verdict": verdict,
        "driving_features": {"Cholesterol": raw_cholesterol},
    }


# ==========================================
# TESTING ZONE
# ==========================================
if __name__ == "__main__":
    print("Testing fuzzy_translator.py locally...\n")

    print("Scenario A: ML Prob = 0.12, Cholesterol = 180 (Normal)")
    result_a = generate_linguistic_inference(0.12, 180)
    print(f"Result: {result_a}\n")

    print("Scenario B: ML Prob = 0.20, Cholesterol = 350 (High)")
    result_b = generate_linguistic_inference(0.20, 350)
    print(f"Result: {result_b}\n")

    print("Scenario C: ML Prob = 0.88, Cholesterol = 290 (High)")
    result_c = generate_linguistic_inference(0.88, 290)
    print(f"Result: {result_c}\n")
