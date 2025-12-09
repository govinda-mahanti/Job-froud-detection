# ============================================
# app.py — Job Fraud Detection Backend API (Transformer Version)
# ============================================


import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
import torch

# Load environment variables from .env file
load_dotenv()

# Set Hugging Face token if available
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    os.environ['HF_TOKEN'] = hf_token
    # Optional: Suppress Hugging Face warnings
    hf_logging.set_verbosity_error()

# -------------------------
# Load Fine-Tuned Transformer Model
# -------------------------
MODEL_DIR = "bert_tiny_finetuned_model"   # <- CHANGE if your folder name differs

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, token=hf_token if hf_token else None)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, token=hf_token if hf_token else None)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    if "authentication" in str(e).lower() and not hf_token:
        print("Hint: You might need to set HF_TOKEN in your .env file for private models")

DEVICE = torch.device("cpu")
model.to(DEVICE)
model.eval()

# -------------------------
# Import scoring functions
# -------------------------
from scoring import (
    company_osint_score,
    email_scoring,
    interpret_company_score,
    interpret_job_score,
    job_recommendation_label,
    overall_multimodal_label
)

# -------------------------
# Job scoring using finetuned model
# -------------------------
def transformer_job_fraud_probability(job_title, job_description):
    text = (job_title or "") + " " + (job_description or "")
    
    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"].to(DEVICE),
            attention_mask=inputs["attention_mask"].to(DEVICE)
        )
        logits = outputs.logits.cpu().numpy()
    
    # Softmax to get probability
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    fraud_prob = float(probs[0][1])   # class "1" = fraud

    return fraud_prob


# -------------------------
# Initialize Flask
# -------------------------
app = Flask(__name__)


# -------------------------
# Prediction Route
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Job Fraud Detection API is running",
        "endpoints": {
            "GET /": "Health check (this endpoint)",
            "POST /predict": "Predict job fraud probability"
        }
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    job_title       = payload.get("job_title", "")
    job_description = payload.get("job_description", "")
    company_name    = payload.get("company_name", "")
    company_domain  = payload.get("company_domain", "")
    salary_raw      = payload.get("salary_raw")
    location        = payload.get("location")
    email_subject   = payload.get("email_subject", "")
    email_body      = payload.get("email_body", "")

    # --- Company authenticity score ---
    company_score = company_osint_score(company_name, company_domain)
    company_label, company_bucket = interpret_company_score(company_score)

    # --- Email scoring ---
    email_auth_score, email_fraud_score, email_fraud_prob = email_scoring(
        email_subject,
        email_body
    )
    email_interpret = (
        "Legit" if email_fraud_prob <= 0.2 else
        ("Suspicious" if email_fraud_prob <= 0.5 else "Likely Phishing")
    )

    # --- Job scoring using FINETUNED MODEL ---
    base_fraud_prob = transformer_job_fraud_probability(job_title, job_description)

    # Blend with company + email evidence (your rule)
    job_fraud_prob = base_fraud_prob * 0.8 + (1 - company_score/100) * 0.1 + email_fraud_prob * 0.1
    job_fraud_prob = min(max(job_fraud_prob, 0), 1)  # clamp 0-1

    job_auth_score = round((1 - job_fraud_prob) * 100, 2)
    job_label, job_bucket = interpret_job_score(job_auth_score)

    # --- Final Recommendation ---
    final_label = job_recommendation_label(job_fraud_prob)
    overall_label, overall_prob = overall_multimodal_label(job_fraud_prob, email_fraud_prob)

    # -------------------------
    # Build result
    # -------------------------
    result = {
        "company_auth_score": round(company_score, 2),
        "company_label": company_label,
        "company_bucket": company_bucket,

        "job_auth_score": job_auth_score,
        "job_label": job_label,
        "job_bucket": job_bucket,
        "job_fraud_probability": round(job_fraud_prob, 4),

        "email_auth_score": round(email_auth_score, 2),
        "email_fraud_score": round(email_fraud_score, 2),
        "email_fraud_probability": round(email_fraud_prob, 4),
        "email_interpretation": email_interpret,

        "final_recommendation": final_label,

        "overall_multimodal_label": overall_label,
        "overall_multimodal_probability": round(overall_prob, 4),
    }

    return jsonify(result)


# -------------------------
# Run the server
# -------------------------
if __name__ == "__main__":
    print("🚀 API running at: http://127.0.0.1:5000/predict")
    app.run(host="0.0.0.0", port=5000, debug=True)
