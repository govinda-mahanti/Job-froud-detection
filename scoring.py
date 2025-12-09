import numpy as np

# -------------------------
# Company OSINT scoring
# -------------------------
def company_osint_score(company_name=None, company_domain=None):
    score = 50.0
    if company_domain and len(company_domain) > 5:
        score += 20
    if company_name and len(company_name.split()) >= 2:
        score += 10
    return float(max(0, min(100, score)))


# -------------------------
# Email phishing scoring
# -------------------------
def email_scoring(subject, body):
    text = (subject or "") + " " + (body or "")
    risk = 0

    if "urgent" in text.lower():
        risk += 1
    if "immediately" in text.lower():
        risk += 1
    if "click here" in text.lower():
        risk += 2
    if "western union" in text.lower():
        risk += 3

    fraud_score = min(100, risk * 25)
    auth_score = 100 - fraud_score
    fraud_prob = fraud_score / 100
    return auth_score, fraud_score, fraud_prob


# -------------------------
# Job ML fraud scoring
# -------------------------
def job_auth_score_from_text(job_title, job_description, company_score, email_prob):
    from app import embed, clf_lr, clf_rf, clf_xgb

    text = (job_title or "") + " " + (job_description or "")
    emb = embed.encode([text])[0]

    probs = []
    try: probs.append(clf_lr.predict_proba([emb])[:,1][0])
    except: pass
    try: probs.append(clf_rf.predict_proba([emb])[:,1][0])
    except: pass
    try:
        if clf_xgb:
            probs.append(clf_xgb.predict_proba([emb])[:,1][0])
    except: pass

    p = float(np.mean(probs)) if probs else 0.5
    job_fraud_prob = p * 0.85 + (1 - company_score/100) * 0.15
    job_fraud_prob = min(1.0, job_fraud_prob * 0.7 + email_prob * 0.3)
    job_auth_score = 100 * (1 - job_fraud_prob)

    return job_auth_score, job_fraud_prob


# -------------------------
# Interpretation buckets
# -------------------------
def interpret_company_score(score):
    s = float(score)
    if s >= 80: return "Low Risk", "80–100"
    if s >= 60: return "Moderate–Low Risk", "60–79"
    if s >= 40: return "Moderate Risk", "40–59"
    if s >= 20: return "High Risk", "20–39"
    return "Critical Risk (Likely Fraudulent)", "0–19"


def interpret_job_score(score):
    s = float(score)
    if s >= 80: return "Strong Authentic Job", "80–100"
    if s >= 60: return "Mostly Safe", "60–79"
    if s >= 40: return "Suspicious", "40–59"
    if s >= 20: return "High Fraud Probability", "20–39"
    return "Definite Scam", "0–19"


def job_recommendation_label(prob):
    if prob < 0.40:
        return "SAFE_TO_APPLY"
    if prob < 0.75:
        return "APPLY_WITH_CAUTION"
    return "DO_NOT_APPLY"


def overall_multimodal_label(job_prob, email_prob):
    overall = 0.7 * job_prob + 0.3 * email_prob
    label = (
        "SAFE_TO_APPLY" if overall < 0.40 else
        "APPLY_WITH_CAUTION" if overall < 0.75 else
        "DO_NOT_APPLY"
    )
    return label, overall
