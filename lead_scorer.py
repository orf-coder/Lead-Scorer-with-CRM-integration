import csv
import os
import glob
import pickle
import numpy as np
import pandas as pd
import models
import importlib.util
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

def extract_keywords_nlp(text):
    """Extract relevant keywords using NLP techniques."""
    if not NLP_AVAILABLE:
        return []

    doc = nlp(text.lower())
    keywords = []

    # Extract noun chunks and important tokens
    for chunk in doc.noun_chunks:
        keywords.append(chunk.text)

    # Extract important adjectives and verbs
    for token in doc:
        if token.pos_ in ['ADJ', 'VERB'] and not token.is_stop and len(token.text) > 2:
            keywords.append(token.text)

    # Remove duplicates and short words
    keywords = list(set(keywords))
    keywords = [k for k in keywords if len(k.split()) <= 3]  # Keep phrases up to 3 words

    return keywords

# Load the google_sheets module
gs_path = os.path.join(os.path.dirname(__file__), 'Google', 'google_sheets.py')
spec = importlib.util.spec_from_file_location("google_sheets", gs_path)
google_sheets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(google_sheets)

def judge_lead(message, job_title=None, source=None, company=None, timestamp=None):
    """Judge a lead based on relevance, intent, and potential scores.

    Parameters:
    - message: the lead message text
    - job_title: optional job title string
    - source: optional source string (e.g., 'email', 'LinkedIn')

    Returns:
    - total_score: combined score from relevance, intent, potential
    - label: 'Hot', 'Warm', or 'Cold'
    - reason: explanation string
    - scores: dict with 'relevance', 'intent', 'potential' scores
    """
    # Relevance: Interest in product/service features
    relevance_keywords = {
        "demo": 25,
        "pricing": 25,
        "quote": 20,
        "features": 15,
        "trial": 20,
        "consultation": 15,
        "info": 10,
        "details": 10,
        "specs": 15,
        "benefits": 15,
        "capabilities": 15,
        "collaborate": 5,
        "understand": 10
    }

    # Intent: Buyer's intent to purchase or urgency
    intent_keywords = {
        "buy": 15,
        "urgent": 10,
        "purchase": 15,
        "order": 15,
        "interested": 10,
        "need": 10,
        "asap": 10,
        "deadline": 10,
        "soon": 5,
        "now": 10,
        "require": 10
    }

    # Potential: Lead's value based on role, source, engagement
    potential_keywords = {
        "VP": 15,
        "HR": 10,
        "CEO": 20,
        "CTO": 20,
        "Director": 15,
        "Manager": 10,
        "Founder": 15,
        "Owner": 10,
        "Lead": 10,
        "Senior": 10
    }

    negative_potential_keywords = {
        "student": -50,
        "job": -20,
        "career": -20,
        "internship": -10,
        "freelance": -15,
        "unemployed": -20,
        "seeking": -15,
        "entry": -10,
        "junior": -10,
        "trainee": -10,
        "learning": -10,
        "exploring": -5,
        "cold": -15,
        "not looking to purchase": -50,
        "currently": -10,
        "purposes": -5,
        "just": -5,
        "nothing right now": -30,
        "not interested": -40,
        "no budget": -40,
        "no intention": -40,
        "researching": -10,
        "curious": -5,
        "academic": -20,
        "educational": -15,
        "right now": -10,
        "at this time": -10,
        "for now": -5
    }

    message_lower = str(message or "").lower()
    job_title_lower = str(job_title or "").lower()
    source_lower = str(source or "").lower()
    company_lower = str(company or "").lower()

    # Extract dynamic keywords using NLP
    nlp_keywords = extract_keywords_nlp(message or "")
    nlp_keywords_lower = [k.lower() for k in nlp_keywords]

    relevance_score = 0
    intent_score = 0
    potential_score = 0

    found_relevance = []
    found_intent = []
    found_potential = []
    found_negative_potential = []
    found_nlp = []

    # Relevance score
    for keyword, score in relevance_keywords.items():
        if keyword in message_lower:
            relevance_score += score
            found_relevance.append(keyword)

    # Intent score
    for keyword, score in intent_keywords.items():
        if keyword in message_lower:
            intent_score += score
            found_intent.append(keyword)

    # Potential score from positive keywords
    for keyword, score in potential_keywords.items():
        if keyword in message_lower:
            potential_score += score
            found_potential.append(keyword)

    # Job title scoring with graduated bonuses for senior posts
    job_title_scores = {
        "ceo": 25,
        "cto": 25,
        "cfo": 20,
        "vp": 20,
        "director": 15,
        "head": 15,
        "senior": 10,
        "lead": 10,
        "manager": 5,
        "principal": 10,
        "chief": 20
    }
    for title, score in job_title_scores.items():
        if title in job_title_lower:
            potential_score += score
            found_potential.append(f"job_title:{title}")

    # Source is email -> higher potential
    if "email" in source_lower:
        potential_score += 10
        found_potential.append("source:email")

    # Message long (>50 words) -> higher engagement potential
    try:
        word_count = len(message.split())
    except Exception:
        word_count = 0
    if word_count > 50:
        potential_score += 10
        found_potential.append("long_message")

    # Time-based weighting: recent interactions get higher scores
    if timestamp:
        try:
            from datetime import datetime
            if isinstance(timestamp, str):
                email_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                email_date = timestamp
            now = datetime.now(email_date.tzinfo) if email_date.tzinfo else datetime.now()
            days_diff = (now - email_date).days
            if days_diff <= 1:
                potential_score += 15  # Very recent
                found_potential.append("very_recent")
            elif days_diff <= 7:
                potential_score += 10  # Recent week
                found_potential.append("recent_week")
            elif days_diff <= 30:
                potential_score += 5  # Recent month
                found_potential.append("recent_month")
        except Exception:
            pass  # Ignore timestamp errors

    # Negative potential
    for keyword, score in negative_potential_keywords.items():
        if keyword in message_lower:
            potential_score += score
            found_negative_potential.append(keyword)

    # Dynamic NLP-based scoring
    if nlp_keywords:
        # Relevance keywords from NLP
        nlp_relevance_terms = ['demo', 'pricing', 'quote', 'feature', 'trial', 'consultation', 'information', 'detail', 'specification', 'benefit', 'capability']
        for term in nlp_relevance_terms:
            if any(term in kw for kw in nlp_keywords_lower):
                relevance_score += 10
                found_nlp.append(f"nlp_relevance:{term}")

        # Intent keywords from NLP
        nlp_intent_terms = ['buy', 'purchase', 'order', 'interested', 'need', 'urgent', 'deadline', 'asap']
        for term in nlp_intent_terms:
            if any(term in kw for kw in nlp_keywords_lower):
                intent_score += 5
                found_nlp.append(f"nlp_intent:{term}")

        # Potential from NLP (business-focused keywords)
        nlp_potential_terms = ['business', 'company', 'organization', 'team', 'project', 'solution', 'implementation']
        for term in nlp_potential_terms:
            if any(term in kw for kw in nlp_keywords_lower):
                potential_score += 5
                found_nlp.append(f"nlp_potential:{term}")

    # Boost potential if job_title words appear in message
    if job_title_lower:
        job_title_words = set(job_title_lower.split()) - {''}
        for word in job_title_words:
            if len(word) > 2 and word in message_lower:
                potential_score += 5
                found_potential.append(f"job_word:{word}")

    # Boost potential if company words appear in message
    if company_lower:
        company_words = set(company_lower.split()) - {''}
        for word in company_words:
            if len(word) > 2 and word in message_lower:
                potential_score += 3
                found_potential.append(f"company_word:{word}")

    # Total score
    total_score = relevance_score + intent_score + potential_score

    # Determine label based on confidence
    conf = rule_score_to_confidence(total_score)
    if conf > 0.8:
        label = "Hot"
    elif conf < 0.5:
        label = "Cold"
    else:
        label = "Warm"

    # Reason
    all_positive = found_relevance + found_intent + found_potential + found_nlp
    all_negative = found_negative_potential
    positive_count = len(all_positive)
    negative_count = len(all_negative)

    nlp_info = f", NLP: {found_nlp}" if found_nlp else ""

    if all_positive and all_negative:
        if negative_count >= positive_count + 3:
            reason = f"Negative potential indicators: {all_negative}"
        else:
            reason = f"Mixed signals. Relevance: {found_relevance}, Intent: {found_intent}, Potential: {found_potential}{nlp_info}, Negative: {all_negative}"
    elif all_positive:
        reason = f"Positive indicators - Relevance: {found_relevance}, Intent: {found_intent}, Potential: {found_potential}{nlp_info}"
    elif all_negative:
        reason = f"Negative potential indicators: {all_negative}"
    else:
        reason = "No relevant keywords found."

    scores = {
        'relevance': relevance_score,
        'intent': intent_score,
        'potential': potential_score
    }

    return total_score, label, reason, scores

def save_lead_to_csv(name, company, job_title, source, message, label, csv_path=None):
    """Save the lead to the specified CSV file."""
    if csv_path is None:
        csv_path = os.path.join("Data", "testing.csv")

    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Name', 'Company', 'Job title', 'Source', 'Message', 'Label']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header only if file is new
                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    'Name': name,
                    'Company': company,
                    'Job title': job_title,
                    'Source': source,
                    'Message': message,
                    'Label': label
                })
            # If saving to training data, append to Google Sheet
            if csv_path == os.path.join("Data", "csvfile.csv"):
                try:
                    google_sheets.append_entry_to_sheet(name, company, job_title, source, message, label)
                except Exception as e:
                    print(f"Failed to append to Google Sheet: {e}")
            return True  # Success
        except PermissionError:
            if attempt < max_retries - 1:
                print(f"PermissionError: Could not write to {csv_path}. Retrying in 1 second... (attempt {attempt + 1}/{max_retries})")
                time.sleep(1)
            else:
                print(f"PermissionError: Could not write to {csv_path} after {max_retries} attempts. Please close the file if it's open in another program.")
                return False
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            return False
    return False

def ensure_model():
    """Load existing model or train a new one using available CSVs."""
    # Try to load saved model
    clf = models.load_classifier()
    if clf is not None:
        return clf

    # If no saved model, merge csv files and train
    data_dir = "Data"
    primary = os.path.join(data_dir, "csvfile.csv")
    secondary = os.path.join(data_dir, "dummy_leads_dataset_full_names_80.csv")
    merged = os.path.join(data_dir, "merged_training.csv")

    # Read available files
    frames = []
    for path in (primary, secondary):
        if os.path.isfile(path):
            try:
                df = pd.read_csv(path)
                # Keep only required columns
                if {'Message', 'Label'}.issubset(df.columns):
                    frames.append(df[['Message', 'Label']])
            except Exception:
                continue

    if not frames:
        print("No training data found to train the model. Skipping ML model creation.")
        return None

    combined = pd.concat(frames, ignore_index=True)
    # Save merged for transparency
    combined.to_csv(merged, index=False)

    # Train and save model using models.create_and_train_classifier
    clf = models.create_and_train_classifier(csv_path=merged)
    return clf


def load_all_saved_models():
    """Find and load all saved model pickle files matching 'lead_classifier*.pkl'.
    Returns a list of dicts with keys: name, pipeline, source
    """
    models_list = []
    # Friendly names mapping
    name_map = {
        'logistic': 'Logistic Regression',
        'svm': 'Support Vector Machine',
        'naive_bayes': 'Naive Bayes',
        'ensemble': 'Ensemble'
    }
    # look for pickles in current working directory
    pkl_paths = glob.glob(os.path.join('.', 'lead_classifier*.pkl'))
    for p in pkl_paths:
        try:
            with open(p, 'rb') as f:
                obj = pickle.load(f)
            base = os.path.basename(p)
            name = None
            for key, friendly in name_map.items():
                if key in base:
                    name = friendly
                    break
            if name is None:
                name = base
            models_list.append({'name': name, 'pipeline': obj, 'source': p})
        except Exception:
            continue
    return models_list


def _get_prediction_and_probs(pipeline, message):
    """Return (pred_label, top_prob or None, probs_dict or {}) for a single pipeline/estimator."""
    # try to get final estimator (for classes_)
    clf = pipeline
    if hasattr(pipeline, 'named_steps'):
        try:
            clf = list(pipeline.named_steps.values())[-1]
        except Exception:
            clf = pipeline

    classes = getattr(clf, 'classes_', None)

    # prefer predict_proba
    try:
        if hasattr(pipeline, 'predict_proba'):
            probs = pipeline.predict_proba([message])[0]
            pred = pipeline.predict([message])[0]
            probs_dict = dict(zip(classes, probs)) if classes is not None else {}
            return pred, float(max(probs)), probs_dict

        # fallback to decision_function -> softmax/sigmoid
        if hasattr(pipeline, 'decision_function'):
            df = pipeline.decision_function([message])[0]
            df_arr = np.array(df)
            if df_arr.ndim == 0:
                # binary decision value
                prob_pos = 1.0 / (1.0 + np.exp(-float(df_arr)))
                if classes is not None and len(classes) == 2:
                    probs = np.array([1 - prob_pos, prob_pos])
                    probs_dict = dict(zip(classes, probs))
                else:
                    probs_dict = {}
                pred = pipeline.predict([message])[0]
                return pred, float(max(probs)) if classes is not None else float(prob_pos), probs_dict
            else:
                exps = np.exp(df_arr - np.max(df_arr))
                probs = exps / exps.sum()
                probs_dict = dict(zip(classes, probs)) if classes is not None else {}
                pred = pipeline.predict([message])[0]
                return pred, float(max(probs)), probs_dict

        # last resort: only predict available
        pred = pipeline.predict([message])[0]
        return pred, None, {}
    except Exception:
        try:
            pred = pipeline.predict([message])[0]
            return pred, None, {}
        except Exception:
            return None, None, {}


def format_ml_results(ml_results):
    """Nicely format and print ML model results (table-like)."""
    if not ml_results:
        print("No ML models loaded.")
        return

    name_width = max(len(str(r.get('name', 'model'))) for r in ml_results) + 2
    header = f"{'Model':<{name_width}} {'Prediction':<10} {'TopConf':>8}  Probabilities"
    print("\nML model predictions:")
    print(header)
    print('-' * max(len(header), 40))
    for r in ml_results:
        name = r.get('name', 'model')
        pred = r.get('pred', '')
        top = f"{r['top_prob']:.1%}" if r.get('top_prob') is not None else "N/A"
        probs = r.get('probs', {}) or {}
        if probs:
            # sort probabilities descending for clarity
            items = sorted(probs.items(), key=lambda x: -x[1])
            probs_str = '  '.join([f"{k}:{v:.1%}" for k, v in items])
        else:
            probs_str = ''
        print(f"{name:<{name_width}} {pred:<10} {top:>8}  {probs_str}")


def rule_score_to_confidence(score, scale=0.03):
    """Map the integer rule-based score to a pseudo-probability [0,1].

    This uses a sigmoid transform so higher positive scores approach 1.0
    and large negative scores approach 0.0. `scale` controls sensitivity.
    """
    import math
    try:
        return 1.0 / (1.0 + math.exp(-float(score) * float(scale)))
    except Exception:
        return 0.0


def compare_models_with_rule(message, job_title=None, source=None, company=None):
    """Compute rule-based label/confidence and compare to all saved ML models.

    Prints formatted ML outputs and, for each model, whether the model or the
    rule-based scorer has higher confidence.
    Returns a dict with `rule_confidence`, `rule_label`, `ml_results`, `comparisons`.
    """
    score, rule_label, reason, scores = judge_lead(message, job_title, source, company)
    rule_conf = rule_score_to_confidence(score)

    # ensure a model exists (may train/save default)
    _ = ensure_model()
    models_loaded = load_all_saved_models()
    if not models_loaded:
        try:
            clf_obj = models.load_classifier()
            if clf_obj is not None and getattr(clf_obj, 'model', None) is not None:
                models_loaded.append({'name': 'in-memory', 'pipeline': clf_obj.model, 'source': 'memory'})
        except Exception:
            pass

    ml_results = []
    for m in models_loaded:
        pred, top_prob, probs_dict = _get_prediction_and_probs(m['pipeline'], message)
        ml_results.append({'name': m.get('name', 'model'), 'pred': pred, 'top_prob': top_prob, 'probs': probs_dict})

    # print formatted ML table
    format_ml_results(ml_results)

    # compare per-model
    comparisons = []
    print("\nComparison vs rule-based scorer:")
    for r in ml_results:
        model_name = r.get('name')
        m_conf = r.get('top_prob')
        if m_conf is None:
            winner = 'rule' if rule_conf is not None else 'tie'
            diff = None
        else:
            if m_conf > rule_conf:
                winner = 'model'
            elif m_conf < rule_conf:
                winner = 'rule'
            else:
                winner = 'tie'
            diff = (m_conf - rule_conf)

        left = f"Model({model_name}): {m_conf:.2%}" if m_conf is not None else f"Model({model_name}): N/A"
        right = f"Rule: {rule_conf:.2%} (score={score})"
        if diff is None:
            comp_str = f"- {left}  vs  {right}  -> winner: {winner}"
        else:
            comp_str = f"- {left}  vs  {right}  -> winner: {winner} (diff {diff:+.2%})"
        print(comp_str)
        comparisons.append({'model': model_name, 'model_conf': m_conf, 'rule_conf': rule_conf, 'winner': winner, 'diff': diff})

    return {'rule_confidence': rule_conf, 'rule_label': rule_label, 'rule_score': score, 'rule_scores': scores, 'ml_results': ml_results, 'comparisons': comparisons}

# Main program
if __name__ == "__main__":
    print("=== Lead Scorer ===\n")
    
    # Get user input matching CSV format
    name = input("Enter name: ")
    company = input("Enter company: ")
    job_title = input("Enter job title: ")
    source = input("Enter source (e.g., email, LinkedIn, contact form, website form): ")
    message = input("Enter message: ")
    
    # Judge the lead
    score, keyword_label, reason, scores = judge_lead(message, job_title, source, company)

    # Ensure an ML model exists (train if needed), then load all saved models
    # ensure_model will create and save a model if none exist
    _ = ensure_model()

    models_loaded = load_all_saved_models()
    # If nothing on disk, try to load the in-memory model returned by models.load_classifier()
    if not models_loaded:
        try:
            clf_obj = models.load_classifier()
            if clf_obj is not None and getattr(clf_obj, 'model', None) is not None:
                models_loaded.append({'name': 'in-memory', 'pipeline': clf_obj.model, 'source': 'memory'})
        except Exception:
            pass

    ml_results = []
    for m in models_loaded:
        pred, top_prob, probs_dict = _get_prediction_and_probs(m['pipeline'], message)
        ml_results.append({'name': m.get('name', 'model'), 'pred': pred, 'top_prob': top_prob, 'probs': probs_dict})

    # Primary ML label comes from the first loaded model (if any)
    ml_label = ml_results[0]['pred'] if ml_results else None
    ml_prob = ml_results[0]['top_prob'] if ml_results else None

    # Choose final label: prefer ML label if available, otherwise keyword label
    final_label = ml_label if ml_label else keyword_label
    
    # Save to CSV (primary dataset)
    save_lead_to_csv(name, company, job_title, source, message, final_label)

    print(f"\n--- Results ---")
    print(f"Total Score: {score}")
    print(f"Relevance: {scores['relevance']}, Intent: {scores['intent']}, Potential: {scores['potential']}")
    print(f"Label: {keyword_label}")
    if ml_results:
        print("\nML model predictions:")
        for r in ml_results:
            if r['top_prob'] is not None:
                print(f"- {r['name']}: Prediction: {r['pred']} (confidence: {r['top_prob']:.2%})")
                if r['probs']:
                    probs_str = ', '.join([f"{k}:{v:.2%}" for k, v in r['probs'].items()])
                    print(f"  Probabilities: {probs_str}")
            else:
                print(f"- {r['name']}: Prediction: {r['pred']} (confidence: N/A)")
    else:
        print(f"ML Label: {ml_label}")
    print(f"Final Label saved: {final_label}")
    print(f"Reason: {reason}")
    print(f"\nLead appended to Data/testing.csv")
    