import os
import sys
import logging
from functools import lru_cache
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()

import hubspot
from hubspot.crm.contacts import ApiException, SimplePublicObjectInput
# Removed HubSpot transactional imports since using Gmail SMTP instead
from lead_scorer import judge_lead, load_all_saved_models, _get_prediction_and_probs, rule_score_to_confidence
from email_automation import send_lead_email

# Configuration from environment
HUBSPOT_ACCESS_TOKEN = os.getenv('HUBSPOT_ACCESS_TOKEN')
HOT_CONFIDENCE_THRESHOLD = float(os.getenv('HOT_CONFIDENCE_THRESHOLD', 0.75))
COLD_CONFIDENCE_THRESHOLD = float(os.getenv('COLD_CONFIDENCE_THRESHOLD', 0.5))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'lead_scorer.log')

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize HubSpot client
if not HUBSPOT_ACCESS_TOKEN:
    logger.error("HUBSPOT_ACCESS_TOKEN not found in environment variables")
    raise ValueError("HUBSPOT_ACCESS_TOKEN is required")
client = hubspot.Client(access_token=HUBSPOT_ACCESS_TOKEN)

def fetch_contacts(limit=100):
    """
    Fetch recent contacts from HubSpot with error handling and logging.

    Args:
        limit (int): Maximum number of contacts to fetch (default: 100)

    Returns:
        list: List of contact objects, or empty list on failure
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching contacts from HubSpot (attempt {attempt + 1})")
            api_response = client.crm.contacts.get_all(limit=10000, properties=['firstname', 'lastname', 'email', 'jobtitle', 'company', 'notes', 'message'])
            if isinstance(api_response, list):
                contacts = api_response
            elif hasattr(api_response, 'results'):
                contacts = api_response.results
            else:
                logger.warning(f"Unexpected response structure: {api_response}")
                return []
            logger.info(f"Successfully fetched {len(contacts)} contacts")
            return contacts
        except ApiException as e:
            logger.error(f"API Exception when fetching contacts (attempt {attempt + 1}): Status {e.status}, Body: {e.body}")
            if attempt == max_retries - 1:
                return []
        except Exception as e:
            logger.error(f"Unexpected error when fetching contacts (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return []
    return []

@lru_cache(maxsize=1)
def load_ml_models():
    """
    Load ML models with caching to avoid repeated loading.

    Returns:
        list: List of loaded ML model dictionaries
    """
    return load_all_saved_models()

def validate_contact_data(properties):
    """
    Validate and sanitize contact data.

    Args:
        properties (dict): Contact properties dictionary

    Returns:
        dict: Sanitized properties

    Raises:
        ValueError: If properties is not a dictionary
    """
    if not isinstance(properties, dict):
        raise ValueError("Properties must be a dictionary")
    
    # Sanitize strings
    sanitized = {}
    for key, value in properties.items():
        if value is None:
            sanitized[key] = ''
        elif isinstance(value, str):
            sanitized[key] = value.strip()[:1000]  # Limit length
        else:
            sanitized[key] = str(value)
    return sanitized

def score_contact(contact, timestamp=None):
    """
    Score a single contact using lead scorer logic with validation and logging.

    Args:
        contact: HubSpot contact object or dict
        timestamp: Optional datetime for time-based scoring

    Returns:
        tuple: (score, label, reason) or (0, 'Cold', error_message) on failure
    """
    try:
        # Handle both object and dict formats
        if hasattr(contact, 'properties'):
            properties = contact.properties
        else:
            properties = contact.get('properties', {})

        properties = validate_contact_data(properties)

        # Extract fields
        firstname = properties.get('firstname', '')
        lastname = properties.get('lastname', '')
        name = f"{firstname} {lastname}".strip()
        company = properties.get('company', '')
        job_title = properties.get('jobtitle', '')
        notes = properties.get('notes', '')
        message_prop = properties.get('message', '')

        # Combine all available information into the message for scoring
        message_parts = [name, company, notes, message_prop]
        message = ' '.join(part for part in message_parts if part).strip()

        contact_id = getattr(contact, 'id', 'unknown')
        logger.info(f"Scoring contact {contact_id}: name='{name}', company='{company}'")

        # If insufficient information, mark as Cold
        if not (message or job_title):
            logger.warning(f"Insufficient information for contact {contact_id}")
            return 0, 'Cold', 'Insufficient information for scoring'

        # Load and predict with all ML models
        combined_input = f"{message} {job_title} {company}".strip()
        models_loaded = load_ml_models()
        ml_preds = []
        ml_confs = []
        for m in models_loaded:
            try:
                pred, top_prob, _ = _get_prediction_and_probs(m['pipeline'], combined_input)
                if pred:
                    ml_preds.append(pred)
                    if top_prob is not None:
                        ml_confs.append(top_prob)
            except Exception as e:
                logger.error(f"Error with model {m.get('name', 'unknown')}: {e}")

        # Determine ML consensus
        if ml_preds:
            from collections import Counter
            pred_counts = Counter(ml_preds)
            most_common_ml = pred_counts.most_common(1)[0][0]
            final_label = most_common_ml
            average_ml_conf = sum(ml_confs) / len(ml_confs) if ml_confs else 0.5
            # Map to score (0-100 based on confidence)
            rule_score = int(average_ml_conf * 100)
        else:
            # Fallback to rule-based if no ML models
            rule_score, rule_label, rule_reason, _ = judge_lead(message, job_title, source='hubspot', company=company, timestamp=timestamp)
            final_label = rule_label
            average_ml_conf = rule_score_to_confidence(rule_score)

        logger.info(f"Contact {contact_id} scored using ML models: final_label={final_label}, confidence={average_ml_conf:.2f}")
        return rule_score, final_label, "ML model consensus"

    except Exception as e:
        contact_id = getattr(contact, 'id', 'unknown')
        logger.error(f"Error scoring contact {contact_id}: {e}")
        return 0, 'Cold', f'Scoring error: {str(e)}'

def update_contact_score(contact_id, score, label):
    """
    Update the contact with score and label in HubSpot.

    Args:
        contact_id (str): HubSpot contact ID
        score (int|float): Lead score
        label (str): Lead label

    Returns:
        bool: True if successful, False otherwise
    """
    if not isinstance(score, (int, float)) or not isinstance(label, str):
        logger.error(f"Invalid score or label for contact {contact_id}: score={score}, label={label}")
        return False

    # Map labels to match HubSpot dropdown options
    label_mapping = {
        'hot': 'hot_lead',
        'Hot': 'hot_lead',
        'warm': 'warm_lead',
        'Warm': 'warm_lead',
        'cold': 'cold_lead',
        'Cold': 'cold_lead'
    }
    mapped_label = label_mapping.get(label, label.lower() + '_lead')  # Default to original if not found

    properties = {
        'lead_score': int(score),
        'lead_label': mapped_label
    }
    simple_public_object_input = SimplePublicObjectInput(properties=properties)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Updating contact {contact_id} with score {score}, label {label} (attempt {attempt + 1})")
            client.crm.contacts.basic_api.update(contact_id, simple_public_object_input)
            logger.info(f"Successfully updated contact {contact_id}")
            return True
        except ApiException as e:
            logger.error(f"API Exception updating contact {contact_id} (attempt {attempt + 1}): Status {e.status}, Body: {e.body}")
            if attempt == max_retries - 1:
                return False
        except Exception as e:
            logger.error(f"Unexpected error updating contact {contact_id} (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return False
    return False

def record_email_sent(contact_id, subject, body):
    """
    Record the sent email details in HubSpot contact's notes with validation.

    Args:
        contact_id (str): HubSpot contact ID
        subject (str): Email subject
        body (str): Email body

    Returns:
        bool: True if successful, False otherwise
    """
    if not all(isinstance(x, str) for x in [contact_id, subject, body]):
        logger.error(f"Invalid parameters for recording email: contact_id={contact_id}, subject={subject}")
        return False

    # Get current notes
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching current notes for contact {contact_id} (attempt {attempt + 1})")
            current_contact = client.crm.contacts.basic_api.get_by_id(contact_id, properties=['notes'])
            current_notes = current_contact.properties.get('notes', '')
            break
        except ApiException as e:
            logger.error(f"API Exception fetching notes for {contact_id} (attempt {attempt + 1}): Status {e.status}")
            if attempt == max_retries - 1:
                current_notes = ''
        except Exception as e:
            logger.error(f"Unexpected error fetching notes for {contact_id} (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                current_notes = ''

    # Append email details to notes
    email_log = f"\n--- Email Sent ---\nSubject: {subject}\nBody: {body[:2000]}...\n"
    updated_notes = current_notes + email_log

    properties = {
        'notes': updated_notes
    }
    simple_public_object_input = SimplePublicObjectInput(properties=properties)
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Recording email sent for contact {contact_id} (attempt {attempt + 1})")
            client.crm.contacts.basic_api.update(contact_id, simple_public_object_input)
            logger.info(f"Successfully recorded email for contact {contact_id}")
            return True
        except ApiException as e:
            error_body = str(e.body)
            if "PROPERTY_DOESNT_EXIST" in error_body and "notes" in error_body:
                logger.warning(f"Notes property does not exist for contact {contact_id}. Skipping email recording.")
                return True
            logger.error(f"API Exception recording email for {contact_id} (attempt {attempt + 1}): Status {e.status}, Body: {e.body}")
            if attempt == max_retries - 1:
                return False
        except Exception as e:
            logger.error(f"Unexpected error recording email for {contact_id} (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return False
    return False

def generate_email_content(label, contact_name, company):
    """
    Generate email subject and body based on lead label.

    Args:
        label (str): Lead label
        contact_name (str): Contact name
        company (str): Company name

    Returns:
        tuple: (subject, body) or ('', '') for unsupported labels
    """
    if label.lower() in ['hot_lead', 'hot']:
        subject = f"Exciting Opportunity: Let's Schedule a Call, {contact_name}"
        company_info = f" from {company}" if company else ""
        body = f"Hi {contact_name},\n\nWe noticed your recent interest{company_info} and believe you're an excellent fit for our services. Based on our analysis, your profile aligns well with our target market. We'd love to schedule a quick call to discuss how we can help you achieve your goals.\n\n\n\nBest regards,\n Sales Team"
    elif label.lower() in ['warm_lead', 'warm']:
        subject = f"Following Up on Our Conversation, {contact_name}"
        company_info = f" at {company}" if company else ""
        body = f"Hi {contact_name},\n\nI hope this email finds you well. I'm reaching out to follow up on your recent interest{company_info} and see if there's anything specific we can assist you with at this time. Perhaps we can explore potential collaboration opportunities or answer any questions you might have about our offerings.\n\n\n\nBest regards,\n Sales Team"
    elif label.lower() in ['cold_lead', 'cold']:
        subject = f"Exclusive Resources to Accelerate Your Business Growth, {contact_name}"
        company_info = f" from {company}" if company else ""
        body = f"Hi {contact_name},\n\nThank you for reaching out{company_info}. We're committed to providing value to potential partners like yourself. Here are some resources that might be helpful:\n- Product brochure: https://example.com/brochure\n- Case studies: https://example.com/case-studies\n- Latest webinar on industry trends: https://example.com/webinar\n\nFeel free to explore these at your convenience.\n\n\n\nBest regards,\n Sales Team"
    else:
        subject = ""
        body = ""
    return subject, body

def process_contact(contact):
    """
    Process a single contact: score, update, and send email if applicable.

    Args:
        contact: HubSpot contact object

    Returns:
        bool: True if processing successful, False otherwise
    """
    contact_id = getattr(contact, 'id', 'unknown')
    logger.info(f"Processing contact {contact_id}")
    
    score, label, reason = score_contact(contact)
    if score is None:
        logger.warning(f"Skipping contact {contact_id} due to scoring failure")
        return False

    if not update_contact_score(contact_id, score, label):
        logger.error(f"Failed to update score for contact {contact_id}")
        return False

    # Send email for all leads (Hot, Warm, Cold) using Gmail SMTP, but skip HubSpot accounts
    if label in ['Hot', 'Warm', 'Cold']:
        properties = validate_contact_data(contact.properties)
        email = properties.get('email', '').strip()
        if not email:
            logger.warning(f"No email found for contact {contact_id}")
            return True

        # Skip sending emails to HubSpot accounts
        if email.lower().endswith('@hubspot.com'):
            logger.info(f"Skipping email to HubSpot account: {email}")
            return True

        firstname = properties.get('firstname', '').strip()
        lastname = properties.get('lastname', '').strip()
        contact_name = f"{firstname} {lastname}".strip() or "Valued Contact"
        company = properties.get('company', '').strip()

        subject, body = generate_email_content(label, contact_name, company)
        if not subject or not body:
            logger.warning(f"Failed to generate email content for contact {contact_id}")
            return True

        if send_lead_email(email, label, contact_name, company):
            record_email_sent(contact_id, subject, body)
        else:
            logger.error(f"Failed to send email to contact {contact_id}")
    return True

def sync_leads():
    """
    Main function to fetch, score, and update leads with comprehensive error handling.

    Orchestrates the complete lead processing pipeline:
    1. Fetch contacts from HubSpot
    2. Score each contact using hybrid rule-based + ML approach
    3. Update contact scores and labels in HubSpot
    4. Send personalized emails for Hot/Warm/Cold leads
    5. Record email details in HubSpot notes
    """
    logger.info("Starting HubSpot lead sync process")
    
    contacts = fetch_contacts()
    if not contacts:
        logger.warning("No contacts fetched from HubSpot")
        return

    processed = 0
    successful = 0
    
    for contact in contacts:
        try:
            if process_contact(contact):
                successful += 1
            processed += 1
        except Exception as e:
            contact_id = getattr(contact, 'id', 'unknown')
            logger.error(f"Unexpected error processing contact {contact_id}: {e}")
            processed += 1

    logger.info(f"Sync complete: {successful}/{processed} contacts processed successfully")