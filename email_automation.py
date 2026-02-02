import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
load_dotenv()

# Configuration from environment
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))

# Setup logging
logger = logging.getLogger(__name__)

def validate_email_params(recipient_email, lead_label, contact_name):
    """Validate email sending parameters."""
    if not isinstance(recipient_email, str) or not recipient_email.strip():
        raise ValueError("Recipient email must be a non-empty string")
    if not isinstance(lead_label, str) or not lead_label.strip():
        raise ValueError("Lead label must be a non-empty string")
    if not isinstance(contact_name, str) or not contact_name.strip():
        raise ValueError("Contact name must be a non-empty string")
    
    # Basic email validation
    if '@' not in recipient_email or '.' not in recipient_email:
        raise ValueError("Invalid email format")

def send_lead_email(recipient_email, lead_label, contact_name, company='', custom_message=''):
    """
    Send automated email based on lead label using Gmail SMTP with validation and logging.

    Args:
        recipient_email (str): Email address of the lead
        lead_label (str): Lead classification ('hot_lead', 'cold_lead', etc.)
        contact_name (str): Name of the contact
        company (str): Company name (optional)
        custom_message (str): Additional message to include (optional)

    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Validate inputs
        validate_email_params(recipient_email, lead_label, contact_name)
        
        # Check required config
        if not all([SENDER_EMAIL, SENDER_PASSWORD, SMTP_SERVER, SMTP_PORT]):
            logger.error("Email configuration incomplete")
            return False

        logger.info(f"Preparing email to {recipient_email} for {lead_label}")

        # Email content based on label
        if lead_label.lower() in ['hot_lead', 'hot lead', 'hot']:
            subject = f"Exciting Opportunity, {contact_name}"
            company_info = f" from {company}" if company else ""
            body = f"Hi {contact_name},\n\nWe noticed your recent interest{company_info} and believe you're an excellent fit for our services. Based on our analysis, your profile aligns well with our target market. We'd love to schedule a quick call to discuss how we can help you achieve your goals. Let us know what time works for you.\n\n{custom_message}\n\nBest regards,\nSales Team"
        elif lead_label.lower() in ['warm_lead', 'warm lead', 'warm']:
            subject = f"Follow Up, {contact_name}"
            company_info = f" at {company}" if company else ""
            body = f"Hi {contact_name},\n\nI hope this email finds you well. I'm reaching out to follow up on your recent interest{company_info} and see if there's anything specific we can assist you with at this time. Perhaps we can explore potential collaboration opportunities or answer any questions you might have about our offerings.\n\n{custom_message}\n\nBest regards,\nSales Team"
        elif lead_label.lower() in ['cold_lead', 'cold lead', 'cold']:
            subject = f"Resources to Accelerate Your Business Growth, {contact_name}"
            company_info = f" from {company}" if company else ""
            body = f"Hi {contact_name},\n\nThank you for reaching out{company_info}. We're committed to providing value to potential partners like yourself. Here are some resources that might be helpful:\n- Product brochure: https://example.com/brochure\n- Case studies: https://example.com/case-studies\n-  Webinar on industry trends: https://example.com/webinar\n\nFeel free to explore these at your convenience.\n\n{custom_message}\n\nBest regards,\nSales Team"
        else:
            logger.warning(f"Skipping email for unsupported label: {lead_label}")
            return False

        # Create message
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Send email with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending email to {recipient_email} (attempt {attempt + 1})")
                server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30)
                server.starttls()
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
                server.quit()
                logger.info(f"Email sent successfully to {recipient_email}")
                return True
            except smtplib.SMTPAuthenticationError:
                logger.error(f"SMTP authentication failed for {recipient_email}")
                return False
            except smtplib.SMTPRecipientsRefused:
                logger.error(f"Recipient refused: {recipient_email}")
                return False
            except Exception as e:
                logger.error(f"Failed to send email to {recipient_email} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return False
        return False

    except ValueError as e:
        logger.error(f"Validation error for email to {recipient_email}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending email to {recipient_email}: {e}")
        return False