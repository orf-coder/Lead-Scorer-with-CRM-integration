# Lead Scorer - CRM Integration

Hybrid rule-based + ML lead scoring system with Gmail and HubSpot integration.

## Features
- Email processing from Gmail
- Lead scoring using ML models
- HubSpot CRM integration
- Streamlit dashboard

## Setup Instructions

### 1. Create Required Files

You need to create the following files before running the application:

#### `.env` file
Create a `.env` file in the root directory:
```env
# HubSpot Configuration
HUBSPOT_ACCESS_TOKEN=your_hubspot_access_token_here

# Email Configuration
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password_here
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Scoring Thresholds
HOT_CONFIDENCE_THRESHOLD=0.75
COLD_CONFIDENCE_THRESHOLD=0.5

# Logging
LOG_LEVEL=INFO
LOG_FILE=lead_scorer.log
```

To get a HubSpot access token:
1. Go to HubSpot Developer Portal
2. Create a private app
3. Set CRM scopes (contacts, objects)
4. Copy the access token

For Gmail app password:
1. Go to Google Account > Security
2. Enable 2-Step Verification
3. Go to App Passwords
4. Create a new app password for "Mail"

#### `Google/google_credentials.json` file
Create this file for Gmail API access:
```json
{
  "installed": {
    "client_id": "your_client_id.apps.googleusercontent.com",
    "project_id": "your_project_id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "your_client_secret",
    "redirect_uris": ["http://localhost"]
  }
}
```

To get Google credentials:
1. Go to Google Cloud Console
2. Create a new project
3. Enable Gmail API
4. Create OAuth 2.0 credentials
5. Download JSON and rename to google_credentials.json

#### `Google/token.json` file
This file will be auto-generated when you first authenticate with Google.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Locally
```bash
streamlit run app.py
```

## Deployment on Streamlit Community Cloud

1. Push this code to GitHub (excluding sensitive files)
2. Connect your GitHub account to Streamlit
3. Select this repository
4. Add the following secrets in Streamlit settings:
   - `HUBSPOT_ACCESS_TOKEN`
   - `SENDER_EMAIL`
   - `SENDER_PASSWORD`
5. Deploy the `main` branch

## Project Structure
```
Lead Scorer/
├── app.py                 # Main Streamlit app
├── automation_dashboard.py # Automation dashboard
├── gmail_lead_scorer.py   # Gmail integration
├── email_automation.py    # Email sending
├── hubspot_sync.py        # HubSpot integration
├── lead_scorer.py         # Lead scoring logic
├── models.py              # ML models
├── run_cv.py              # Cross-validation
├── check_overfitting.py   # Overfitting check
├── test_predict.py        # Prediction tests
├── Data/                  # CSV data files
├── Google/                # Google API credentials
├── HubSpot/               # HubSpot integration
├── pages/                 # Streamlit pages
├── Visualization/         # Visualizations
└── Chatbot/               # Chatbot module
```

## License
MIT
