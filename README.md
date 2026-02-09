# Lead Scorer - Intelligent CRM Integration System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An intelligent AI-powered lead scoring system that combines rule-based analysis with machine learning to automatically evaluate and categorize potential customers from Gmail and other sources, with seamless HubSpot CRM integration.

## 🚀 Features

### Core Functionality
- **AI-Powered Scoring**: Combines rule-based analysis with machine learning for accurate lead evaluation
- **Email Automation**: Monitors Gmail inbox for new messages and scores leads automatically
- **Smart Responses**: Sends category-specific email templates based on lead score
- **CRM Integration**: Syncs with HubSpot CRM to create/update contacts with detailed lead information
- **Real-Time Dashboard**: Web-based dashboard for monitoring lead activity and statistics
- **Multi-Model Ensemble**: 4 machine learning models (Logistic Regression, Naive Bayes, SVM, and Ensemble) working together for optimal performance

### Lead Classification System
```
+---------------------------------------------------------------------+
|                    LEAD CLASSIFICATION SYSTEM                       |
+-----------------+-----------------+---------------------------------+
|    HOT          |    WARM         |    COLD                         |
+-----------------+-----------------+---------------------------------+
| High Potential  | Moderate Interest| Low Potential                  |
| Ready to Buy    | Needs Nurturing | Not Ready                      |
| Immediate Action| Follow-up Needed| Educational Content            |
+-----------------+-----------------+---------------------------------+
| Score: 75-100   | Score: 50-75    | Score: 0-50                     |
+-----------------+-----------------+---------------------------------+
```

## 📋 Prerequisites

- Python 3.8 or higher
- Gmail account with API access
- HubSpot CRM account with API token
- Google Cloud Platform project for Gmail API
- Internet connection

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/lead-scorer.git
cd lead-scorer
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### 1. Create Required Files

#### `.env` File
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

#### `Google/google_credentials.json` File
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

#### `Google/token.json` File
This file will be auto-generated when you first authenticate with Google.

### 2. Obtain API Credentials

#### HubSpot Access Token
1. Go to [HubSpot Developer Portal](https://developers.hubspot.com/)
2. Create a private app
3. Set CRM scopes (contacts, objects)
4. Copy the access token

#### Gmail App Password
1. Go to Google Account > Security
2. Enable 2-Step Verification
3. Go to App Passwords
4. Create a new app password for "Mail"

#### Google API Credentials
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable Gmail API
4. Create OAuth 2.0 credentials
5. Download JSON and rename to `google_credentials.json`

## 🚀 Usage

### Run the Streamlit App
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Available Pages

1. **Main Dashboard**: Overview of lead statistics and recent activity
2. **Lead Scorer**: Manual lead entry and scoring
3. **Automation Settings**: Configure email automation and CRM sync
4. **Chatbot**: AI-powered assistant for lead-related questions

### Email Automation
1. Go to the "Automation Settings" page
2. Enable Gmail monitoring
3. Configure frequency and email filters
4. The system will automatically process new emails

## 📊 Project Structure

```
Lead Scorer/
├── app.py                 # Main Streamlit application
├── automation_dashboard.py # Automation dashboard
├── gmail_lead_scorer.py   # Gmail API integration
├── email_automation.py    # Email sending functionality
├── hubspot_sync.py        # HubSpot CRM integration
├── lead_scorer.py         # Lead scoring logic
├── models.py              # Machine learning models
├── run_cv.py              # Cross-validation utilities
├── check_overfitting.py   # Overfitting detection
├── test_predict.py        # Prediction tests
├── Data/                  # CSV data files
├── Google/                # Google API credentials and token
├── HubSpot/               # HubSpot integration files
├── pages/                 # Additional Streamlit pages
├── Visualization/         # Visualization assets and scripts
├── Chatbot/               # AI chatbot module
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
├── .env.example           # Example environment variables
└── README.md             # This file
```

## 🧠 Machine Learning Models

### Model Architecture
The system uses a hybrid approach combining:
- **Rule-Based Analysis**: Keyword matching, job title analysis, negative indicator detection
- **Machine Learning Ensemble**: 4 models working together:
  - Logistic Regression (85-90% accuracy)
  - Naive Bayes (80-85% accuracy)
  - Support Vector Machine (88-92% accuracy)
  - Ensemble Classifier (90-95% accuracy)

### Training Pipeline
```
1. Load Data: CSV with labeled leads (hot/warm/cold)
2. Preprocess: Clean, tokenize, remove stopwords, lemmatize
3. Vectorize: TF-IDF transformation (5000+ terms, unigrams + bigrams)
4. Train: Cross-validation, model fitting
5. Evaluate: Accuracy, precision, recall, F1-score
6. Save: Serialize to .pkl files
```

## 📈 Scoring System

### Scoring Dimensions (100-point scale)

1. **Relevance Score (40%)** - Measures interest in product/service
   - Positive keywords: demo (+25), pricing (+25), quote (+20), trial (+20)
   
2. **Intent Score (40%)** - Measures buying readiness
   - Positive keywords: buy (+15), purchase (+15), urgent (+10), asap (+10)

3. **Potential Score (20%)** - Measures customer value
   - Job titles: CEO/CTO/CFO (+25), VP/Director (+20), Manager (+15)
   - Negative indicators: student (-50), job seeker (-40), no budget (-40)

## 🚀 Deployment

### Streamlit Community Cloud
1. Push code to GitHub (exclude sensitive files)
2. Connect GitHub account to Streamlit
3. Select repository
4. Add secrets in Streamlit settings
5. Deploy main branch

### Local Deployment
```bash
# Run with custom port
streamlit run app.py --server.port 8080

# Run in headless mode
streamlit run app.py --server.headless true
```

## 📚 Documentation

- **Complete Documentation**: [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) - Detailed system architecture and technical specifications
- **IEEE Format**: [DOCUMENTATION_IEEE.tex](DOCUMENTATION_IEEE.tex) - Academic paper format
- **HubSpot Integration**: [HUBSPOT_EMAIL_DOCS.md](HUBSPOT_EMAIL_DOCS.md) - Email integration documentation

## 🐛 Troubleshooting

### Common Issues

1. **Gmail Authentication Failed**: Ensure `google_credentials.json` is correct and token is fresh
2. **HubSpot API Errors**: Check access token permissions and validity
3. **Model Loading Issues**: Ensure all `.pkl` model files are present in root directory
4. **Email Sending Errors**: Verify SMTP settings and app password

### Logging
Check `lead_scorer.log` for detailed error information and debugging.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or support, please open an issue or contact the development team.

---
