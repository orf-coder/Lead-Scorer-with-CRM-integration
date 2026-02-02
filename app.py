import streamlit as st
import pandas as pd
import logging
import os
import subprocess
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from lead_scorer import (
    judge_lead,
    load_all_saved_models,
    _get_prediction_and_probs,
    rule_score_to_confidence,
    save_lead_to_csv,
)
from gmail_lead_scorer import process_recent_emails_no_send, send_selected_emails
from HubSpot.hubspot_integration import sync_leads

def get_email_content(lead_label, contact_name, company=''):
    """Generate email content based on lead label."""
    if lead_label.lower() in ['hot_lead', 'hot']:
        subject = f"Exciting Opportunity, {contact_name}"
        company_info = f" from {company}" if company else ""
        body = f"Hi {contact_name},\n\nWe noticed your recent interest{company_info} and believe you're an excellent fit for our services. Based on our analysis, your profile aligns well with our target market. We'd love to schedule a quick call to discuss how we can help you achieve your goals. Let us know what time works for you.\n\nBest regards,\nSales Team"
    elif lead_label.lower() in ['warm_lead', 'warm']:
        subject = f"Follow Up, {contact_name}"
        company_info = f" at {company}" if company else ""
        body = f"Hi {contact_name},\n\nI hope this email finds you well. I'm reaching out to follow up on your recent interest{company_info} and see if there's anything specific we can assist you with at this time. Perhaps we can explore potential collaboration opportunities or answer any questions you might have about our offerings.\n\nBest regards,\nSales Team"
    elif lead_label.lower() in ['cold_lead', 'cold']:
        subject = f"Resources to Accelerate Your Business Growth, {contact_name}"
        company_info = f" from {company}" if company else ""
        body = f"Hi {contact_name},\n\nThank you for reaching out{company_info}. We're committed to providing value to potential partners like yourself. Here are some resources that might be helpful:\n- Product brochure: https://example.com/brochure\n- Case studies: https://example.com/case-studies\n-  Webinar on industry trends: https://example.com/webinar\n\nFeel free to explore these at your convenience.\n\nBest regards,\nSales Team"
    else:
        subject = f"Message from Sales Team, {contact_name}"
        body = f"Hi {contact_name},\n\nThank you for your message. We'll get back to you soon.\n\nBest regards,\nSales Team"
    return subject, body

def simplify_reason(reason):
    """Simplify the scoring reason for display."""
    if "Positive indicators" in reason:
        return "Positive indicators"
    elif "Negative potential indicators" in reason:
        return "Negative indicators"
    elif "Mixed signals" in reason:
        return "Mixed signals"
    else:
        return reason

def parse_lead_from_text(text):
    """
    Parse lead information from a text string.

    The text is expected to contain key-value pairs separated by colons, one per line.
    Keys are normalized to lowercase with underscores replacing spaces.

    Args:
        text (str): The text containing lead information in key:value format.

    Returns:
        dict: A dictionary with parsed key-value pairs. Keys are normalized.
    """
    lines = text.split('\n')
    data = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            data[key] = value
    return data

# Configure logging
log_file = 'lead_scorer_app.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Theme state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Scoring state
if 'scored' not in st.session_state:
    st.session_state.scored = False

# Page configuration
st.set_page_config(
    page_title="Lead Scorer",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Main title section
st.title("📊 Lead Scoring System")
st.markdown("Hybrid rule-based and machine learning lead classification")
st.caption("💡 **Keyboard Shortcuts:** Tab navigation, Enter to submit, Ctrl+R to refresh")
with st.expander("🚀 Quick Start Guide", expanded=False):
        st.markdown("""
        **Welcome!** This system helps you score and respond to leads automatically.

        **Tabs Overview:**
        - **📝 Input & Scoring**: Manually score individual leads
        - **📋 Results**: View detailed scoring results and comparisons
        - **🤖 Automation**: Automate email processing and responses
        - **📊 Dashboard**: Overall analytics and trends

        **Getting Started:**
        1. Set up your email credentials in the Automation tab
        2. Try scoring a sample lead in Input & Scoring
        3. Explore the Results for detailed analysis
        4. Enable automation for hands-free lead processing

        **Tips:**
        - The system uses both rule-based keywords and ML models for accurate scoring
        - Automation sends personalized responses based on lead quality
        - Check the Dashboard for performance insights
        """)

# Main tabs
tab_options = ["📝 Input & Scoring", "📋 Results", "🤖 Automation", "📊 Dashboard", "❓ Help & FAQ"]
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = tab_options[0]
active_tab = st.radio("Navigate", tab_options, index=tab_options.index(st.session_state.active_tab), key="tab_radio", horizontal=True, label_visibility="collapsed")
st.session_state.active_tab = active_tab

# Apply theme
light_css = """
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.hot-label { color: #ff4444; font-weight: bold; }
.warm-label { color: #ff8800; font-weight: bold; }
.cold-label { color: #0088ff; font-weight: bold; }
[data-testid="stAppViewContainer"] {
    background-color: #e0e0e0;
    color: black;
}
[data-testid="stSidebar"] {
    background-color: #e0e0e0;
    color: black;
}
.stSidebar * {
    color: black !important;
}
h1, h2, h3, h4, h5, h6 {
    color: black !important;
    font-weight: bold !important;
}
.stTextInput label, .stSelectbox label, .stTextArea label, .stFileUploader label, .stCheckbox label, .stSegmentedControl label {
    color: black !important;
}
.stSegmentedControl div {
    background-color: black !important;
    color: white !important;
}
.stButton button {
    border-radius: 8px;
    transition: all 0.3s ease;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
}
/* Consistent text alignment and margins */
.stMarkdown, .stText, .stCaption, .stHeader, .stSubheader {
    margin-left: 0 !important;
    margin-right: 0 !important;
    text-align: left !important;
}
.stContainer, .stColumn {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}
.stExpander {
    margin-left: 0 !important;
    margin-right: 0 !important;
}
</style>
"""

dark_css = """
<style>
.metric-card {
    background-color: #2e2e2e;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    color: white;
}
.hot-label { color: #ff6666; font-weight: bold; }
.warm-label { color: #ffaa00; font-weight: bold; }
.cold-label { color: #66aaff; font-weight: bold; }
[data-testid="stAppViewContainer"] {
    background-color: black;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: black;
    color: white;
}
.stSidebar * {
    color: white !important;
}
h1, h2, h3, h4, h5, h6 {
    color: white !important;
    font-weight: bold !important;
}
.stTextInput label, .stSelectbox label, .stTextArea label, .stFileUploader label, .stCheckbox label, .stSegmentedControl label {
    color: white !important;
}
.stButton button {
    border-radius: 8px;
    transition: all 0.3s ease;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(255,255,255,0.2);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
}
</style>
"""

if st.session_state.theme == 'light':
    st.markdown(light_css, unsafe_allow_html=True)
else:
    st.markdown(dark_css, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This AI-powered system combines:
    - **Rule-based scoring** (keywords, job title, source)
    - **ML models** (Logistic Regression, SVM, Naive Bayes)
    - **Confidence comparison** (rule vs ML models)
    """)
    st.divider()
    st.markdown("**🤖 Available Models:**")
    models_list = load_all_saved_models()
    models_count = len(models_list)
    if models_list:
        for model in models_list:
            st.text(f"• {model['name']}")
    else:
        st.text("• No models available")
    st.divider()
    st.markdown("**📊 System Status:**")
    col_status1, col_status2 = st.columns(2)
    with col_status1:
        st.metric("Trained Models", models_count)
    with col_status2:
        processing_status = "Ready" if models_count > 0 else "Setup Required"
        if models_count > 0:
            st.success(f"✅ {processing_status}")
        else:
            st.warning(f"⚠️ {processing_status}")

    # Performance indicator
    if models_count > 0:
        st.caption("🚀 System Performance: High (All models loaded)")
    else:
        st.caption("⏳ System Performance: Limited (Training required)")

if active_tab == tab_options[0]:
    st.header("📝 Manual Lead Scoring")
    st.markdown("Enter lead details to score them using our hybrid AI system.")
    st.info("💡 **Tip:** Fill in as many fields as possible for more accurate scoring. The system analyzes keywords, job titles, and company info.")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Name", placeholder="John Doe", key="name")
        st.caption("Enter the lead's full name.")
        company = st.text_input("Company", placeholder="Acme Corp", key="company")
        st.caption("Enter the company name.")

    with col2:
        job_title = st.text_input("Job Title", placeholder="CTO, Director, etc.", key="job_title")
        st.caption("Enter the job title for better scoring.")
        source = st.segmented_control(
            "Source",
            ["email", "LinkedIn", "contact form", "website form", "phone"],
            key="source"
        )
        st.caption("Select the source of the lead.")

    message = st.text_area(
        "Message / Lead Description",
        placeholder="Enter the lead message or description here...",
        height=120,
        key="message",
        help="The main content that will be analyzed for scoring. Include any relevant details about the lead's needs or background."
    )
    st.caption("💡 Provide the lead's message or any additional description for comprehensive analysis.")

    # Scoring button
    if st.button("🔍 Score Lead", type="primary", use_container_width=True):
        with st.spinner("Scoring lead..."):
            name = st.session_state.name
            company = st.session_state.company
            job_title = st.session_state.job_title
            source = st.session_state.source
            final_message = st.session_state.message

            logging.info(f"Lead scoring initiated - Name: {name}, Company: {company}, Job: {job_title}, Source: {source}, Message length: {len(final_message)}")
            if not final_message.strip():
                st.error("Please enter a message to score")
                logging.warning("Scoring attempted without message")
                st.session_state.scored = False
            else:
                # Compute rule-based score
                rule_score, rule_label, rule_reason, _ = judge_lead(final_message, job_title, source, company)
                rule_confidence = rule_score_to_confidence(rule_score)
                logging.info(f"Rule-based scoring - Score: {rule_score}, Label: {rule_label}, Confidence: {rule_confidence:.2%}")
                final_label = rule_label

                # Initialize variables
                consensus = rule_label
                ml_preds = []

                # Load and predict with all ML models
                combined_input = f"{final_message} {job_title} {company}".strip()
                models_loaded = load_all_saved_models()
                ml_results = []
                for m in models_loaded:
                    try:
                        pred, top_prob, probs_dict = _get_prediction_and_probs(m['pipeline'], combined_input)
                        ml_results.append({
                            'name': m.get('name', 'model'),
                            'pred': pred,
                            'top_prob': top_prob,
                            'probs': probs_dict
                        })
                        top_str = f"{top_prob:.2%}" if top_prob is not None else "N/A"
                        logging.info(f"ML Model {m.get('name')}: Prediction {pred}, Confidence {top_str}")
                    except Exception as e:
                        logging.error(f"Failed to load/predict with model {m.get('name')}: {e}")
                        st.warning(f"Could not load model {m.get('name')}: {e}")

                # Store results in session state
                st.session_state.scored = True
                st.session_state.rule_score = rule_score
                st.session_state.rule_label = rule_label
                st.session_state.rule_reason = rule_reason
                st.session_state.rule_confidence = rule_confidence
                st.session_state.ml_results = ml_results
                st.session_state.consensus = consensus
                st.session_state.final_label = final_label
                st.session_state.final_message = final_message
                st.session_state.scored_name = name
                st.session_state.scored_company = company
                st.session_state.scored_job_title = job_title
                st.session_state.scored_source = source

                # Switch to Results tab
                st.session_state.active_tab = tab_options[1]
                st.rerun()

if active_tab == tab_options[1]:
    st.header("📋 Detailed Scoring Results")
    st.markdown("View comprehensive analysis of lead scoring, including rule-based vs ML model comparisons.")
    st.info("💡 **How it works:** The system combines keyword analysis with machine learning predictions for robust lead qualification.")

    # Results filters
    if st.session_state.scored:
        st.subheader("🔍 Filter Results")
        col_filter1, col_filter2, col_filter3 = st.columns(3)

        with col_filter1:
            show_rule = st.checkbox("Show Rule-Based Details", value=True, key="show_rule")
        with col_filter2:
            show_ml = st.checkbox("Show ML Model Details", value=True, key="show_ml")
        with col_filter3:
            show_charts = st.checkbox("Show Analysis Charts", value=True, key="show_charts")

    if not st.session_state.scored:
        st.info("Enter details in the Input & Scoring tab to see results here.")
    elif st.session_state.scored:
        rule_score = st.session_state.rule_score
        rule_label = st.session_state.rule_label
        rule_reason = st.session_state.rule_reason
        rule_confidence = st.session_state.rule_confidence
        ml_results = st.session_state.ml_results
        consensus = st.session_state.consensus
        final_label = st.session_state.final_label
        final_message = st.session_state.final_message
        name = st.session_state.scored_name
        company = st.session_state.scored_company
        job_title = st.session_state.scored_job_title
        source = st.session_state.scored_source

        # Display rule-based result
        st.divider()
        st.header("📋 Rule-Based Scoring")

        rule_col1, rule_col2 = st.columns(2)

        with rule_col1:
            st.metric("Rule Score", f"{rule_score}", delta="Integer Score")

        with rule_col2:
            label_class = f"{rule_label}-label"
            st.markdown(f"<div class='{label_class}'>Label: {rule_label}</div>", unsafe_allow_html=True)

        st.info(f"**Reason**: {rule_reason}")

        # Display ML results
        if ml_results:
            st.divider()
            st.header("🤖 Machine Learning Models")

            # Create ML results dataframe for display
            ml_data = []
            for r in ml_results:
                ml_data.append({
                    "Model": r['name'],
                    "Prediction": r['pred']
                })

            ml_df = pd.DataFrame(ml_data)
            st.dataframe(ml_df, use_container_width=True, hide_index=True)


            # Comparison: Rule vs ML
            st.divider()
            st.header("⚖️ Rule vs ML Comparison")

            comparison_data = []
            for r in ml_results:
                comparison_data.append({
                    "Model": r['name'],
                    "ML Prediction": r['pred'],
                    "Rule Prediction": rule_label,
                    "Match": "✅" if r['pred'] == rule_label else "❌"
                })

            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Summary and recommendation
            st.divider()
            st.header("📌 Summary & Recommendation")

            # Determine consensus
            ml_preds = [r['pred'] for r in ml_results if r['pred'] is not None]
            ml_confs = [r['top_prob'] for r in ml_results if r['top_prob'] is not None]
            if ml_preds:
                from collections import Counter
                pred_counts = Counter(ml_preds)
                most_common_ml = pred_counts.most_common(1)[0][0]
                consensus = most_common_ml
                average_ml_conf = sum(ml_confs) / len(ml_confs) if ml_confs else None
            else:
                consensus = rule_label
                average_ml_conf = None

            # Calculate final confidence as average of rule and ML
            if average_ml_conf is not None:
                final_confidence = (rule_confidence + average_ml_conf) / 2
            else:
                final_confidence = rule_confidence

            final_label = consensus

            # Override final label based on final confidence thresholds
            if final_confidence < 0.5:
                final_label = 'Cold'
            elif final_confidence > 0.75:
                final_label = 'Hot'

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Rule Prediction", rule_label)

            with col2:
                if ml_preds:
                    st.metric("ML Consensus", consensus)
                else:
                    st.metric("ML Consensus", "N/A")

            with col3:
                st.markdown(f"<h3 style='text-align: center;'>Final: <span class='{final_label.lower()}-label'>{final_label}</span></h3>",
                            unsafe_allow_html=True)

            with col4:
                st.metric("Final Confidence", f"{final_confidence:.1%}")

            # Recommendation based on final label
            st.divider()
            st.subheader("🎯 Recommended Action")
            if final_label == 'Hot':
                options = ["Assign to SDR", "Schedule Discovery Call", "Send Premium Email Template"]
            elif final_label == 'Warm':
                options = ["Send Follow-up Email", "Add to Nurture Campaign", "Offer Free Trial"]
            else:  # Cold
                options = ["Discard Lead", "Add to Cold List", "Monitor for Future Engagement"]

            st.markdown("**Recommended Actions:**")
            for option in options:
                st.markdown(f"- {option}")

            # Interactive Charts Section
            st.divider()
            st.header("📊 Analysis Dashboard")

            # Model Confidence Distribution
            if ml_results:
                st.subheader("Model Confidence Distribution")
                # Filter to only models with valid confidences
                valid_results = [r for r in ml_results if r.get('top_prob') is not None]
                if valid_results:
                    model_names = [r['name'] for r in valid_results]
                    confidences = [r['top_prob'] for r in valid_results]

                    fig_conf = px.bar(
                        x=model_names,
                        y=confidences,
                        labels={'x': 'Model', 'y': 'Confidence'},
                        title="Model Prediction Confidences",
                        color=confidences,
                        color_continuous_scale='RdYlGn'
                    )
                    fig_conf.update_layout(height=400)
                    st.plotly_chart(fig_conf, use_container_width=True)

            # Rule vs ML Comparison Chart
            if ml_results:
                st.subheader("Rule-Based vs ML Models")
                comparison_data = []
                for r in ml_results:
                    comparison_data.append({
                        'Model': r['name'],
                        'Type': 'ML Model',
                        'Confidence': r['top_prob'] or 0
                    })
                comparison_data.append({
                    'Model': 'Rule-Based',
                    'Type': 'Rule-Based',
                    'Confidence': rule_confidence
                })

                df_comp = pd.DataFrame(comparison_data)
                fig_comp = px.bar(
                    df_comp,
                    x='Model',
                    y='Confidence',
                    color='Type',
                    barmode='group',
                    title="Rule-Based vs ML Model Confidences",
                    labels={'Confidence': 'Confidence Score'}
                )
                fig_comp.update_layout(height=400)
                st.plotly_chart(fig_comp, use_container_width=True)


            # Export options
            st.divider()
            col_export1, col_export2 = st.columns(2)

            with col_export1:
                # Export single result
                single_result = {
                    'Name': name,
                    'Company': company,
                    'Job_Title': job_title,
                    'Source': source,
                    'Message': final_message,
                    'Rule_Label': rule_label,
                    'ML_Consensus': consensus if ml_preds else 'N/A',
                    'Final_Label': final_label,
                    'Rule_Score': rule_score,
                    'Confidence': final_confidence
                }
                if st.button("💾 Save to Training Data", use_container_width=True):
                    # Use absolute path based on script location
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    csv_path = os.path.join(script_dir, "Data", "csvfile.csv")
                    success = save_lead_to_csv(name, company, job_title, source, final_message, final_label, csv_path=csv_path)
                    if success:
                        st.success(f"✅ Lead saved as **{final_label}** to Data/csvfile.csv for training")
                        logging.info(f"Lead saved to training data - Label: {final_label}")
                        # Trigger retraining check in background
                        subprocess.Popen(["python", "auto_retrain.py"])
                        st.info("Entry saved successfully to CSV.")
                    else:
                        st.error("Failed to save lead to CSV after multiple attempts. Please ensure the file is not open in another program.")
                        logging.error("Failed to save lead to CSV")

            with col_export2:
                # Export results as JSON
                import json
                results_data = {
                    'lead_info': {
                        'name': name,
                        'company': company,
                        'job_title': job_title,
                        'source': source
                    },
                    'scoring': {
                        'rule_score': rule_score,
                        'rule_label': rule_label,
                        'rule_confidence': rule_confidence,
                        'final_label': final_label,
                        'final_confidence': final_confidence
                    },
                    'ml_results': ml_results,
                    'message': final_message[:500] + '...' if len(final_message) > 500 else final_message
                }
                json_str = json.dumps(results_data, indent=2, default=str)
                st.download_button(
                    label="📥 Export Results (JSON)",
                    data=json_str,
                    file_name=f"lead_scoring_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

        else:
            # Attempt to trigger training automatically in the background
            st.warning("No ML models found. Attempting to start training in background...")
            logging.warning("No ML models found; launching training script run_train_all.py")
            try:
                # Start training in a background process so Streamlit stays responsive
                subprocess.Popen(["python", "run_train_all.py"])
                st.info("Training started in background. Refresh this app after training completes.")
            except Exception as e:
                st.error("Failed to start background training. Please run `python run_train_all.py` locally.")
                logging.error(f"Failed to launch training: {e}")

if active_tab == tab_options[2]:
    st.header("🤖 Email Automation")
    st.markdown("Automatically process incoming emails, score leads, and send personalized responses.")
    st.info("💡 **Setup Required:** Configure Gmail API credentials first. Then enable auto-processing for hands-free lead management.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("HubSpot Sync")
        st.markdown("Sync and score existing HubSpot contacts, update scores, and send automated emails.")
        if st.button("🔄 Run HubSpot Sync", use_container_width=True):
            with st.spinner("Running HubSpot sync... This may take a few minutes."):
                try:
                    sync_leads()
                    st.success("✅ HubSpot sync completed successfully!")
                    logging.info("HubSpot sync completed via web app")
                except Exception as e:
                    st.error(f"❌ HubSpot sync failed: {e}")
                    logging.error(f"HubSpot sync failed via web app: {e}")

    with col2:
        st.subheader("Gmail Lead Processing")
        st.markdown("Process recent Gmail messages, score leads, and automatically send emails.")

        auto_send = st.checkbox("Auto-send emails to all leads", value=True, key="auto_send")

        if st.button("📧 Process and Send Emails", use_container_width=True):
            # Check if credentials exist
            creds_path = "Google/google_credentials.json"
            if not os.path.exists(creds_path):
                st.error("❌ Gmail API credentials not found!")
                st.markdown("**Setup Required:**")
                st.markdown("1. Go to [Google Cloud Console](https://console.cloud.google.com/)")
                st.markdown("2. Create a project and enable Gmail API")
                st.markdown("3. Create OAuth 2.0 credentials")
                st.markdown("4. Download `credentials.json` and rename to `google_credentials.json`")
                st.markdown("5. Place it in the `Google/` folder")
                st.markdown("6. Run the app to authenticate")
                st.info("💡 Check the Help tab for detailed setup instructions.")
            else:
                with st.spinner("Processing recent emails and sending automated responses..."):
                    try:
                        leads = process_recent_emails_no_send()
                        if leads:
                            # Auto-send to all leads
                            sent_count, errors = send_selected_emails(leads)
                            if sent_count > 0:
                                st.success(f"✅ Processed {len(leads)} leads and sent {sent_count} automated responses.")
                            else:
                                st.warning(f"⚠️ Processed {len(leads)} leads but no emails were sent.")
                            logging.info(f"Automated email processing completed for {len(leads)} leads, {sent_count} emails sent via web app")

                            # Show errors if any
                            if errors:
                                st.error("❌ Some operations failed:")
                                for error in errors:
                                    st.write(f"- {error}")

                            # Show results
                            st.subheader("📊 Processing Results")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("Total Leads Processed", len(leads))
                            with col2:
                                hot_count = sum(1 for l in leads if l['label'] == 'Hot')
                                st.metric("Hot Leads", hot_count)
                            with col3:
                                warm_count = sum(1 for l in leads if l['label'] == 'Warm')
                                st.metric("Warm Leads", warm_count)
                            with col4:
                                cold_count = sum(1 for l in leads if l['label'] == 'Cold')
                                st.metric("Cold Leads", cold_count)
                            with col5:
                                st.metric("Emails Sent", sent_count)

                            # List of processed leads
                            st.subheader("📋 Processed Leads")
                            for lead in leads:
                                with st.expander(f"{lead['sender_name']} ({lead['from_email']}) - Score: {lead['score']}, Label: {lead['label']}"):
                                    st.markdown("**Lead Message:**")
                                    st.text_area("", value=lead['full_message'], height=100, disabled=True, key=f"msg_{lead['msg_id']}")

                                    # Generate outgoing email content
                                    subject, body = get_email_content(lead['label'].lower(), lead['sender_name'])

                                    st.markdown("**Outgoing Email Subject:**")
                                    st.text(subject)

                                    st.markdown("**Outgoing Email Body:**")
                                    st.text_area("", value=body, height=150, disabled=True, key=f"out_{lead['msg_id']}")
                        else:
                            st.info("ℹ️ No new leads found in recent emails.")
                    except Exception as e:
                        st.error(f"❌ Failed to process emails: {e}")


if active_tab == tab_options[3]:
    st.header("📊 Analytics Dashboard")
    st.markdown("Track performance trends, lead distributions, and system effectiveness over time.")
    st.info("💡 **Insights:** Monitor how your lead scoring improves and identify patterns in lead quality.")

    # Lead Scoring Trends (if we have historical data)
    @st.cache_data
    def load_trends_data():
        if os.path.exists("Data/csvfile.csv"):
            return pd.read_csv("Data/csvfile.csv", on_bad_lines='skip', encoding='cp1252')
        return pd.DataFrame()

    try:
        df_trends = load_trends_data()
        if len(df_trends) > 0:
            st.subheader("Lead Scoring Trends")

            # Label distribution
            label_counts = df_trends['Label'].value_counts()
            fig_labels = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                title="Overall Lead Label Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_labels, use_container_width=True)

            # Source distribution
            if 'Source' in df_trends.columns:
                source_counts = df_trends['Source'].value_counts()
                fig_sources = px.bar(
                    x=source_counts.index,
                    y=source_counts.values,
                    title="Lead Sources Distribution",
                    labels={'x': 'Source', 'y': 'Count'},
                    color=source_counts.values,
                    color_continuous_scale='Blues'
                )
                fig_sources.update_layout(height=400)
                st.plotly_chart(fig_sources, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load trend data: {e}")

if active_tab == tab_options[4]:
    st.header("❓ Help & FAQ")
    st.markdown("Get help with using the Lead Scoring System.")

    with st.expander("🔧 Setup & Configuration", expanded=False):
        st.markdown("""
        **Gmail API Setup:**
        1. Go to Google Cloud Console
        2. Create a project and enable Gmail API
        3. Create credentials (OAuth 2.0 Client ID)
        4. Download `google_credentials.json` to the `Google/` folder
        5. Run the app once to authenticate

        **HubSpot Integration:**
        - Set `HUBSPOT_ACCESS_TOKEN` in `.env`
        - Ensure proper permissions for contacts and deals

        **Environment Variables:**
        - `SENDER_EMAIL`: Your Gmail address
        - `SENDER_PASSWORD`: App password (not regular password)
        - `SMTP_SERVER`: Usually smtp.gmail.com
        """)

    with st.expander("📊 How Scoring Works", expanded=False):
        st.markdown("""
        **Rule-Based Scoring:**
        - Analyzes keywords in message, job title, company
        - Assigns points for relevance, intent, and potential
        - Considers recency and source quality

        **ML Models:**
        - Trained on historical lead data
        - Predicts Hot/Warm/Cold labels
        - Combines multiple algorithms for accuracy

        **Final Label:**
        - Consensus between rule-based and ML
        - Confidence thresholds determine final classification
        """)

    with st.expander("🤖 Automation Features", expanded=False):
        st.markdown("""
        **Email Processing:**
        - Scans recent Gmail messages
        - Filters out no-reply and self-emails
        - Scores each valid lead

        **Auto-Response:**
        - Sends personalized emails based on lead quality
        - Hot: Exciting opportunity follow-up
        - Warm: General follow-up
        - Cold: Resources and information

        **HubSpot Sync:**
        - Updates contact scores and labels
        - Records email interactions
        """)

    with st.expander("🔍 Troubleshooting", expanded=False):
        st.markdown("""
        **Common Issues:**
        - **No ML models found**: Run training scripts first
        - **Gmail auth failed**: Check credentials and token
        - **HubSpot errors**: Verify API token and permissions
        - **Scoring errors**: Check input data format

        **Performance Tips:**
        - Train models regularly with new data
        - Monitor confidence levels
        - Review and adjust keyword weights
        """)

    with st.expander("📈 Understanding Results", expanded=False):
        st.markdown("""
        **Label Meanings:**
        - **Hot**: High-quality leads, immediate follow-up
        - **Warm**: Promising leads, nurture campaign
        - **Cold**: Low-priority, informational response

        **Confidence Scores:**
        - Higher = More certain prediction
        - Compare rule vs ML for consistency
        - Use for quality assurance
        """)

    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    Lead Scoring System v1.0 | Built with Streamlit<br>
    Rule-Based + ML Hybrid Approach<br>
    For support, check the documentation or GitHub issues.
    </div>
    """, unsafe_allow_html=True)