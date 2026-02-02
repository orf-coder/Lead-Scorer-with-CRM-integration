import streamlit as st
import pandas as pd
import logging
import os
import subprocess
import time
from datetime import datetime
from gmail_lead_scorer import process_recent_emails_no_send, send_selected_emails
from HubSpot.hubspot_integration import sync_leads

# Configure page
st.set_page_config(
    page_title="Lead Automation Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .automation-controls {
        background: #e9ecef;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'automation_running' not in st.session_state:
    st.session_state.automation_running = False
if 'automation_thread' not in st.session_state:
    st.session_state.automation_thread = None
if 'last_check' not in st.session_state:
    st.session_state.last_check = None
if 'automation_stats' not in st.session_state:
    st.session_state.automation_stats = {
        'total_processed': 0,
        'emails_sent': 0,
        'hubspot_updated': 0,
        'last_run': None,
        'label_distribution': {'hot_lead': 0, 'warm_lead': 0, 'cold_lead': 0}
    }

def run_automation_check():
    """Single automation check for emails"""
    try:
        leads = process_recent_emails_no_send()
        if leads:
            sent_count, errors = send_selected_emails(leads)

            # Update stats
            st.session_state.automation_stats['total_processed'] += len(leads)
            st.session_state.automation_stats['emails_sent'] += sent_count
            st.session_state.automation_stats['hubspot_updated'] += len(leads)
            st.session_state.automation_stats['last_run'] = datetime.now()

            # Count labels for display and update stats
            label_counts = {'hot_lead': 0, 'warm_lead': 0, 'cold_lead': 0}
            for lead in leads:
                label = lead.get('mapped_label', lead.get('label', 'Unknown'))
                if label in label_counts:
                    label_counts[label] += 1
                # Update global stats
                if label in st.session_state.automation_stats['label_distribution']:
                    st.session_state.automation_stats['label_distribution'][label] += 1

            # Format detailed email-label breakdown
            email_label_parts = []
            for lead in leads:
                email = lead['from_email']
                mapped_label = lead.get('mapped_label', lead.get('label', 'Unknown'))
                display_label = {'hot_lead': 'Hot', 'warm_lead': 'Warm', 'cold_lead': 'Cold'}.get(mapped_label, mapped_label)
                email_label_parts.append(f"{email} ({display_label})")

            email_label_text = ", ".join(email_label_parts[:3])  # Show first 3
            if len(email_label_parts) > 3:
                email_label_text += f" +{len(email_label_parts)-3} more"

            return f"✅ Processed leads: {email_label_text} | Sent {sent_count} emails"
        else:
            return "ℹ️ No new leads found"

    except Exception as e:
        return f"❌ Error: {str(e)}"

def automation_loop():
    """Main automation loop that runs in the UI thread"""
    placeholder = st.empty()

    while st.session_state.automation_running:
        # Show countdown
        for remaining in range(20, 0, -1):
            if not st.session_state.automation_running:
                return
            placeholder.info(f"⏱️ Next email check in {remaining} seconds...")
            time.sleep(1)

        if not st.session_state.automation_running:
            return

        # Perform check
        placeholder.info("🔍 Checking for new emails...")
        result = run_automation_check()
        st.session_state.last_check = result
        placeholder.success(result)

        # Force UI update
        st.rerun()

def start_automation():
    """Start the automation loop"""
    st.session_state.automation_running = True
    # Reset stats for new session
    st.session_state.automation_stats = {
        'total_processed': 0,
        'emails_sent': 0,
        'hubspot_updated': 0,
        'last_run': None,
        'label_distribution': {'hot_lead': 0, 'warm_lead': 0, 'cold_lead': 0}
    }

def stop_automation():
    """Stop the automation loop"""
    st.session_state.automation_running = False

# Main UI
st.markdown('<h1 class="main-header">🤖 Lead Automation Dashboard</h1>', unsafe_allow_html=True)

# Status Overview
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Leads Processed", st.session_state.automation_stats['total_processed'])
with col2:
    st.metric("Emails Sent", st.session_state.automation_stats['emails_sent'])
with col3:
    st.metric("HubSpot Updates", st.session_state.automation_stats['hubspot_updated'])
with col4:
    status = "🟢 Running" if st.session_state.automation_running else "🔴 Stopped"
    st.metric("Status", status)

# Automation Controls
st.markdown("### 🎛️ Automation Controls")
st.markdown('<div class="automation-controls">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("▶️ Start Automation", use_container_width=True, type="primary"):
        start_automation()
        st.success("✅ Automation started! Checking for emails every minute.")

with col2:
    if st.button("⏹️ Stop Automation", use_container_width=True):
        stop_automation()
        st.info("⏹️ Automation stopped.")

with col3:
    last_check = st.session_state.last_check or "No checks performed yet"
    st.markdown(f"**Last Check:** {last_check}")

st.markdown('</div>', unsafe_allow_html=True)

# Show countdown when automation is running
if st.session_state.automation_running:
    automation_loop()

# Manual Operations
st.markdown("### 🔧 Manual Operations")

tab1, tab2, tab3 = st.tabs(["📧 Process Emails", "🔄 HubSpot Sync", "📊 View Stats"])

with tab1:
    st.markdown("Manually process recent emails and send automated responses.")
    if st.button("🚀 Process Emails Now", use_container_width=True):
        with st.spinner("Processing emails..."):
            try:
                leads = process_recent_emails_no_send()
                if leads:
                    sent_count, errors = send_selected_emails(leads)

                    # Update stats
                    st.session_state.automation_stats['total_processed'] += len(leads)
                    st.session_state.automation_stats['emails_sent'] += sent_count
                    st.session_state.automation_stats['hubspot_updated'] += len(leads)
                    st.session_state.automation_stats['last_run'] = datetime.now()

                    st.success(f"✅ Processed {len(leads)} leads and sent {sent_count} emails!")

                    if errors:
                        st.error("Some operations failed:")
                        for error in errors:
                            st.write(f"- {error}")

                    # Show lead details
                    st.markdown("#### 📋 Processed Leads & Sent Emails")
                    for lead in leads:
                        email_sent = "✅ Email sent" if sent_count > 0 else "❌ Email failed"
                        with st.expander(f"{lead['sender_name']} - {lead['label']} Lead - {email_sent}"):
                            st.write(f"**From Email:** {lead['from_email']}")
                            st.write(f"**Reply Sent To:** {lead['from_email']}")
                            st.write(f"**Score:** {lead['score']}")
                            st.write(f"**Subject:** {lead['subject']}")
                            st.text_area("Message Preview", lead['full_message'][:500] + "...", height=100, disabled=True)
                else:
                    st.info("ℹ️ No new leads found in recent emails.")
            except Exception as e:
                st.error(f"❌ Processing failed: {e}")

with tab2:
    st.markdown("Sync existing HubSpot contacts with lead scoring.")
    if st.button("🔄 Run HubSpot Sync", use_container_width=True):
        with st.spinner("Syncing with HubSpot..."):
            try:
                sync_leads()
                st.success("✅ HubSpot sync completed!")
            except Exception as e:
                st.error(f"❌ HubSpot sync failed: {e}")

with tab3:
    st.markdown("View detailed automation statistics.")

    # Automation Stats
    st.subheader("🤖 Automation Statistics")
    stats_df = pd.DataFrame({
        'Metric': ['Total Leads Processed', 'Emails Sent', 'HubSpot Updates', 'Hot Leads', 'Warm Leads', 'Cold Leads', 'Last Run'],
        'Value': [
            st.session_state.automation_stats['total_processed'],
            st.session_state.automation_stats['emails_sent'],
            st.session_state.automation_stats['hubspot_updated'],
            st.session_state.automation_stats['label_distribution']['hot_lead'],
            st.session_state.automation_stats['label_distribution']['warm_lead'],
            st.session_state.automation_stats['label_distribution']['cold_lead'],
            st.session_state.automation_stats['last_run'].strftime('%Y-%m-%d %H:%M:%S') if st.session_state.automation_stats['last_run'] else 'Never'
        ]
    })
    st.dataframe(stats_df, use_container_width=True)

    # Label Distribution Chart
    st.subheader("📊 Lead Label Distribution")
    label_data = st.session_state.automation_stats['label_distribution']

    # Map to display names
    display_labels = {
        'hot_lead': 'Hot',
        'warm_lead': 'Warm',
        'cold_lead': 'Cold'
    }

    if sum(label_data.values()) > 0:
        import plotly.express as px

        # Create pie chart with display names
        display_names = [display_labels.get(k, k) for k in label_data.keys()]
        fig = px.pie(
            values=list(label_data.values()),
            names=display_names,
            title="Distribution of Scored Leads by Label",
            color_discrete_map={
                'Hot': '#dc3545',
                'Warm': '#ffc107',
                'Cold': '#6c757d'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No leads scored yet. Start processing emails to see the distribution chart.")

    # Activity Log
    st.subheader("📝 Activity Log")
    st.info("Activity logging would be implemented here with database storage.")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Keep this dashboard open for continuous monitoring, or run `python gmail_lead_scorer.py` in a separate terminal for background processing.")

if st.session_state.automation_running:
    st.markdown('<div class="status-card">🔄 Automation is running. The system checks for new emails every minute and processes them automatically.</div>', unsafe_allow_html=True)