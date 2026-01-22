# 🔑 ENTRY POINT (ORCHESTRATOR)
import streamlit as st
import time

from services.db_service import init_db, fetch_tickets
from ui.dashboard import show_metrics
from ui.ticket_form import show_ticket_form
from ui.ticket_table import show_ticket_table
from ui.filters import apply_filters
from ui.login import login_page

# -------------------------------------------------
# SESSION STATE SETUP
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# -------------------------------------------------
# INITIALIZE DATABASE
# -------------------------------------------------
init_db()

st.set_page_config(
    page_title="AI Ticket Management System",
    layout="wide"
)

# -------------------------------------------------
# LOGIN GATE
# -------------------------------------------------
if not st.session_state["logged_in"]:
    login_page()
    st.stop()

# -------------------------------------------------
# LOGOUT BUTTON + USER INFO
# -------------------------------------------------
col1, col2 = st.columns([6, 1])

with col1:
    st.title("🎫 AI Ticket Management Dashboard")

with col2:
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

# -------------------------------------------------
# FETCH DATA
# -------------------------------------------------
tickets_df = fetch_tickets()

# -------------------------------------------------
# ROLE-BASED UI CONTROL (LIGHTWEIGHT)
# -------------------------------------------------
role = st.session_state.get("role", "User")

if role in ["Support", "Admin"]:
    show_metrics(tickets_df)

st.divider()

if role == "User":
    st.subheader("🧑 User View")
    show_ticket_form()

elif role == "Support":
    st.subheader("🛠️ Support Team View")

elif role == "Admin":
    st.subheader("👑 Admin View")

st.divider()

# -------------------------------------------------
# FILTERS + TABLES
# -------------------------------------------------
tickets_df = apply_filters(tickets_df)

tab1, tab2 = st.tabs(["📋 Active Tickets", "📦 Closed Tickets"])

with tab1:
    show_ticket_table(tickets_df[tickets_df["status"] != "Closed"])

with tab2:
    show_ticket_table(tickets_df[tickets_df["status"] == "Closed"])

# -------------------------------------------------
# AUTO REFRESH (SAFE)
# -------------------------------------------------
auto = st.checkbox("🔄 Auto Refresh (5s)")

if auto:
    time.sleep(5)
    st.rerun()
