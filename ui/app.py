import streamlit as st
from login import login_page
from ticket_form import show_ticket_form
from ticket_table import show_ticket_table
from dashboard import show_metrics
from services.db_service import fetch_tickets

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Ticket Management System",
    layout="wide"
)

# -------------------------------------------------
# LOAD CSS
# -------------------------------------------------
with open("ui/css/lavender.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------------------------------------
# SESSION DEFAULTS
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# -------------------------------------------------
# AUTH ROUTING
# -------------------------------------------------
if not st.session_state["logged_in"]:
    login_page()
    st.stop()

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
st.sidebar.markdown("## 🎟 AI Ticket System")
st.sidebar.write(f"👤 User: {st.session_state.get('user')}")

page = st.sidebar.radio(
    "Navigation",
    ["Create Ticket", "View Tickets", "Dashboard"]
)

if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()

# -------------------------------------------------
# PAGE CONTENT
# -------------------------------------------------
tickets_df = fetch_tickets()

if page == "Create Ticket":
    show_ticket_form()

elif page == "View Tickets":
    show_ticket_table(tickets_df)

elif page == "Dashboard":
    show_metrics(tickets_df)
