# ui/sla.py
from datetime import datetime
import streamlit as st


def get_sla_status(created_at: str):
    """
    Returns (color, hours_passed)
    """
    created_time = datetime.fromisoformat(created_at)
    hours_passed = (datetime.now() - created_time).total_seconds() / 3600

    if hours_passed < 2:
        return "green", hours_passed
    elif hours_passed < 6:
        return "yellow", hours_passed
    else:
        return "red", hours_passed


def render_sla(created_at: str):
    color, hours = get_sla_status(created_at)
    hours = round(hours, 2)

    if color == "green":
        st.success(f"SLA OK — {hours} hrs")
    elif color == "yellow":
        st.warning(f"SLA Warning — {hours} hrs")
    else:
        st.error(f"SLA Breached — {hours} hrs")
