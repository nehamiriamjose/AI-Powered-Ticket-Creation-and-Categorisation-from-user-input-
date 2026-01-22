# Metrics & analytics
import streamlit as st
from datetime import datetime


def average_resolution_time(tickets_df):
    resolved = tickets_df[tickets_df["status"] == "Resolved"]

    if resolved.empty:
        return "N/A"

    times = []
    for _, row in resolved.iterrows():
        created = datetime.fromisoformat(row["created_at"])
        updated = datetime.fromisoformat(row["updated_at"])
        times.append((updated - created).total_seconds() / 3600)

    return f"{sum(times)/len(times):.2f} hrs"


def show_metrics(tickets_df):
    if tickets_df.empty:
        st.info("No tickets available yet.")
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Tickets", len(tickets_df))
    col2.metric("Open Tickets", len(tickets_df[tickets_df["status"] == "Open"]))
    col3.metric(
        "High Priority Tickets",
        len(tickets_df[tickets_df["priority"] == "High"])
    )
    col4.metric("Closed Tickets", len(tickets_df[tickets_df["status"] == "Closed"]))
    col5.metric("Avg Resolution Time", average_resolution_time(tickets_df))
