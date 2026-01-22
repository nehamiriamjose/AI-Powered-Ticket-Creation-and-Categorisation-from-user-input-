 # Filters & search
import streamlit as st


def apply_filters(tickets_df):
    st.subheader("🔍 Filters")

    col1, col2 = st.columns(2)

    status_filter = col1.selectbox(
        "Filter by Status",
        ["All", "Open", "In Progress", "Resolved", "Closed"]
    )

    priority_filter = col2.selectbox(
        "Filter by Priority",
        ["All", "High", "Medium", "Low"]
    )

    if status_filter != "All":
        tickets_df = tickets_df[tickets_df["status"] == status_filter]

    if priority_filter != "All":
        tickets_df = tickets_df[tickets_df["priority"] == priority_filter]

    return tickets_df
