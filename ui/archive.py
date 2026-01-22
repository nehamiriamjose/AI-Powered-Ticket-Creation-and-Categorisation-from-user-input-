import streamlit as st


def show_closed_tickets(tickets_df):
    st.subheader("📦 Closed Tickets Archive")

    closed = tickets_df[tickets_df["status"] == "Closed"]

    if closed.empty:
        st.info("No closed tickets.")
    else:
        st.dataframe(closed)
