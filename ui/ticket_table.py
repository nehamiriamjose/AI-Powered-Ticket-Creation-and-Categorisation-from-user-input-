# ui/ticket_table.py
import streamlit as st
from services.db_service import update_ticket_status
from ui.sla import render_sla


def show_ticket_table(tickets_df):
    st.subheader("📋 Ticket List")

    if tickets_df.empty:
        st.info("No tickets found.")
        return

    status_options = ["Open", "In Progress", "Resolved", "Closed"]

    for _, row in tickets_df.iterrows():
        with st.container():
            c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 3, 2, 2, 2, 2, 2])

            # Ticket ID
            c1.write(f"#{row['id']}")

            # Description
            c2.write(row["description"])

            # Category
            c3.write(row["category"])

            # Priority (highlight HIGH)
            if row["priority"] == "High":
                c4.error("🔥 High")
            else:
                c4.write(row["priority"])

            # Status dropdown
            current_index = status_options.index(row["status"])
            new_status = c5.selectbox(
                "Status",
                status_options,
                index=current_index,
                key=f"status_{row['id']}"
            )

            if new_status != row["status"]:
                update_ticket_status(row["id"], new_status)
                st.rerun()

            # Created time
            c6.write(row["created_at"])

            # SLA column
            with c7:
                render_sla(row["created_at"])

        st.divider()
