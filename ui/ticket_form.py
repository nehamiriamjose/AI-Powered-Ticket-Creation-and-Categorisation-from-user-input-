import streamlit as st
from services.api_client import create_ticket
from datetime import datetime


def show_ticket_form():
    st.subheader("➕ Create New Ticket")

    MIN_DESC_LENGTH = 20

    # ------------------------------
    # INITIALIZE SESSION STATE
    # ------------------------------
    if "show_ticket_details" not in st.session_state:
        st.session_state["show_ticket_details"] = False

    if "ticket_created" not in st.session_state:
        st.session_state["ticket_created"] = False

    # ------------------------------
    # CREATE TICKET FORM
    # ------------------------------
    with st.form("create_ticket_form"):
        description = st.text_area(
            "Describe the issue",
            height=120,
            placeholder="Example: Unable to login to VPN from office laptop"
        )

        submit = st.form_submit_button("Generate Ticket")

        if submit:
            description_clean = description.strip()

            # 🚫 EMPTY INPUT
            if not description_clean:
                st.warning("⚠️ Please enter a description.")
                st.session_state["ticket_created"] = False
                st.session_state.pop("latest_ticket", None)
                st.session_state["show_ticket_details"] = False
                return

            # 🚫 SHORT INPUT
            if len(description_clean) < MIN_DESC_LENGTH:
                st.warning("✍️ Please describe the issue in more detail.")
                st.session_state["ticket_created"] = False
                st.session_state.pop("latest_ticket", None)
                st.session_state["show_ticket_details"] = False
                return

            # ✅ VALID INPUT — CALL API
            response = create_ticket(description_clean)

            if response.status_code == 200:
                data = response.json()

                st.session_state["latest_ticket"] = {
                    "ticket_id": data["ticket_id"],
                    "category": data["category"],
                    "priority": data["priority"],
                    "description": data["description"],
                    "status": data["status"],
                    "created_at": data["created_at"],
                }

                st.session_state["ticket_created"] = True
                st.session_state["show_ticket_details"] = False

            else:
                st.session_state["ticket_created"] = False
                st.error(f"❌ Failed to create ticket (Status {response.status_code})")

    # ------------------------------
    # SUCCESS MESSAGE (OUTSIDE FORM)
    # ------------------------------
    if st.session_state.get("ticket_created"):
        st.success("✅ Ticket created successfully")

    # ------------------------------
    # VIEW / HIDE TICKET DETAILS
    # ------------------------------
    if "latest_ticket" in st.session_state:
        ticket = st.session_state["latest_ticket"]

        if st.button(
            "👁 View Ticket Details"
            if not st.session_state["show_ticket_details"]
            else "❌ Hide Ticket Details"
        ):
            st.session_state["show_ticket_details"] = not st.session_state["show_ticket_details"]

        if st.session_state["show_ticket_details"]:
            st.markdown("## 🎫 Ticket Details")

            col1, col2, col3 = st.columns([2, 3, 2])
            with col1:
                st.caption("Ticket ID")
                st.markdown(f"### {ticket['ticket_id']}")
            with col2:
                st.caption("Category")
                st.markdown(f"### {ticket['category']}")
            with col3:
                st.caption("Priority")
                st.markdown(f"### {ticket['priority']}")

            st.divider()

            st.markdown("### 📝 Issue Description")
            st.markdown(
                f"""
                <div style="
                    padding:12px;
                    border-left:4px solid #4CAF50;
                    background-color:rgba(255,255,255,0.03);
                    border-radius:6px;
                ">
                    {ticket['description']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.divider()

            col4, col5 = st.columns([2, 3])
            with col4:
                st.caption("Status")
                st.markdown(
                    f"""
                    <span style="
                        padding:6px 12px;
                        background-color:rgba(76,175,80,0.15);
                        color:#4CAF50;
                        border-radius:12px;
                        font-weight:600;
                    ">
                        {ticket['status']}
                    </span>
                    """,
                    unsafe_allow_html=True
                )

            with col5:
                created_time = datetime.fromisoformat(ticket["created_at"])
                st.caption("Created At")
                st.markdown(
                    f"**{created_time.strftime('%d %b %Y, %I:%M %p')}**"
                )
