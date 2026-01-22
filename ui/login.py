import streamlit as st
from services.db_service import get_user_by_username


def login_page():
    st.markdown("## 🔐 Login")

    username = st.text_input("Username")

    if st.button("Login"):
        if not username.strip():
            st.error("Please enter a username")
            return

        user = get_user_by_username(username.strip())

        if user:
            # ✅ Save login state
            st.session_state["logged_in"] = True
            st.session_state["username"] = user["username"]
            st.session_state["role"] = user["role"]

            st.success(f"Welcome {user['username']} ({user['role']})")
            st.rerun()
        else:
            st.error("User not found")
