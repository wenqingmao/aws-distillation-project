import streamlit as st
import requests
import os

# Page configuration
st.set_page_config(layout="centered", page_title="Echo Chatbot")
st.title("🤖 Simple Echo Chatbot")

# Get the FastAPI backend URL from environment variable
FASTAPI_URL = os.getenv("FASTAPI_BACKEND_URL", "http://localhost:8000/").rstrip('/')

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm an echo bot. Send me a message, and I'll repeat it using the backend."}]

# --- Backend Connectivity Check ---
backend_status_placeholder = st.sidebar.empty()

def check_backend_status():
    try:
        response = requests.get(f"{FASTAPI_URL}/", timeout=5) # Check root endpoint
        if response.status_code == 200 and response.json().get("message"):
            backend_status_placeholder.success("Backend is responsive!")
            return True
        else:
            backend_status_placeholder.warning(f"Backend might be down (Status {response.status_code}).")
            return False
    except requests.exceptions.RequestException:
        backend_status_placeholder.error("Backend connection failed.")
        return False

# Initial check
check_backend_status()
st.sidebar.button("Re-check Backend Status", on_click=check_backend_status)
st.sidebar.markdown(f"**API Target:** `{FASTAPI_URL}`")

# --- Display existing chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input ---
if prompt := st.chat_input("Say something..."):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send user message to FastAPI backend's /echo/ endpoint
    if check_backend_status(): # Ensure backend is reachable before sending
        payload = {"text": prompt}
        try:
            api_response = requests.post(f"{FASTAPI_URL}/echo/", json=payload, timeout=10)
            if api_response.status_code == 200:
                backend_echo_data = api_response.json()
                bot_response_text = backend_echo_data.get("backend_echoes", "Backend echoed something but the format was unexpected.")
                
                # Display bot's response (which is the echoed message)
                with st.chat_message("assistant"):
                    st.markdown(bot_response_text)
                st.session_state.messages.append({"role": "assistant", "content": bot_response_text})
            else:
                error_message = f"Error from backend (Status {api_response.status_code}): {api_response.text}"
                with st.chat_message("assistant"):
                    st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
        except requests.exceptions.RequestException as e:
            error_message = f"Could not reach backend echo service: {e}"
            with st.chat_message("assistant"):
                st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
    else:
        no_backend_msg = "Backend is not responding. Cannot send message."
        with st.chat_message("assistant"):
            st.warning(no_backend_msg)
        st.session_state.messages.append({"role": "assistant", "content": no_backend_msg})