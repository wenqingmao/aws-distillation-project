import streamlit as st
import requests
import os
import json 

# Page configuration
st.set_page_config(layout="wide", page_title="Model Chatbot")
st.title("Distilled SLM: PubMedQA Interface")

# Get the FastAPI backend URL from environment variable
FASTAPI_URL = os.getenv("FASTAPI_BACKEND_URL", "http://localhost:8000/").rstrip('/')

# --- Sidebar for Backend Status and Controls ---
with st.sidebar:
    st.header("Backend Connection")
    backend_status_placeholder = st.empty() # Placeholder for status message

    def check_backend_status():
        try:
            response = requests.get(f"{FASTAPI_URL}/health", timeout=10) # Use health endpoint
            if response.status_code == 200:
                health_data = response.json()
                status = health_data.get("status", "unknown").lower()
                if status.startswith("healthy"):
                    backend_status_placeholder.success(f"Backend Healthy ({health_data.get('device_used_by_model', 'N/A')})")
                    st.caption(f"Model: {health_data.get('model_config_type', 'N/A')}")
                    return True
                else:
                    reason = health_data.get('reason', 'Unknown reason')
                    if health_data.get('model_load_error') not in [None, "None", ""]:
                        reason = health_data.get('model_load_error')
                    backend_status_placeholder.warning(f"Backend Unhealthy: {reason}")
                    return False
            else:
                backend_status_placeholder.warning(f"Backend Status Error ({response.status_code}). Check backend logs.")
                return False
        except requests.exceptions.RequestException as e:
            backend_status_placeholder.error(f"Backend Connection Failed: {type(e).__name__}")
            return False

    if st.button("Refresh Backend Status", key="refresh_status_button"):
        check_backend_status()
    
    st.markdown(f"**API Target:** `{FASTAPI_URL}`")
    st.markdown("---")
    st.header("Chat Controls")
    if st.button("Clear Chat History"):
        # Initialize with a greeting from the assistant
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm ready for PubMedQA questions. What would you like me to analyze?"}]
        st.rerun() # Rerun to update the display

# Initialize/Load chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm ready for PubMedQA questions. What would you like me to analyze?"}]

# Initial backend check on first load
if "backend_checked_initially" not in st.session_state:
    check_backend_status()
    st.session_state.backend_checked_initially = True

# --- Main Chat Area and Info Panel using columns ---
col_chat, col_info = st.columns([2, 1]) # Chat area takes 2/3, Info panel takes 1/3

with col_chat:
    st.subheader("Conversation")
    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # If the message is from assistant and has details, show an expander
            if message["role"] == "assistant" and "details" in message and message["details"]:
                with st.expander("View Prediction Details"):
                    st.json(message["details"])

# Chat input - place it logically after displaying messages
with col_chat:
    if prompt := st.chat_input("Enter your PubMedQA question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Rerun to display the user's message immediately
        st.rerun()

# Process the latest user message if it exists and hasn't been "answered" by the assistant yet
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_input_text = st.session_state.messages[-1]["content"]
    
    if check_backend_status(): # Check backend before making a call
        payload = {"text": user_input_text}
        try:
            api_response = requests.post(f"{FASTAPI_URL}/predict/", json=payload, timeout=30) # Call /predict/
            
            assistant_response_content = "Sorry, I encountered an issue." # Error message
            prediction_details_for_expander = {}

            if api_response.status_code == 200:
                prediction_data = api_response.json()
                predicted_label = prediction_data.get("predicted_label", "N/A") # This will be "No", "Maybe", or "Yes"
                
                assistant_response_content = f"The model's analysis suggests: **{predicted_label}**"
                prediction_details_for_expander = prediction_data # Store the full response for the expander
            else:
                assistant_response_content = f"Error from backend (Status {api_response.status_code}): {api_response.text}"
                prediction_details_for_expander = {"error_details": api_response.text}

            st.session_state.messages.append({
                "role": "assistant", 
                "content": assistant_response_content,
                "details": prediction_details_for_expander 
            })
            st.rerun() # Rerun to display the assistant's new message

        except requests.exceptions.RequestException as e:
            error_msg = f"Could not reach backend prediction service: {type(e).__name__}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg, "details": {"error": str(e)}})
            st.rerun() # Rerun to display the error
    else:
        # If backend is not responsive when user tries to send a message
        no_backend_msg = "Backend is not responding. Cannot get prediction at the moment."
        st.session_state.messages.append({"role": "assistant", "content": no_backend_msg})
        st.rerun() # Rerun to display the status

with col_info:
    st.subheader("💡 Info Panel")
    st.markdown("""
    This is your AI assistant powered by a distilled student model.
    - Enter text in the chat input below the conversation.
    - The assistant will provide a classification: "No", "Maybe", or "Yes".
    - You can view detailed probabilities in the expander below the assistant's response.
    - Use the sidebar to check backend status or clear the chat.
    """)
    # if st.button("Show Raw Chat History (Debug)", key="debug_history"):
    #     st.json(st.session_state.messages)