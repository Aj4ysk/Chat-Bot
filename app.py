import streamlit as st
import os
import time
from groq import Groq
from typing import List, Dict, Any
import json

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Configure the page
st.set_page_config(
    page_title="LLaMA 3 Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Debug information in sidebar
st.sidebar.title("Debug Info")
st.sidebar.write("Environment Check:")
st.sidebar.write(f"Streamlit Version: {st.__version__}")
st.sidebar.write(f"Python Version: {os.sys.version}")
st.sidebar.write("Secrets Available:", list(st.secrets.keys()) if hasattr(st.secrets, "keys") else "No secrets")

# Custom CSS for better chat interface
st.markdown("""
<style>
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #475063;
    }
    .chat-message .content {
        display: flex;
        flex-direction: column;
    }
</style>
""", unsafe_allow_html=True)

def initialize_groq_client() -> Groq:
    """Initialize the Groq client with API key."""
    try:
        # Try to get API key from Streamlit secrets
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
            st.sidebar.success("✅ API key found in Streamlit secrets")
        # Try to get API key from environment variable
        elif "GROQ_API_KEY" in os.environ:
            api_key = os.environ["GROQ_API_KEY"]
            st.sidebar.success("✅ API key found in environment variables")
        else:
            st.error("""
            ❌ Groq API key not found. Please set it in Streamlit Cloud:

            1. Go to your app settings (three dots ⋮ next to your app)
            2. Click on 'Settings'
            3. Click on 'Secrets'
            4. Add this exact format (replace with your key):
            ```toml
            GROQ_API_KEY = "gsk_...07Hw"
            ```
            
            Make sure to:
            - Include the quotes
            - No extra spaces
            - No newlines
            - Use your actual API key
            """)
            return None

        if not api_key or api_key.strip() == "":
            st.error("❌ API key is empty. Please provide a valid API key.")
            return None

        # Mask the API key for security
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
        st.sidebar.write(f"Using API key: {masked_key}")
        
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Error initializing Groq client: {str(e)}")
        return None

def get_model_response(client: Groq, messages: List[Dict[str, str]], model: str = "llama3-8b-8192") -> str:
    """Get response from Groq API."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting response from Groq API: {str(e)}")
        return None

def simulate_streaming(text: str, placeholder: st.empty) -> None:
    """Simulate streaming response by displaying text gradually."""
    full_response = ""
    for chunk in text.split():
        full_response += chunk + " "
        placeholder.markdown(full_response)
        time.sleep(0.05)  # Adjust delay as needed

def display_chat_message(role: str, content: str) -> None:
    """Display a chat message with proper formatting."""
    with st.chat_message(role):
        if "```" in content:
            # Split content by code blocks
            parts = content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Regular text
                    st.markdown(part)
                else:  # Code block
                    st.code(part, language="python")
        else:
            st.markdown(content)

def main():
    st.title("🤖 LLaMA 3 Chatbot")
    
    # Model selection
    model = st.sidebar.selectbox(
        "Select Model",
        ["llama3-8b-8192", "llama3-70b-8192"],
        index=0
    )
    
    # Clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Initialize Groq client
    client = initialize_groq_client()
    if not client:
        st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Create a placeholder for the assistant's response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # Show loading spinner
            with st.spinner("Thinking..."):
                # Get response from Groq API
                response = get_model_response(
                    client,
                    [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    model
                )
                
                if response:
                    # Simulate streaming
                    simulate_streaming(response, response_placeholder)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Failed to get response from the model. Please try again.")

if __name__ == "__main__":
    main() 