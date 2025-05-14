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
        api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("Groq API key not found. Please set it in your environment variables or Streamlit secrets.")
            return None
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
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