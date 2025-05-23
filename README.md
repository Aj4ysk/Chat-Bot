# LLaMA 3 Chatbot with Streamlit and Groq

A web-based chatbot application built with Streamlit that uses Groq's LLaMA 3 API for generating responses.

## Features

- Clean and modern chat interface
- Support for both LLaMA 3 8B and 70B models
- Chat history persistence using Streamlit session state
- Code block formatting for technical responses
- Simulated streaming responses
- Error handling and graceful fallbacks
- Secure API key management

## Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Groq API key:

   Option 1: Environment variable
   ```bash
   # On Windows
   set GROQ_API_KEY=your-api-key-here
   
   # On Linux/Mac
   export GROQ_API_KEY=your-api-key-here
   ```

   Option 2: Streamlit secrets (recommended for deployment)
   Create a file `.streamlit/secrets.toml` with:
   ```toml
   GROQ_API_KEY = "your-api-key-here"
   ```

## Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Usage

1. Select your preferred LLaMA 3 model from the sidebar (8B or 70B)
2. Type your message in the chat input at the bottom
3. Wait for the model's response
4. Use the "Clear Chat" button in the sidebar to reset the conversation

## Deployment

This application can be easily deployed on Streamlit Cloud:

1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your repository
4. Add your Groq API key in the secrets management section
5. Deploy!

## Notes

- The application uses simulated streaming for a better user experience
- Code blocks in responses are automatically formatted
- The chat history is maintained during the session
- Error handling is implemented for API failures and connection issues #   C h a t - B o t  
 #   A i - C h a t - B o t  
 #   A i - C h a t - B o t  
 #   A i - C h a t - B o t  
 #   C h a t - B o t  
 #   C h a t - B o t  
 