import streamlit as st
from llama_index.llms.groq import Groq
# We removed other imports to isolate the connection issue

st.set_page_config(page_title="Connection Test")
st.title("🚑 Cloud Connection Test")

# 1. Debug Secrets
# This checks if Streamlit can actually read your API key
api_key = st.secrets.get("GROQ_API_KEY")

if not api_key:
    st.error("❌ CRITICAL: Secret 'GROQ_API_KEY' not found.")
    st.info("Go to Streamlit Dashboard -> App -> Settings -> Secrets and paste your key.")
    st.stop()
else:
    # We show the first few characters to prove it read something
    st.success(f"✅ API Key found (starts with: {api_key[:5]}...)")

# 2. Test Groq Connection
# This attempts a simple 'Hello' to the AI
try:
    llm = Groq(model="llama-3.1-8b-instant", api_key=api_key)
    response = llm.complete("Hello, are you online?")
    st.write("### Groq Response:")
    st.success(response)
except Exception as e:
    # THIS IS THE KEY: It prints the specific error to the UI
    st.error(f"❌ Connection Failed: {e}")