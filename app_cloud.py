import streamlit as st
import os
from llama_index.core import (
    StorageContext, 
    load_index_from_storage, 
    SQLDatabase, 
    Settings,
    VectorStoreIndex
)
from llama_index.core.query_engine import NLSQLTableQueryEngine, RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq  # <--- NEW IMPORT
from sqlalchemy import create_engine

st.set_page_config(page_title="Cloud AI Architect", layout="wide")
st.title("☁️ Enterprise AI (Cloud Edition)")

# SECRETS MANAGEMENT
# We get the API key from Streamlit's secret storage (secure)
# If running locally, make sure you set this in your terminal or .env
groq_api_key = st.secrets.get("GROQ_API_KEY") 

@st.cache_resource
def load_cloud_pipeline():
    # 1. SETUP EMBEDDINGS (Local CPU is fine for this)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    
    # 2. SETUP LLM (The Switch to Groq)
    if not groq_api_key:
        st.error("GROQ_API_KEY not found!")
        return None
        
    llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key)
    Settings.llm = llm

    # 3. SETUP TOOLS (Same as before)
    # A. RAG
    try:
        # We assume you uploaded the './storage' folder to GitHub
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        vector_index = load_index_from_storage(storage_context)
        vector_engine = vector_index.as_query_engine()
        rag_tool = QueryEngineTool(
            query_engine=vector_engine,
            metadata=ToolMetadata(
                name="document_search", 
                description="For qualitative questions/policies."
            )
        )
    except Exception as e:
        st.warning(f"RAG Load Error: {e}")
        return None

    # B. SQL
    try:
        # We assume you uploaded 'company_data.db' to GitHub
        engine = create_engine("sqlite:///company_data.db")
        sql_database = SQLDatabase(engine, include_tables=["sales"])
        sql_engine = NLSQLTableQueryEngine(sql_database=sql_database)
        sql_tool = QueryEngineTool(
            query_engine=sql_engine,
            metadata=ToolMetadata(
                name="sales_database", 
                description="For quantitative/sales data."
            )
        )
    except Exception as e:
        st.warning(f"SQL Load Error: {e}")
        return None

    # 4. ROUTER
    return RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=[rag_tool, sql_tool],
        verbose=True
    )

# --- APP INTERFACE (Standard Streamlit) ---
with st.spinner("Connecting to Cloud Intelligence..."):
    router = load_cloud_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am live on the Cloud. Ask away."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = router.query(prompt)
        st.markdown(response.response)
        st.session_state.messages.append({"role": "assistant", "content": str(response.response)})