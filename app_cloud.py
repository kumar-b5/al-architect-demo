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
from llama_index.core.agent import AgentRunner, ReActAgentWorker
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
@st.cache_resource
@st.cache_resource
def load_cloud_pipeline():
    # 1. SETUP MODEL
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    
    if not groq_api_key:
        return None
        
    llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key)
    Settings.llm = llm

    # 2. SETUP TOOLS
    # A. RAG Tool
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        vector_index = load_index_from_storage(storage_context)
        vector_engine = vector_index.as_query_engine()
        rag_tool = QueryEngineTool(
            query_engine=vector_engine,
            metadata=ToolMetadata(
                name="document_search", 
                description="Use this for questions about text, policies, summaries, or documents."
            )
        )
    except Exception as e:
        st.warning(f"RAG Load Error: {e}")
        return None

    # B. SQL Tool
    try:
        engine = create_engine("sqlite:///company_data.db")
        sql_database = SQLDatabase(engine, include_tables=["sales"])
        sql_engine = NLSQLTableQueryEngine(sql_database=sql_database)
        sql_tool = QueryEngineTool(
            query_engine=sql_engine,
            metadata=ToolMetadata(
                name="sales_database", 
                description="Use this for questions about numbers, sales, revenue, and math."
            )
        )
    except Exception as e:
        st.warning(f"SQL Load Error: {e}")
        return None

    # 3. THE AGENT (MANUAL CONSTRUCTION - THE FIX)
    try:
        # We build the worker (logic) and runner (loop) separately
        agent_worker = ReActAgentWorker.from_tools(
            [rag_tool, sql_tool], 
            llm=llm, 
            verbose=True,
            max_iterations=10
        )
        agent = AgentRunner(agent_worker)
        return agent
    except Exception as e:
        st.error(f"Agent Construction Error: {e}")
        return None

# --- APP INTERFACE (Standard Streamlit) ---
with st.spinner("Connecting to Cloud Intelligence..."):
    router = load_cloud_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am live on the Cloud. Ask away."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ... (Top of file remains the same) ...

# HANDLE INPUT
i# ... (Top logic remains same)

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # USE .chat() instead of .query()
                response = router.chat(prompt)
                
                st.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": str(response.response)})
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                if "no such table" in str(e).lower():
                    st.warning("Hint: Did you forget to upload 'company_data.db' to GitHub?")