import streamlit as st
import json
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
from llama_index.llms.groq import Groq
from sqlalchemy import create_engine

st.set_page_config(page_title="Cloud AI Architect", layout="wide")
st.title("☁️ Enterprise AI (Cloud Edition)")

# SECRETS
groq_api_key = st.secrets.get("GROQ_API_KEY")

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
                description="Useful for questions about text, policies, summaries, or specific documents."
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
                description="Useful for questions about numbers, sales, revenue, counts, and math."
            )
        )
    except Exception as e:
        st.warning(f"SQL Load Error: {e}")
        return None

    # 3. THE ROUTER (With Guardrails)
    # We use the standard selector which is stable across versions
    selector = LLMSingleSelector.from_defaults(llm=llm)
    
    router_engine = RouterQueryEngine(
        selector=selector,
        query_engine_tools=[rag_tool, sql_tool],
        verbose=True
    )
    
    return router_engine

# --- INITIALIZATION ---
with st.spinner("Connecting to Cloud Intelligence..."):
    router = load_cloud_pipeline()

# --- CHAT LOOP ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am online. Ask me about Sales or Documents."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # The Router attempts to pick the right tool
                response = router.query(prompt)
                st.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": str(response.response)})
            
            except ValueError as e:
                # Catch JSON errors specifically and retry or inform
                st.error("I got confused routing that request. Try asking slightly differently.")
            except Exception as e:
                st.error(f"System Error: {e}")