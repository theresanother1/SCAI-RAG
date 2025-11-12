"""
Intelligent Knowledge Retrieval System - Minimal Frontend
"""

import streamlit as st
import time
from model.model import RAGModel
from rails import input as input_guard
from rails.output import OutputGuardrails
import os
try:
    import secrets_local
    HF_TOKEN = secrets_local.HF
except ImportError:
    HF_TOKEN = os.environ.get("HF_TOKEN")
from helper import ROLE_ASSISTANT, AUTO_ANSWERS, sanitize, Answer
from rag import retriever
from dataclasses import dataclass
from typing import List
from database.setup_db import setup_database
from rag.build_vector_store import build_vector_store

# ============================================
# APPLICATION SETUP
# ============================================

@st.cache_resource
def setup_application():
    """
    Sets up the database and vector store on application startup.
    """
    setup_database()
    build_vector_store()

# ============================================
# BACKEND INTEGRATION
# ============================================

def query_rag_pipeline(user_query: str, model: RAGModel, output_guardRails: OutputGuardrails, input_guardrails: input_guard.InputGuardRails, input_guardrails_active: bool = True, output_guardrails_active: bool = True) -> Answer:
    """
    Query the Hugging Face model with the user query, with input and output guardrails, if enabled. 
    Parameters: 
        - user_query(str): The user input
        - model (RAGModel): Model class for model interaction
        - output_guardRails (OutputGuardrails): Class for output guardrails, checking for hallucinations, relevance etc.
        - input_guardrails_active (bool): Whether or not to have input guard rails active 
        - output_guardrails_active (bool): Whether or not to have output guard rails active 
    Returns: 
        Answer 
    """
    start_time = time.time()

    # 1. Input Guardrails
    if input_guardrails_active:
        checked_answer = input_guardrails.is_valid(user_query)
        if not checked_answer.accepted:
            return Answer(
                answer=checked_answer.reason if checked_answer.reason else "Invalid input. Please try again.", 
                sources=[], 
                processing_time=start_time - time.time()
            )

    # 2. Context Retrieval
    retrieved_docs = retriever.search(user_query)

    # 3. LLM Generation 
    response = model.generate_response(user_query, retrieved_docs)
    sources = [{"title": str(sanitize(doc))} for doc in retrieved_docs]


    # 4. Output Guardrails    
    if output_guardrails_active: 
        gr_result = output_guardRails.check(user_query, response, retrieved_docs)
        response = output_guardRails.redact_svnrs(response)

    else: 
        end_time = time.time()
        return Answer(
            answer = sanitize(response), 
            sources = sources,
            processing_time =  end_time - start_time
        )
    
    end_time = time.time()
    
    # 5. Final Answer
    if all(gr_result.passed for gr_result in gr_result.values()): 
        return Answer(
            answer = sanitize(response), 
            sources = sources,
            processing_time =  end_time - start_time
        )
    
    else: 
        return Answer(
            answer = output_guardRails.format_guardrail_issues(gr_result, response),
            sources = sources,
            processing_time = end_time - start_time
        )

    


# ============================================
# HAUPTANWENDUNG
# ============================================

def main():
    st.set_page_config(
        page_title="Knowledge Retrieval System",
        page_icon="🤖",
        layout="wide"  # Changed to wide for better dashboard layout
    )
    
    # Import dashboard after page config
    from experimental_dashboard import render_experiment_dashboard
    
    setup_application()
    
    # Use sidebar for navigation instead of tabs to avoid state issues
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["💬 Chat Interface", "🧪 Experiments"],
        index=0
    )
    
    if page == "💬 Chat Interface":
        render_chat_interface()
    elif page == "🧪 Experiments":
        render_experiment_dashboard()

def render_chat_interface():
    """Render the main chat interface"""
    
    # Header
    st.title("🎓 University Knowledge Assistant")
    st.markdown("""
    **Welcome to the University RAG System!** 
    
    This intelligent assistant helps you find information about our university database using advanced AI technology. 
    Here's what you can ask about:
    
    📚 **Student Information**: Questions about enrolled students and their courses  
    👨‍🏫 **Faculty Details**: Information about professors and their departments  
    📖 **Course Catalog**: Details about available courses and instructors  
    📊 **Academic Data**: Enrollment statistics and university insights
    
    **How it works:** The system uses Retrieval-Augmented Generation (RAG) to search through our knowledge database 
    and provides accurate, context-aware answers with source references.
    
    *Try asking: "Who teaches computer science?" or "What courses is Maria taking?"*
    """)
    st.markdown("---")

    model = RAGModel(HF_TOKEN)
    output_guardrails = OutputGuardrails()
    input_guardrails = input_guard.InputGuardRails()
    
    # Chat-Historie initialisieren
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Welcome message
        st.session_state.messages.append({
            "role": ROLE_ASSISTANT,
            "content": "Hello! I can help you with questions about our knowledge database. What would you like to know?",
            "sources":[]
        })
    

    if 'responses' not in st.session_state:
        st.session_state.responses = []

    # Chat-Historie anzeigen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show sources if available
            if message["sources"]:
                with st.expander("📚 Sources Used"):
                    for source in message["sources"]:
                        st.write(f"• {source['title']}")
    
    # Chat input - placed at the bottom
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt,"sources":[]})
        
        # Generate RAG response
        with st.spinner("Searching database and generating answer..."):
            # RAG Pipeline aufrufen
            print(prompt)
            response = query_rag_pipeline(prompt, model, output_guardrails, input_guardrails)

            # Save answer in history
            st.session_state.messages.append({
                "role": ROLE_ASSISTANT,
                "content": response.answer if response.answer else AUTO_ANSWERS.UNEXPECTED_ERROR.value,
                "sources": response.sources
            })
        
        # Rerun to display the new messages
        st.rerun()
            
        



if __name__ == "__main__":
    main()
