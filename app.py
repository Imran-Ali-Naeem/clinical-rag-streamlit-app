import streamlit as st
import os
import time
from typing import List, Tuple
import traceback
from pathlib import Path
import sys

# Check and install missing packages
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
    from langchain.schema import Document
except ImportError as e:
    st.error(f"❌ Missing required packages: {e}")
    st.info("Please install required packages: pip install -r requirements.txt")
    st.stop()

# LLM
try:
    from openai import OpenAI
    import openai
except ImportError:
    st.error("❌ OpenAI package not installed. Run: pip install openai")
    st.stop()

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Clinical RAG Assistant",
    page_icon="🏥",
    layout="wide"
)

# ==================== CACHED RESOURCES ====================
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """Load embedding model"""
    try:
        # Use a reliable, fast model
        model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True
            }
        )
        return model
    except Exception as e:
        st.error(f"❌ Failed to load embedding model: {e}")
        return None

@st.cache_resource(show_spinner="Initializing LLM client...")
def setup_llm_client():
    """Setup OpenAI client"""
    try:
        # Check for API key in multiple places
        api_key = None
        
        # Try Streamlit secrets first
        if hasattr(st, 'secrets') and 'openai_api_key' in st.secrets:
            api_key = st.secrets['openai_api_key']
        # Try environment variable
        elif 'OPENAI_API_KEY' in os.environ:
            api_key = os.environ['OPENAI_API_KEY']
        else:
            st.error("❌ OpenAI API key not found")
            st.info("""
            Add your API key to:
            1. Streamlit Cloud: App settings → Secrets
            2. Local: Create `.streamlit/secrets.toml` with:
                openai_api_key = "your-key-here"
            """)
            return None
        
        client = OpenAI(api_key=api_key, timeout=30.0)
        
        # Test connection
        try:
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            st.sidebar.success("✅ OpenAI API connected")
            return client
        except Exception as e:
            st.error(f"❌ OpenAI API test failed: {str(e)[:100]}")
            return client  # Still return client for now
            
    except Exception as e:
        st.error(f"❌ LLM setup error: {e}")
        return None

@st.cache_resource(show_spinner="Loading FAISS vector database...")
def load_faiss_index(_embedding_model):
    """Load or create FAISS index"""
    try:
        if _embedding_model is None:
            return None
            
        index_path = Path("faiss_index")
        
        # Try to load existing index
        if index_path.exists():
            try:
                vector_store = FAISS.load_local(
                    str(index_path),
                    _embedding_model,
                    allow_dangerous_deserialization=True
                )
                st.sidebar.success(f"✅ Loaded FAISS index with {vector_store.index.ntotal} vectors")
                return vector_store
            except Exception as e:
                st.warning(f"⚠️ Could not load existing index: {e}")
        
        # Create new index with sample data
        st.sidebar.info("🔄 Creating new FAISS index with sample data...")
        
        # Sample clinical documents
        sample_docs = [
            Document(
                page_content="""MIGRAINE WITH AURA DIAGNOSTIC CRITERIA:
A. At least 2 attacks fulfilling criteria B-D
B. Aura consisting of visual, sensory, speech/language, motor, brainstem, or retinal symptoms
C. At least two of: 1) aura spreads gradually ≥5 minutes, 2) multiple symptoms in succession, 
   3) each symptom lasts 5-60 minutes, 4) at least one unilateral symptom
D. Not better explained by another disorder""",
                metadata={"source": "ICHD-3 Guidelines", "page": 1}
            ),
            Document(
                page_content="""ACUTE MIGRAINE TREATMENTS:
First-line: NSAIDs (ibuprofen, naproxen), triptans (sumatriptan, rizatriptan)
Second-line: Anti-emetics (metoclopramide), ergotamines
Avoid triptans in patients with cardiovascular disease""",
                metadata={"source": "Treatment Guidelines", "page": 2}
            ),
            Document(
                page_content="""MIGRAINE PREVENTION:
Medications: Propranolol, topiramate, amitriptyline, CGRP antibodies
Lifestyle: Regular sleep, stress management, trigger avoidance
Avoid excessive caffeine and alcohol""",
                metadata={"source": "Prevention Guidelines", "page": 3}
            )
        ]
        
        # Create and save index
        vector_store = FAISS.from_documents(sample_docs, _embedding_model)
        vector_store.save_local("faiss_index")
        st.sidebar.success(f"✅ Created FAISS index with {len(sample_docs)} documents")
        
        return vector_store
        
    except Exception as e:
        st.error(f"❌ FAISS error: {e}")
        return None

# ==================== PROMPT & GENERATION ====================
def build_prompt(query: str, documents: List[Document]) -> str:
    """Build LLM prompt"""
    if not documents:
        return f"""Question: {query}

Answer based on general medical knowledge since no specific documents were found:"""
    
    docs_text = "\n\n".join([
        f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content[:500]}"
        for i, doc in enumerate(documents)
    ])
    
    prompt = f"""You are a clinical assistant. Use ONLY the following documents to answer.

DOCUMENTS:
{docs_text}

QUESTION: {query}

INSTRUCTIONS:
1. Answer based ONLY on the documents above
2. Be precise and clinical
3. Cite specific details from documents
4. If information is missing, say so

ANSWER:"""
    
    return prompt

def process_query(query: str, vector_store, llm_client):
    """Process a clinical query"""
    results = {
        "answer": "",
        "documents": [],
        "retrieval_time": 0,
        "generation_time": 0,
        "error": None
    }
    
    try:
        # Step 1: Retrieve documents
        retrieval_start = time.time()
        
        if vector_store:
            documents = vector_store.similarity_search(query, k=3)
        else:
            documents = []
            
        results["retrieval_time"] = time.time() - retrieval_start
        results["documents"] = documents
        
        # Step 2: Generate answer
        generation_start = time.time()
        
        if llm_client:
            prompt = build_prompt(query, documents)
            
            response = llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful clinical assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            results["answer"] = response.choices[0].message.content
        else:
            results["answer"] = "❌ LLM client not available. Please check API configuration."
            
        results["generation_time"] = time.time() - generation_start
        
    except Exception as e:
        results["error"] = str(e)
        results["answer"] = f"❌ Error processing query: {str(e)[:200]}"
    
    return results

# ==================== STREAMLIT UI ====================
def main():
    # Initialize session state
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "results" not in st.session_state:
        st.session_state.results = None
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Load components
        with st.spinner("Loading components..."):
            embedding_model = load_embedding_model()
            llm_client = setup_llm_client()
            vector_store = load_faiss_index(embedding_model)
        
        st.divider()
        
        # Example queries
        st.subheader("🧪 Examples")
        examples = [
            "What are migraine diagnostic criteria?",
            "List migraine treatments",
            "What is migraine prevention?"
        ]
        
        for example in examples:
            if st.button(example, use_container_width=True):
                st.session_state.query = example
                st.rerun()
    
    # Main interface
    st.title("🏥 Clinical RAG Assistant")
    st.markdown("Medical Documentation Analysis System")
    
    # API status
    if llm_client:
        st.success("✅ System ready - Enter your clinical question below")
    else:
        st.warning("⚠️ LLM not configured - Answers will be limited")
    
    st.divider()
    
    # Query input
    query = st.text_area(
        "📝 **Enter Clinical Question:**",
        value=st.session_state.query,
        height=100,
        placeholder="e.g., What are the diagnostic criteria for Migraine With Aura?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            if query:
                with st.spinner("Analyzing..."):
                    st.session_state.results = process_query(query, vector_store, llm_client)
                st.rerun()
    
    # Display results
    if st.session_state.results:
        results = st.session_state.results
        
        st.divider()
        st.subheader("🎯 Clinical Answer")
        st.markdown(results["answer"])
        
        st.divider()
        st.subheader("📄 Source Documents")
        
        if results["documents"]:
            for i, doc in enumerate(results["documents"], 1):
                with st.expander(f"Document {i}: {doc.metadata.get('source', 'Source')}"):
                    st.markdown(doc.page_content)
                    st.caption(f"Source: {doc.metadata.get('source', 'N/A')}")
        else:
            st.info("No relevant documents found")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Retrieval", f"{results['retrieval_time']:.2f}s")
        with col2:
            st.metric("⏱️ Generation", f"{results['generation_time']:.2f}s")
        with col3:
            total = results['retrieval_time'] + results['generation_time']
            st.metric("⏱️ Total", f"{total:.2f}s")
        
        # Error display
        if results["error"]:
            st.error(f"Error: {results['error']}")
    
    # Footer
    st.divider()
    st.caption("""
    ⚠️ **Disclaimer:** For educational purposes only. Not for clinical decision-making. 
    Always consult healthcare professionals for medical advice.
    """)

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
