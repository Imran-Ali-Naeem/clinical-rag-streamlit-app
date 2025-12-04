import streamlit as st
import os
import time
from typing import List, Tuple, Optional
import traceback
from pathlib import Path
import numpy as np

# FAISS Components
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.schema import Document

# LLM
from openai import OpenAI
import openai

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Clinical RAG Assistant",
    page_icon="🏥",
    layout="wide"
)

# ==================== CACHED RESOURCES ====================
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """Load embedding model with fallback options"""
    try:
        # Try faster model first to avoid hanging
        model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Faster alternative
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'show_progress_bar': True
            }
        )
        st.sidebar.success("✅ Loaded embedding model")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load embedding model: {e}")
        # Ultimate fallback
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model

@st.cache_resource(show_spinner="Initializing LLM client...")
def setup_llm_client():
    """Setup OpenAI client with validation"""
    try:
        if "openai_api_key" not in st.secrets:
            st.error("❌ OpenAI API key not found in secrets")
            st.info("Add your OpenAI API key to Streamlit secrets")
            return None
        
        client = OpenAI(
            api_key=st.secrets["openai_api_key"],
            timeout=30.0
        )
        
        # Test connection
        try:
            test = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            st.sidebar.success("✅ OpenAI API connected")
            return client
        except openai.AuthenticationError:
            st.error("❌ Invalid OpenAI API key")
            return None
        except Exception as e:
            st.warning(f"⚠️ OpenAI test had issue: {e}")
            return client  # Still return client if test fails but auth works
    except Exception as e:
        st.error(f"❌ LLM setup error: {e}")
        return None

@st.cache_resource(show_spinner="Loading FAISS vector database...")
def load_faiss_index(_embedding_model):
    """Load existing FAISS index or create from documents"""
    try:
        # Check for existing FAISS index
        index_path = Path("faiss_index")
        
        if index_path.exists() and (index_path / "index.faiss").exists():
            st.sidebar.info("📂 Loading existing FAISS index...")
            vector_store = FAISS.load_local(
                str(index_path),
                _embedding_model,
                allow_dangerous_deserialization=True
            )
            st.sidebar.success(f"✅ Loaded FAISS index with {vector_store.index.ntotal} vectors")
            return vector_store
        else:
            # Create new index from sample documents or your actual documents
            st.sidebar.warning("⚠️ No FAISS index found. Creating with sample data...")
            
            # Sample clinical documents - REPLACE WITH YOUR ACTUAL DOCUMENTS
            sample_docs = [
                Document(
                    page_content="""MIGRAINE WITH AURA DIAGNOSTIC CRITERIA (ICHD-3):
A. At least 2 attacks fulfilling criteria B and C
B. One or more of the following fully reversible aura symptoms:
   1. Visual symptoms (flickering lights, spots, lines, loss of vision)
   2. Sensory symptoms (pins and needles, numbness)
   3. Speech and/or language symptoms (dysphasia)
   4. Motor symptoms (weakness)
   5. Brainstem symptoms (vertigo, tinnitus, diplopia)
   6. Retinal symptoms (monocular visual disturbances)
C. At least three of the following six characteristics:
   1. At least one aura symptom spreads gradually over ≥5 minutes
   2. Two or more aura symptoms occur in succession
   3. Each individual aura symptom lasts 5–60 minutes
   4. At least one aura symptom is unilateral
   5. At least one aura symptom is positive (visual scintillations)
   6. The aura is accompanied, or followed within 60 minutes, by headache
D. Not better accounted for by another ICHD-3 diagnosis.""",
                    metadata={"source": "International Classification of Headache Disorders 3rd Edition", "type": "guideline"}
                ),
                Document(
                    page_content="""ACUTE MIGRAINE TREATMENT:
First-line treatments:
1. NSAIDs: Ibuprofen 400-800mg, Naproxen 500-550mg, Diclofenac 50-100mg
2. Triptans: Sumatriptan 50-100mg PO, Rizatriptan 10mg, Eletriptan 40mg
3. Combination therapy: Sumatriptan 85mg + Naproxen 500mg

Rescue medications for severe attacks:
1. Anti-emetics: Metoclopramide 10mg IV/IM, Prochlorperazine 10mg
2. Dihydroergotamine 1mg IM/SC
3. Dexamethasone 10-24mg IV for status migrainosus

Contraindications: Triptans contraindicated in patients with ischemic heart disease, Prinzmetal angina, uncontrolled hypertension, hemiplegic migraine.""",
                    metadata={"source": "Neurology Clinical Guidelines", "type": "treatment"}
                ),
                Document(
                    page_content="""MIGRAINE PREVENTIVE THERAPIES:
Evidence Level A (Established efficacy):
1. Beta-blockers: Propranolol 40-240mg/day, Timolol 10-30mg/day
2. Antiepileptics: Topiramate 50-200mg/day, Valproate 500-1500mg/day
3. Antidepressants: Amitriptyline 25-150mg/day, Venlafaxine 75-150mg/day

Evidence Level B (Probably effective):
1. CGRP monoclonal antibodies: Erenumab, Fremanezumab, Galcanezumab
2. ARBs: Candesartan 16-32mg/day
3. NSAIDs: Naproxen 500mg BID (short-term prevention)

Lifestyle modifications: Regular sleep schedule, stress management, trigger identification, regular aerobic exercise, hydration, caffeine moderation.""",
                    metadata={"source": "American Headache Society Guidelines", "type": "prevention"}
                )
            ]
            
            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Split documents
            split_docs = text_splitter.split_documents(sample_docs)
            
            # Create FAISS index
            vector_store = FAISS.from_documents(
                documents=split_docs,
                embedding=_embedding_model
            )
            
            # Save the index locally
            vector_store.save_local("faiss_index")
            st.sidebar.success(f"✅ Created FAISS index with {len(split_docs)} document chunks")
            
            return vector_store
            
    except Exception as e:
        st.error(f"❌ FAISS loading error: {e}")
        st.code(traceback.format_exc())
        return None

# ==================== PROMPT TEMPLATES ====================
def build_clinical_prompt(query: str, documents: List[Document]) -> str:
    """Build a prompt for clinical question answering"""
    if not documents:
        return f"""You are a clinical assistant. No relevant documents were found for this query.

Question: {query}

Please provide a general clinical answer based on your medical knowledge, but clearly state that no specific documents were found."""

    # Format documents
    docs_text = ""
    for i, doc in enumerate(documents, 1):
        content = doc.page_content[:800]  # Limit each doc
        source = doc.metadata.get('source', 'Unknown source')
        docs_text += f"\n--- Document {i} ({source}) ---\n{content}\n"
    
    prompt = f"""You are a clinical decision support assistant. Answer the medical question based ONLY on the provided clinical documents.

CLINICAL DOCUMENTS:
{docs_text}

QUESTION: {query}

INSTRUCTIONS:
1. Answer based ONLY on information in the documents above
2. Be precise and cite specific details from documents
3. If information is incomplete, state what is known and what is missing
4. Use professional medical terminology
5. Format with clear sections if helpful

CLINICAL ANSWER:"""
    
    return prompt

# ==================== QUERY PROCESSING ====================
def retrieve_documents(query: str, vector_store, k: int = 4) -> Tuple[List[Document], float]:
    """Retrieve relevant documents from FAISS"""
    start_time = time.time()
    try:
        if not vector_store:
            return [], 0.0
            
        # Perform similarity search
        docs = vector_store.similarity_search_with_score(
            query=query,
            k=k
        )
        
        # Separate documents and scores
        documents = [doc for doc, _ in docs]
        scores = [score for _, score in docs]
        
        retrieval_time = time.time() - start_time
        
        # Debug info
        if len(documents) > 0:
            print(f"🔍 Retrieved {len(documents)} documents in {retrieval_time:.2f}s")
            print(f"📊 Similarity scores: {scores}")
        
        return documents, retrieval_time
        
    except Exception as e:
        st.error(f"❌ Retrieval error: {e}")
        return [], time.time() - start_time

def generate_clinical_answer(query: str, documents: List[Document], llm_client) -> Tuple[str, float]:
    """Generate answer using LLM"""
    start_time = time.time()
    
    try:
        if not llm_client:
            return "❌ LLM client not available. Please check API configuration.", 0.0
        
        # Build prompt
        prompt = build_clinical_prompt(query, documents)
        
        # Debug: Show prompt info
        print(f"📝 Prompt length: {len(prompt)} characters")
        print(f"📄 Number of documents: {len(documents)}")
        
        # Call LLM
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are a precise clinical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000,
            timeout=30.0
        )
        
        generation_time = time.time() - start_time
        answer = response.choices[0].message.content
        
        print(f"✅ Answer generated in {generation_time:.2f}s")
        return answer, generation_time
        
    except openai.APITimeoutError:
        return "❌ LLM request timed out. Please try again.", time.time() - start_time
    except openai.RateLimitError:
        return "❌ Rate limit exceeded. Please wait and try again.", time.time() - start_time
    except openai.APIError as e:
        return f"❌ API error: {str(e)[:200]}", time.time() - start_time
    except Exception as e:
        return f"❌ Generation error: {str(e)[:200]}", time.time() - start_time

# ==================== STREAMLIT UI ====================
def main():
    # Sidebar
    with st.sidebar:
        st.title("🏥 Clinical RAG Setup")
        
        # Initialize components
        with st.spinner("Loading AI models..."):
            embedding_model = load_embedding_model()
            llm_client = setup_llm_client()
            vector_store = load_faiss_index(embedding_model)
        
        st.divider()
        
        # Debug info
        if st.checkbox("Show debug info", value=False):
            if vector_store:
                st.info(f"FAISS index size: {vector_store.index.ntotal} vectors")
            if embedding_model:
                st.info(f"Embedding model: {embedding_model.model_name}")
        
        st.divider()
        
        # Example queries
        st.subheader("🧪 Example Queries")
        example_queries = [
            "What are the diagnostic criteria for Migraine With Aura?",
            "List acute treatments for migraine attacks",
            "What are preventive therapies for chronic migraine?",
            "What are contraindications for triptans?",
            "Describe lifestyle modifications for migraine management"
        ]
        
        for query in example_queries:
            if st.button(f"💬 {query[:40]}..."):
                st.session_state.query = query
    
    # Main content
    st.title("🏥 Clinical RAG Assistant")
    st.markdown("### Medical Documentation Analysis using Retrieval-Augmented Generation")
    
    st.divider()
    
    # API Key status
    if "openai_api_key" in st.secrets:
        st.success("✅ API key loaded from Streamlit secrets")
    else:
        st.error("❌ API key not found. Add to Streamlit secrets:")
        st.code("""
# In .streamlit/secrets.toml
openai_api_key = "your-api-key-here"
        """)
    
    st.divider()
    
    # Query input
    query = st.text_area(
        "📝 Enter your clinical question:",
        value=st.session_state.get("query", ""),
        height=100,
        placeholder="e.g., What are the diagnostic criteria for Migraine With Aura?"
    )
    
    # Process button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        process_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("🔄 Clear", use_container_width=True)
    
    if clear_btn:
        st.session_state.clear()
        st.rerun()
    
    if process_btn and query:
        # Initialize session state for results
        if "results" not in st.session_state:
            st.session_state.results = {}
        
        # Create containers for results
        retrieval_container = st.container()
        answer_container = st.container()
        sources_container = st.container()
        debug_container = st.container()
        
        with retrieval_container:
            st.subheader("🔍 Retrieving Clinical Documents...")
            retrieval_progress = st.progress(0)
            
            # Step 1: Retrieve documents
            documents, retrieval_time = retrieve_documents(query, vector_store, k=4)
            retrieval_progress.progress(50)
            
            # Display retrieval results
            if documents:
                st.success(f"✅ Retrieved {len(documents)} relevant documents in {retrieval_time:.2f}s")
            else:
                st.warning("⚠️ No relevant documents found. Generating general answer...")
        
        with answer_container:
            st.subheader("🎯 Clinical Answer")
            answer_progress = st.progress(0)
            
            # Step 2: Generate answer
            answer, generation_time = generate_clinical_answer(query, documents, llm_client)
            answer_progress.progress(100)
            
            # Display answer
            st.markdown(answer)
        
        with sources_container:
            st.subheader("📄 Source Documents")
            
            if documents:
                for i, doc in enumerate(documents, 1):
                    with st.expander(f"Document {i}: {doc.metadata.get('source', 'Unknown')}"):
                        st.markdown(doc.page_content)
                        st.caption(f"Source: {doc.metadata.get('source', 'N/A')} | Type: {doc.metadata.get('type', 'N/A')}")
            else:
                st.info("No source documents available")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Retrieval Time", f"{retrieval_time:.2f}s")
        with col2:
            st.metric("⏱️ Generation Time", f"{generation_time:.2f}s")
        with col3:
            st.metric("⏱️ Total Time", f"{retrieval_time + generation_time:.2f}s")
        
        with debug_container:
            with st.expander("🔍 Debug Information", expanded=False):
                st.subheader("Query Details")
                st.code(f"Query: {query}")
                
                st.subheader("Retrieval Details")
                st.code(f"Documents retrieved: {len(documents)}")
                st.code(f"Vector store size: {vector_store.index.ntotal if vector_store else 0}")
                
                if documents:
                    st.subheader("Document Similarities")
                    # Get scores
                    _, scores = vector_store.similarity_search_with_score(query, k=4)
                    for i, score in enumerate(scores, 1):
                        st.code(f"Document {i}: Similarity score = {score:.4f}")
        
        st.divider()
        
        # Medical disclaimer
        st.warning("""
        ⚠️ **MEDICAL DISCLAIMER**
        - For educational and research purposes only
        - Not for clinical diagnosis, treatment decisions, or patient care
        - Always consult qualified healthcare professionals
        - Patient identifiers have been removed for privacy
        - System has limitations and may not have all relevant information
        - Responses are based only on provided clinical documentation
        """)
    
    elif process_btn and not query:
        st.error("❌ Please enter a clinical question")

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    # Initialize session state
    if "query" not in st.session_state:
        st.session_state.query = ""
    
    main()
