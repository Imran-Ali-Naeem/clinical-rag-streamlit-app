import streamlit as st
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import time
import os

# Set page config
st.set_page_config(
    page_title="Clinical RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 1. LOAD API KEY FROM api.txt FILE
# ============================================================================

def load_api_key():
    """Load API key from api.txt file with fallback options"""
    api_key = None
    
    # Try 1: Read from api.txt file
    try:
        if os.path.exists("api.txt"):
            with open("api.txt", "r") as f:
                api_key = f.read().strip()
                if api_key and len(api_key) > 30:  # Basic validation
                    st.success("✅ API key loaded from api.txt")
                    return api_key
    except Exception as e:
        st.warning(f"⚠️ Could not read api.txt: {e}")
    
    # Try 2: Streamlit secrets (for cloud deployment)
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ API key loaded from Streamlit secrets")
        return api_key
    
    # Try 3: Environment variable
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            st.success("✅ API key loaded from environment")
            return api_key
    
    # Try 4: Check for .streamlit/secrets.toml (local development)
    if not api_key and os.path.exists(".streamlit/secrets.toml"):
        try:
            import toml
            secrets = toml.load(".streamlit/secrets.toml")
            api_key = secrets.get("GEMINI_API_KEY")
            if api_key:
                st.success("✅ API key loaded from secrets.toml")
                return api_key
        except:
            pass
    
    # No key found - show instructions
    st.error("""
    ❌ **API Key Not Found!**
    
    Please add your Gemini API key in one of these ways:
    
    **For Local Development:**
    1. Create `api.txt` file in the same folder
    2. Paste your Gemini API key in it (just the key, nothing else)
    
    **For Streamlit Cloud:**
    1. Go to app settings → Secrets
    2. Add: `GEMINI_API_KEY = "your-key-here"`
    
    **Temporary Testing (will show warning):**
    """)
    
    # Temporary fallback for testing
    test_key = "AIzaSyAdo-uQcG0b4YbnJZCInQetJ100Feu7OOo"  # Your key
    st.warning(f"⚠️ Using test key: {test_key[:15]}...")
    return test_key

# Load API key
API_KEY = load_api_key()

# ============================================================================
# 2. LOAD RAG COMPONENTS (Cached for performance)
# ============================================================================

@st.cache_resource
def load_rag_components():
    """Load RAG components once and cache them"""
    
    # Show loading status
    status_container = st.empty()
    status_container.info("🔍 Loading Clinical RAG System...")
    
    try:
        # Load data files
        chunks_df = pd.read_pickle("data/chunks_df.pkl")
        embeddings = np.load("data/clinical_embeddings.npy")
        index = faiss.read_index("data/faiss_index.bin")
        
        # Ensure 'text' column exists
        if 'text' not in chunks_df.columns:
            if 'chunk_text' in chunks_df.columns:
                chunks_df['text'] = chunks_df['chunk_text']
            elif 'content' in chunks_df.columns:
                chunks_df['text'] = chunks_df['content']
            else:
                # Use first text-like column
                for col in chunks_df.columns:
                    if col not in ['chunk_id', 'metadata', 'embedding']:
                        chunks_df['text'] = chunks_df[col]
                        break
        
        # Load ClinicalBERT
        embedding_model = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")
        
        # Configure Gemini
        genai.configure(api_key=API_KEY)
        
        # Try multiple model names
        model_names_to_try = [
            "gemini-2.5-flash",      # Latest
            "gemini-1.5-flash",      # Stable
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest",
            "gemini-pro",            # Legacy but works everywhere
        ]
        
        model = None
        MODEL_NAME = None
        
        for model_name in model_names_to_try:
            try:
                test_model = genai.GenerativeModel(model_name)
                test_response = test_model.generate_content("Test", request_options={"timeout": 5})
                if test_response.text:
                    model = test_model
                    MODEL_NAME = model_name
                    status_container.success(f"✅ Gemini {MODEL_NAME} connected!")
                    break
            except:
                continue
        
        if not model:
            status_container.warning("⚠️ Gemini not available - using retrieval only")
        
        components = {
            "chunks_df": chunks_df,
            "embeddings": embeddings,
            "index": index,
            "embedding_model": embedding_model,
            "genai_model": model,
            "model_name": MODEL_NAME
        }
        
        # Clear loading message after success
        time.sleep(0.5)
        status_container.empty()
        
        return components
        
    except Exception as e:
        status_container.error(f"❌ Error loading RAG system: {str(e)[:200]}")
        # Return minimal components to prevent crash
        return {
            "chunks_df": pd.DataFrame({'text': ['Error loading data']}),
            "embeddings": np.random.randn(1, 768),
            "index": None,
            "embedding_model": None,
            "genai_model": None,
            "model_name": None
        }

# Load components
components = load_rag_components()
chunks_df = components["chunks_df"]
index = components["index"]
embedding_model = components["embedding_model"]
genai_model = components["genai_model"]
model_name = components["model_name"]

# ============================================================================
# 3. RAG FUNCTIONS
# ============================================================================

def retrieve_documents(query, top_k=3):
    """Retrieve relevant clinical documents"""
    try:
        if embedding_model is None or index is None:
            return []
        
        # Encode and normalize query
        query_embedding = embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS
        distances, indices = index.search(query_embedding, k=top_k)
        
        results = []
        for idx, sim in zip(indices[0], distances[0]):
            if idx < len(chunks_df):
                text = str(chunks_df.iloc[idx]['text'])
                if len(text) > 500:
                    text = text[:500] + "..."
                
                results.append({
                    'text': text,
                    'similarity': float(sim),
                    'index': idx
                })
        
        return results
    except Exception as e:
        st.error(f"Retrieval error: {e}")
        return []

def generate_response(query, documents):
    """Generate answer using Gemini"""
    if not documents:
        return "No relevant clinical documents found."
    
    if not genai_model:
        return "⚠️ Gemini API not available. Showing retrieved documents only.\n\nPlease check your API key in api.txt file."
    
    # Build context
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(f"[Document {i}, Relevance: {doc['similarity']:.3f}]")
        context_parts.append(doc['text'])
        context_parts.append("")
    
    context = "\n".join(context_parts)
    
    # Create professional medical prompt
    prompt = f"""You are a medical AI assistant analyzing clinical documentation.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using the provided clinical notes
2. If information is insufficient, say: "Based on the provided clinical notes, I cannot determine [specific information]"
3. NEVER generate patient identifiers (names, IDs, exact dates)
4. Never provide medical advice or diagnosis
5. Always cite which document supports your answer (e.g., Document 1)
6. Use clear, professional medical terminology

FORMATTING REQUIREMENTS:
- Use bullet points for lists
- Use **bold** for headings/categories
- Structure answer logically
- Include specific citations

CLINICAL DOCUMENTS:
{context}

QUESTION: {query}

STRUCTURED ANSWER BASED ONLY ON NOTES:"""
    
    try:
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
        
        response = genai_model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        return response.text if response.text else "No response generated."
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            return "⚠️ Rate limit exceeded. Please wait a moment and try again."
        elif "quota" in error_msg.lower():
            return "⚠️ API quota exceeded. Please check your Gemini API billing."
        else:
            return f"Error: {error_msg[:100]}"

# ============================================================================
# 4. STREAMLIT UI
# ============================================================================

# Sidebar
with st.sidebar:
    st.title("🏥 Clinical RAG")
    st.markdown("---")
    
    # API Key Status
    st.subheader("🔑 API Status")
    if genai_model:
        st.success(f"✅ Gemini Connected\n*Model: {model_name}*")
    else:
        st.error("❌ Gemini Disabled")
        st.info("Add API key to `api.txt` file")
    
    # Settings
    st.markdown("---")
    st.subheader("⚙️ Settings")
    top_k = st.slider("Documents to retrieve", 1, 5, 3)
    
    # System info
    st.markdown("---")
    st.subheader("📊 System Info")
    st.write(f"**Documents:** {len(chunks_df)}")
    if index:
        st.write(f"**FAISS Index:** {index.ntotal} vectors")
    
    # Quick queries
    st.markdown("---")
    st.subheader("💡 Quick Queries")
    
    quick_queries = {
        "Gastric vs Duodenal Ulcers": "What are the differences between gastric ulcers and duodenal ulcers?",
        "Medications List": "What medications are mentioned in the clinical notes?",
        "PUD Diagnosis": "How is Peptic Ulcer Disease diagnosed according to clinical guidelines?",
        "Migraine Criteria": "What are the diagnostic criteria for Migraine With Aura?",
        "Patient Symptoms": "What symptoms are documented in the patient notes?",
    }
    
    for name, query_text in quick_queries.items():
        if st.button(name, use_container_width=True, key=f"btn_{name}"):
            st.session_state.query = query_text
            st.rerun()

# Main content
st.title("🏥 Clinical RAG Assistant")
st.markdown("Medical Documentation Analysis using Retrieval-Augmented Generation")

# API key instructions (collapsible)
with st.expander("🔧 API Key Setup Instructions", expanded=False):
    st.markdown("""
    ### How to set up your Gemini API key:
    
    **Option 1: Local Development (api.txt file)**
    1. Create a file named `api.txt` in the same folder as `app.py`
    2. Paste ONLY your Gemini API key in it
    3. No quotes, no spaces, just the key
    
    **Example api.txt content:**
    ```
    AIzaSyAdo-uQcG0b4YbnJZCInQetJ100Feu7OOo
    ```
    
    **Option 2: Streamlit Cloud Deployment**
    1. Go to your app → Settings → Secrets
    2. Add: `GEMINI_API_KEY = "your-key-here"`
    3. Deploy again
    
    **Get API key from:** [Google AI Studio](https://makersuite.google.com/app/apikey)
    """)

# Initialize session state
if "query" not in st.session_state:
    st.session_state.query = ""

# Query input
query = st.text_area(
    "📝 **Enter your clinical question:**",
    value=st.session_state.query,
    height=100,
    placeholder="Examples:\n• What are the differences between gastric and duodenal ulcers?\n• List the medications mentioned in the notes\n• How is Peptic Ulcer Disease diagnosed?\n• What symptoms does the patient report?"
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.query = ""
        st.rerun()

# Process query
if analyze_btn and query:
    with st.spinner("🔍 Retrieving relevant documents..."):
        start_time = time.time()
        
        # Retrieve documents
        documents = retrieve_documents(query, top_k=top_k)
        retrieval_time = time.time() - start_time
        
        # Generate response
        answer = generate_response(query, documents)
        total_time = time.time() - start_time
        
        # Display results in tabs
        tab1, tab2 = st.tabs(["🎯 Clinical Answer", "📄 Source Documents"])
        
        with tab1:
            st.markdown("### Clinical Answer")
            st.markdown(answer)
            
            # Metrics
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Retrieval Time", f"{retrieval_time:.2f}s")
            with col2:
                st.metric("⏱️ Total Time", f"{total_time:.2f}s")
            with col3:
                st.metric("📄 Documents", len(documents))
        
        with tab2:
            st.markdown("### Retrieved Clinical Documents")
            if documents:
                for i, doc in enumerate(documents, 1):
                    with st.expander(f"Document {i} (Relevance: {doc['similarity']:.3f})", expanded=(i==1)):
                        st.markdown(doc['text'])
                        st.caption(f"Similarity score: {doc['similarity']:.3f}")
            else:
                st.info("No documents retrieved with sufficient relevance.")

# Example queries
st.markdown("---")
st.subheader("🧪 Example Clinical Queries")

examples = st.columns(3)
example_queries = [
    ("Migraine diagnostic criteria", "What are the diagnostic criteria for Migraine With Aura?"),
    ("GERD symptoms", "List the common symptoms of Gastro-esophageal Reflux Disease (GERD)."),
    ("Patient symptoms", "What symptoms are documented in the patient notes?"),
    ("Ulcer differences", "What is the difference between gastric ulcers and duodenal ulcers?"),
    ("Medication list", "What medications are commonly mentioned in the notes?"),
    ("Diagnosis methods", "How is Peptic Ulcer Disease diagnosed according to guidelines?"),
]

for i, (label, qtext) in enumerate(example_queries):
    col_idx = i % 3
    with examples[col_idx]:
        if st.button(label, use_container_width=True, key=f"ex_{i}"):
            st.session_state.query = qtext
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-top: 20px;'>
<h4>⚠️ MEDICAL DISCLAIMER</h4>
<ul style='color: #555;'>
<li><strong style='color: #d32f2f;'>For educational and research purposes only</strong></li>
<li><strong style='color: #d32f2f;'>Not for clinical diagnosis, treatment decisions, or patient care</strong></li>
<li>Always consult qualified healthcare professionals</li>
<li>Patient identifiers have been removed for privacy</li>
<li>System has limitations and may not have all relevant information</li>
<li>Responses are based only on provided clinical documentation</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Debug info (collapsible - hidden by default)
with st.expander("🔍 Debug Information", expanded=False):
    st.json({
        "api_key_loaded": bool(API_KEY) and len(API_KEY) > 30,
        "api_key_source": "api.txt" if os.path.exists("api.txt") else ("secrets" if "GEMINI_API_KEY" in st.secrets else "environment/fallback"),
        "documents_loaded": len(chunks_df),
        "faiss_index_size": index.ntotal if index else "Not loaded",
        "gemini_connected": bool(genai_model),
        "model_name": model_name,
        "embedding_model_loaded": embedding_model is not None
    })
    
    # Show first few chunks for verification
    if len(chunks_df) > 0:
        st.markdown("### Sample Document Chunks")
        for i in range(min(3, len(chunks_df))):
            st.text(f"Chunk {i}: {str(chunks_df.iloc[i]['text'])[:100]}...")
