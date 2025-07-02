import streamlit as st
import os
import torch
import requests
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_huggingface.llms import HuggingFacePipeline
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time
import tempfile
import urllib.parse
import zipfile

st.set_page_config(
    page_title="PDF RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .main-header{
    text-align: center;
    padding: 1rem 0;
    margin-bottom: 2rem;
  }
  .chat-container{
    max-width: 800px;
    margin: 0 auto;
    padding: 1rem;
    max-height: 500px;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    margin-bottom: 20px;
  }
  .user-message{
    background-color: #e3f2fd;
    border-radius: 18px;
    padding: 12px 16px;
    margin: 8px 0;
    margin-left: 20%;
    text-align: left;
    border: 1px solid #2196f3;
  }
  .assistant-message{
    background-color: #f1f8e9;
    border-radius: 18px;
    padding: 12px 16px;
    margin: 8px 0;
    margin-right: 20%;
    text-align: left;
    border: 1px solid #4caf50;
  }
  .chat-input-container {
    position: sticky;
    bottom: 0;
    background-color: white;
    padding: 1rem;
    border-top: 2px solid #e0e0e0;
    border-radius: 10px;
    margin-top: 20px;
  }
  .stTextInput > div > div > input {
    border-radius: 25px;
    border: 2px solid #e0e0e0;
    padding: 12px 20px;
    font-size: 16px;
  }
  .document-info {
    background-color: #f8f9fa;
    border-left: 4px solid #28a745;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 4px;
  }
  .status-indicator{
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
  }
  .status-ready{
    background-color: #28a745;
  }
  .status-loading {
    background-color: #ffc107;
  }
  .status-error {
    background-color: #dc3545;
  }
  .thinking-indicator {
    background-color: #f5f5f5;
    border-radius: 18px;
    padding: 12px 16px;
    margin: 8px 0;
    margin-right: 20%;
    text-align: left;
    border: 1px solid #ddd;
    animation: pulse 1.5s ease-in-out infinite;
  }
  @keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
  }
  .upload-section {
    background-color: #f8f9fa;
    border: 2px dashed #28a745;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    text-align: center;
  }
  .file-counter {
    background-color: #e3f2fd;
    border-radius: 5px;
    padding: 5px 10px;
    margin: 5px;
    display: inline-block;
    font-size: 12px;
  }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'pdf_source' not in st.session_state:
    st.session_state.pdf_source = "github"  # Default to GitHub
if 'github_repo_url' not in st.session_state:
    st.session_state.github_repo_url = "https://github.com/Jennifer1907/Time-Series-Team-Hub/tree/main/assets/pdf"
if 'local_folder_path' not in st.session_state:
    st.session_state.local_folder_path = "./knowledge_base"
if 'processing_query' not in st.session_state:
    st.session_state.processing_query = False
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    """Load a lightweight model suitable for deployment"""
    try:
        st.info("🔄 Loading CPU-optimized model for deployment...")
        
        # Use a lightweight model that works well on CPU
        MODEL_NAME = "microsoft/DialoGPT-small"  # Very lightweight for demo
        
        # Load without any quantization
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Create pipeline for CPU
        model_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128,  # Reduced for better performance
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            device=-1  # Force CPU usage
        )
        
        return HuggingFacePipeline(pipeline=model_pipeline)
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please try using a local deployment or check your environment setup.")
        return None

def extract_text_from_uploaded_file(file):
    """Extract text from uploaded file based on file type"""
    file_extension = file.name.split('.')[-1].lower()
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(file.getbuffer())
            tmp_path = tmp_file.name
        
        documents = []
        
        if file_extension == 'pdf':
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
        elif file_extension == 'docx':
            loader = Docx2txtLoader(tmp_path)
            documents = loader.load()
        elif file_extension in ['xlsx', 'xls']:
            loader = UnstructuredExcelLoader(tmp_path)
            documents = loader.load()
        else:
            st.warning(f"Unsupported file type: {file_extension}")
            return []
        
        # Clean up temporary file
        os.unlink(tmp_path)
        return documents
        
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return []

def process_zip_file(zip_file):
    """Process uploaded zip file containing documents"""
    try:
        all_documents = []
        loaded_files = []
        
        # Create temporary directory for extraction
        temp_dir = tempfile.mkdtemp()
        
        # Save uploaded zip file
        zip_path = os.path.join(temp_dir, zip_file.name)
        with open(zip_path, 'wb') as f:
            f.write(zip_file.getbuffer())
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find all document files in extracted folder
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(('.pdf', '.docx', '.xlsx', '.xls')):
                    file_path = os.path.join(root, file)
                    try:
                        if file.lower().endswith('.pdf'):
                            loader = PyPDFLoader(file_path)
                        elif file.lower().endswith('.docx'):
                            loader = Docx2txtLoader(file_path)
                        elif file.lower().endswith(('.xlsx', '.xls')):
                            loader = UnstructuredExcelLoader(file_path)
                        
                        documents = loader.load()
                        all_documents.extend(documents)
                        loaded_files.append(file)
                        st.success(f"✅ Processed from zip: {file}")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {file} from zip: {str(e)}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        return all_documents, loaded_files
        
    except Exception as e:
        st.error(f"Error processing zip file: {str(e)}")
        return [], []

def get_github_pdf_files(repo_url):
    """Link github hiện tại là giao diện dành cho người dùng (HTML), không phải API dành cho máy móc.
    Để đọc trực tiếp danh sách file từ URL này, phải chuyển link github hiện tại sang API"""
    try:
        if "github.com" in repo_url and "/tree/" in repo_url:
            parts = repo_url.replace("https://github.com/", "").split("/tree/")
            repo_path = parts[0]
            branch_and_path = parts[1].split("/", 1)
            branch = branch_and_path[0]
            folder_path = branch_and_path[1] if len(branch_and_path) > 1 else ""

            api_url = f"https://api.github.com/repos/{repo_path}/contents/{folder_path}?ref={branch}"
        else:
            st.error("Invalid GitHub URL format")
            return []

        response = requests.get(api_url)
        if response.status_code == 200:
            files = response.json()
            pdf_files = []
            for file in files:
                if file['name'].endswith('.pdf') and file['type'] == 'file':
                    pdf_files.append({
                        'name': file['name'],
                        'download_url': file['download_url']
                    })
            return pdf_files
        else:
            st.error(f"Failed to access GitHub repository: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error accessing GitHub repository: {str(e)}")
        return []

def download_pdf_from_url(url, filename, temp_dir):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return file_path
        return None
    except Exception as e:
        st.error(f"Error downloading {filename}: {str(e)}")
        return None

def create_rag_chain(all_documents):
    """Create RAG chain from documents"""
    if not all_documents:
        return None, 0
    
    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )

    docs = semantic_splitter.split_documents(all_documents)
    
    # FAISS implementation
    vector_db = FAISS.from_documents(documents=docs, embedding=st.session_state.embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )

    return rag_chain, len(docs)

def load_pdfs_from_github(repo_url):
    pdf_files = get_github_pdf_files(repo_url)

    if not pdf_files:
        st.warning("No PDF files found in the GitHub repository")
        return None, 0, []

    temp_dir = tempfile.mkdtemp()
    all_documents = []
    loaded_files = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, pdf_file in enumerate(pdf_files):
        try:
            status_text.text(f"Downloading and processing: {pdf_file['name']}")
            local_path = download_pdf_from_url(pdf_file['download_url'], pdf_file['name'], temp_dir)

            if local_path:
                loader = PyPDFLoader(local_path)
                documents = loader.load()
                all_documents.extend(documents)
                loaded_files.append(pdf_file['name'])

                st.success(f"✅ Processed: {pdf_file['name']} ({len(documents)} pages)")
            progress_bar.progress((i + 1) / len(pdf_files))
        except Exception as e:
            st.error(f"❌ Error processing {pdf_file['name']}: {str(e)}")

    progress_bar.empty()
    status_text.empty()

    # Clean up temporary directory
    shutil.rmtree(temp_dir)

    if not all_documents:
        return None, 0, loaded_files

    rag_chain, num_chunks = create_rag_chain(all_documents)
    return rag_chain, num_chunks, loaded_files

def load_pdfs_from_folder(folder_path):
    """Load all PDF files from the specified local folder"""
    cleaned_path = folder_path.strip().strip('"').strip("'")
    folder = Path(cleaned_path)

    if not folder.exists():
        st.error(f"❌ Folder does not exist: `{cleaned_path}`")
        return None, 0, []

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        st.warning(f"No PDF files found in folder: {cleaned_path}")
        return None, 0, []

    all_documents = []
    loaded_files = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, pdf_file in enumerate(pdf_files):
        try:
            status_text.text(f"Processing: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            all_documents.extend(documents)
            loaded_files.append(pdf_file.name)
            progress_bar.progress((i + 1) / len(pdf_files))
            st.success(f"✅ Processed: {pdf_file.name} ({len(documents)} pages)")

        except Exception as e:
            st.error(f"❌ Error processing {pdf_file.name}: {str(e)}")

    progress_bar.empty()
    status_text.empty()

    if not all_documents:
        return None, 0, loaded_files

    rag_chain, num_chunks = create_rag_chain(all_documents)
    return rag_chain, num_chunks, loaded_files

def display_chat_message(message, is_user=True):
    if is_user:
        st.markdown(f"""
        <div class="user-message">
            <strong>You:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>AI Assistant:</strong> {message}
        </div>
        """, unsafe_allow_html=True)

def display_thinking_indicator():
    st.markdown(f"""
    <div class="thinking-indicator">
        <strong>AI Assistant:</strong> 🤔 Thinking...
    </div>
    """, unsafe_allow_html=True)

def process_user_query(question):
    try:
        output = st.session_state.rag_chain.invoke(question)
        answer = output.split('Answer:')[1].strip() if 'Answer:' in output else output.strip()
        return answer
    except Exception as e:
        return f"Sorry, an error occurred while processing your question: {str(e)}"

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 PDF RAG Assistant with File Upload</h1>
        <p>Smart AI Assistant - Upload files, folders (zip), or use GitHub repository for Q&A</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuration")

        if st.session_state.models_loaded:
            st.markdown('<span class="status-indicator status-ready"></span>**Models:** Ready', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-loading"></span>**Models:** Loading...', unsafe_allow_html=True)

        # Document loading status
        if st.session_state.documents_loaded:
            st.markdown('<span class="status-indicator status-ready"></span>**Documents:** Loaded (FAISS)', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-error"></span>**Documents:** Not loaded', unsafe_allow_html=True)

        st.divider()

        # Document source selection
        st.subheader("📁 Document Source")

        pdf_source = st.radio(
            "Choose document source:",
            ["Upload Files", "Upload Folder (ZIP)", "GitHub Repository", "Local Folder Path"],
            key="pdf_source_radio"
        )

        if pdf_source == "Upload Files":
            st.session_state.pdf_source = "upload_files"
            
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.markdown("**📎 Upload Individual Files**")
            uploaded_files = st.file_uploader(
                "Choose files to upload:",
                type=['pdf', 'docx', 'xlsx', 'xls'],
                accept_multiple_files=True,
                help="Supported formats: PDF, Word (.docx), Excel (.xlsx, .xls)"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_files:
                st.markdown("**Selected Files:**")
                for i, file in enumerate(uploaded_files):
                    file_size = len(file.getbuffer()) / (1024 * 1024)  # Size in MB
                    st.markdown(f'<span class="file-counter">{i+1}. {file.name} ({file_size:.1f} MB)</span>', unsafe_allow_html=True)
                
                if st.button("📤 Process Uploaded Files", type="primary"):
                    with st.spinner("Processing uploaded files..."):
                        all_documents = []
                        loaded_files = []
                        
                        progress_bar = st.progress(0)
                        
                        for i, file in enumerate(uploaded_files):
                            documents = extract_text_from_uploaded_file(file)
                            if documents:
                                all_documents.extend(documents)
                                loaded_files.append(file.name)
                                st.success(f"✅ Processed: {file.name}")
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        progress_bar.empty()
                        
                        if all_documents:
                            rag_chain, num_chunks = create_rag_chain(all_documents)
                            if rag_chain:
                                st.session_state.rag_chain = rag_chain
                                st.session_state.documents_loaded = True
                                st.success(f"✅ Successfully processed {len(loaded_files)} files!")
                                st.rerun()
                        else:
                            st.error("No documents could be processed.")

        elif pdf_source == "Upload Folder (ZIP)":
            st.session_state.pdf_source = "upload_zip"
            
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.markdown("**📁 Upload Folder as ZIP**")
            zip_file = st.file_uploader(
                "Choose a ZIP file containing documents:",
                type=['zip'],
                help="Upload a ZIP file containing PDF, Word, or Excel files"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if zip_file:
                file_size = len(zip_file.getbuffer()) / (1024 * 1024)  # Size in MB
                st.info(f"📦 Selected ZIP: {zip_file.name} ({file_size:.1f} MB)")
                
                if st.button("📤 Process ZIP File", type="primary"):
                    with st.spinner("Extracting and processing ZIP file..."):
                        all_documents, loaded_files = process_zip_file(zip_file)
                        
                        if all_documents:
                            rag_chain, num_chunks = create_rag_chain(all_documents)
                            if rag_chain:
                                st.session_state.rag_chain = rag_chain
                                st.session_state.documents_loaded = True
                                st.success(f"✅ Successfully processed {len(loaded_files)} files from ZIP!")
                                st.rerun()
                        else:
                            st.error("No valid documents found in ZIP file.")

        elif pdf_source == "GitHub Repository":
            st.session_state.pdf_source = "github"
            github_url = st.text_input(
                "GitHub Repository URL:",
                value=st.session_state.github_repo_url,
                help="URL to GitHub folder containing PDF files"
            )
            st.session_state.github_repo_url = github_url
            
            if st.button("📥 Load from GitHub", type="primary"):
                st.session_state.documents_loaded = False
                st.rerun()

        else:  # Local Folder Path
            st.session_state.pdf_source = "local"
            local_path = st.text_input(
                "Local Folder Path:",
                value=st.session_state.local_folder_path,
                help="Path to local folder containing PDF files"
            )
            st.session_state.local_folder_path = local_path
            
            if st.button("📂 Load from Local Folder", type="primary"):
                st.session_state.documents_loaded = False
                st.rerun()

        st.divider()

        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.processing_query = False
            st.rerun()

        if st.button("🗑️ Clear All Documents"):
            st.session_state.documents_loaded = False
            st.session_state.rag_chain = None
            st.session_state.chat_history = []
            st.session_state.processing_query = False
            st.rerun()

        # FAISS specific settings
        st.divider()
        st.subheader("🔍 FAISS Settings")
        st.info("FAISS is a fast similarity search library. It's more memory efficient than Chroma.")

    if not st.session_state.models_loaded:
        with st.spinner("🚀 Initializing AI models..."):
            st.session_state.embeddings = load_embeddings()
            st.session_state.llm = load_llm()
            if st.session_state.llm:
                st.session_state.models_loaded = True
        if st.session_state.models_loaded:
            st.success("✅ Models ready!")
            time.sleep(1)
            st.rerun()

    if st.session_state.models_loaded and not st.session_state.documents_loaded and st.session_state.pdf_source in ["github", "local"]:
        with st.spinner("📚 Loading documents into FAISS vector store..."):
            if st.session_state.pdf_source == "github":
                rag_chain, num_chunks, loaded_files = load_pdfs_from_github(st.session_state.github_repo_url)
            else:
                rag_chain, num_chunks, loaded_files = load_pdfs_from_folder(st.session_state.local_folder_path)

            if rag_chain:
                st.session_state.rag_chain = rag_chain
                st.session_state.documents_loaded = True

                st.markdown(f"""
                <div class="document-info">
                    <h4>📄 Successfully loaded {len(loaded_files)} PDF documents into FAISS:</h4>
                    <ul>
                        {"".join([f"<li>{file}</li>" for file in loaded_files])}
                    </ul>
                    <p><strong>Total chunks:</strong> {num_chunks}</p>
                    <p><strong>Vector Store:</strong> FAISS (Fast similarity search)</p>
                </div>
                """, unsafe_allow_html=True)

                st.success("✅ Documents ready for Q&A with FAISS!")
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ Failed to load documents. Please check your configuration.")

    if st.session_state.rag_chain:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

        for message in st.session_state.chat_history:
            display_chat_message(message["content"], message["is_user"])

        if st.session_state.processing_query:
            display_thinking_indicator()

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)

        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])

            with col1:
                user_question = st.text_input(
                    "Type your question...",
                    placeholder="Ask anything about the documents...",
                    disabled=st.session_state.processing_query,
                    label_visibility="collapsed"
                )

            with col2:
                send_button = st.form_submit_button(
                    "📤 Send",
                    type="primary",
                    disabled=st.session_state.processing_query
                )

        st.markdown("</div>", unsafe_allow_html=True)

        # Process user input
        if send_button and user_question.strip() and not st.session_state.processing_query:
            st.session_state.processing_query = True

            st.session_state.chat_history.append({
                "content": user_question,
                "is_user": True
            })
            st.rerun()

        if st.session_state.processing_query and len(st.session_state.chat_history) > 0:
            if not st.session_state.chat_history[-1]["is_user"]:
                st.session_state.processing_query = False
            else:
                last_question = st.session_state.chat_history[-1]["content"]
                answer = process_user_query(last_question)

                st.session_state.chat_history.append({
                    "content": answer,
                    "is_user": False
                })

                st.session_state.processing_query = False

                st.rerun()
    else:
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h3>👋 Welcome to Enhanced PDF RAG Assistant!</h3>
            <p>This version supports multiple input methods:</p>
            <ul style='text-align: left; max-width: 500px; margin: 0 auto;'>
                <li><strong>📎 Upload Individual Files:</strong>
                    <ul>
                        <li>PDF documents (.pdf)</li>
                        <li>Word documents (.docx)</li>
                        <li>Excel spreadsheets (.xlsx, .xls)</li>
                    </ul>
                </li>
                <li><strong>📁 Upload Folder (ZIP):</strong> Upload a ZIP file containing multiple documents</li>
                <li><strong>🔗 GitHub Repository:</strong> Load PDF files from a GitHub repository</li>
                <li><strong>📂 Local Folder:</strong> Load files from a local folder path</li>
            </ul>
            <br>
            <p><strong>To get started:</strong></p>
            <ol style='text-align: left; max-width: 500px; margin: 0 auto;'>
                <li>Choose your preferred document source in the sidebar</li>
                <li>Upload files or configure repository/folder settings</li>
                <li>Process your documents</li>
                <li>Start asking questions!</li>
            </ol>
            <br>
            <p><strong>Default Repository:</strong><br>
            <code>https://github.com/Jennifer1907/Time-Series-Team-Hub/tree/main/assets/pdf</code></p>
            <br>
            <p><strong>Features:</strong></p>
            <ul style='text-align: left; max-width: 500px; margin: 0 auto;'>
                <li>✨ Multi-format support (PDF, Word, Excel)</li>
                <li>🚀 FAISS vector store for fast similarity search</li>
                <li>🔄 Multiple input methods</li>
                <li>💬 ChatGPT-like conversation interface</li>
                <li>🎯 Context-aware responses</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()