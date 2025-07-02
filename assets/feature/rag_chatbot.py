import streamlit as st
import os
import torch
import requests
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
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

st.set_page_config(
    page_title = "PDF RAG Assistant",
    layout = "wide",
    initial_sidebar_state = "expanded"
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
    background-color: #003300;
    border-radius: 18px;
    padding: 12px 16px;
    margin: 8px 0;
    margin-left: 20%;
    text-align: left;
    border: 1px solid #2196f3;
  }
  .assistant-message{
    background-color: #000000;
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
        border: 2px dashed #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
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
if 'uploaded_files_info' not in st.session_state:
    st.session_state.uploaded_files_info = []

@st.cache_resource
def load_embeddings():
  return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

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

def get_github_pdf_files(repo_url):
  """
  Link github hiện tại là giao diện dành cho người dùng (HTML), không phải API dành cho máy móc.
  Để đọc trực tiếp danh sách file từ URL này, phải chuyển link github hiện tại sang API"""
  try:
    if "github.com" in repo_url and "/tree/" in repo_url:
      parts = repo_url.replace("https://github.com/", "").split("/tree/")
      repo_path = parts[0]
      branch_and_path = parts[1].split("/",1)
      branch = branch_and_path[0]
      folder_path = branch_and_path[1] if len(branch_and_path)>1 else ""

      api_url = f"https://api.github.com/repos/{repo_path}/contents/{folder_path}?ref={branch}"
    else:
      st.error("Invalid GitHub URL format")
      return[]

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

def save_uploaded_files(uploaded_files):
    """Save uploaded files to temporary directory and return paths"""
    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(temp_path)
    
    return saved_paths, temp_dir

def load_uploaded_pdfs(uploaded_files):
    """Load PDF files from uploaded files"""
    if not uploaded_files:
        return None, 0, []

    all_documents = []
    loaded_files = []
    
    temp_paths, temp_dir = save_uploaded_files(uploaded_files)
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        for i, (temp_path, uploaded_file) in enumerate(zip(temp_paths, uploaded_files)):
            try:
                status_text.text(f"Processing: {uploaded_file.name}")
                
                loader = PyPDFLoader(temp_path)
                documents = loader.load()
                all_documents.extend(documents)
                loaded_files.append(uploaded_file.name)
                progress_bar.progress((i + 1) / len(temp_paths))
                
                st.success(f"✅ Processed: {uploaded_file.name} ({len(documents)} pages)")
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                continue
    
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
        progress_bar.empty()
        status_text.empty()

    if not all_documents:
        return None, 0, loaded_files

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
    
    return rag_chain, len(docs), loaded_files

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

  if not all_documents:
    return None, 0, loaded_files

  semantic_splitter = SemanticChunker(
      embeddings = st.session_state.embeddings,
      buffer_size = 1,
      breakpoint_threshold_type = "percentile",
      breakpoint_threshold_amount = 95,
      min_chunk_size = 500,
      add_start_index = True
  )

  docs = semantic_splitter.split_documents(all_documents)
  
  # FAISS implementation - Changed from Chroma
  vector_db = FAISS.from_documents(documents=docs, embedding=st.session_state.embeddings)
  retriever = vector_db.as_retriever(search_kwargs={"k": 4})  # You can adjust k value

  prompt = hub.pull("rlm/rag-prompt")

  def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
  
  rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )

  # Clean up temporary directory
  shutil.rmtree(temp_dir)

  return rag_chain, len(docs), loaded_files

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

    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )

    docs = semantic_splitter.split_documents(all_documents)
    
    # FAISS implementation - Changed from Chroma
    vector_db = FAISS.from_documents(documents=docs, embedding=st.session_state.embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})  # You can adjust k value

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )
    return rag_chain, len(docs), loaded_files

def display_chat_message(message, is_user = True):
  if is_user:
    st.markdown(f"""
    <div class = "user-message">
        <strong>You:</strong> {message}
    </div>
    """, unsafe_allow_html = True)
  else:
    st.markdown(f"""
    <div class = "assistant-message">
        <strong>AI Assistant:</strong> {message}
    </div>
    """, unsafe_allow_html = True)

def display_thinking_indicator():
  st.markdown(f"""
  <div class = "thinking-indicator">
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
        <h1>🤖 PDF RAG Assistant with FAISS</h1>
        <p>Smart AI Assistant - Q&A with PDF documents using FAISS vector store</p>
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
            ["GitHub Repository (Default)", "Local Folder", "Upload Files"],
            key="pdf_source_radio"
        )

      if pdf_source == "GitHub Repository (Default)":
          st.session_state.pdf_source = "github"
          github_url = st.text_input(
              "GitHub Repository URL:",
              value=st.session_state.github_repo_url,
              help="URL to GitHub folder containing PDF files"
          )
          st.session_state.github_repo_url = github_url
          
      elif pdf_source == "Local Folder":
          st.session_state.pdf_source = "local"
          local_path = st.text_input(
              "Local Folder Path:",
              value=st.session_state.local_folder_path,
              help="Path to local folder containing PDF files"
          )
          st.session_state.local_folder_path = local_path
          
      else:  # Upload Files
          st.session_state.pdf_source = "upload"
          
          st.markdown("""
          <div class="upload-section">
              <h4>📤 Upload PDF Files</h4>
              <p>Select one or multiple PDF files to upload</p>
          </div>
          """, unsafe_allow_html=True)
          
          uploaded_files = st.file_uploader(
              "Choose PDF files:",
              type=['pdf'],
              accept_multiple_files=True,
              help="You can select multiple PDF files at once (Recommended < 10MB per file)"
          )
          
          if uploaded_files:
              st.write(f"📋 **Selected {len(uploaded_files)} files:**")
              total_size = 0
              for file in uploaded_files:
                  file_size = len(file.getbuffer()) / 1024 / 1024  # MB
                  total_size += file_size
                  st.write(f"- {file.name} ({file_size:.1f} MB)")
              
              if total_size > 50:
                  st.warning("⚠️ Total file size > 50MB may cause slow processing")
              
              # Store uploaded files in session state
              st.session_state.current_uploaded_files = uploaded_files
              st.session_state.uploaded_files_info = [(f.name, len(f.getbuffer())) for f in uploaded_files]

      # Load documents button
      if st.button("🔄 Load Documents", type="primary"):
          if st.session_state.pdf_source == "upload" and not hasattr(st.session_state, 'current_uploaded_files'):
              st.error("❌ Please select PDF files to upload first!")
          else:
              st.session_state.documents_loaded = False
              st.rerun()

      if st.button("🗑️ Clear Chat History"):
          st.session_state.chat_history = []
          st.session_state.processing_query = False
          st.rerun()

      # Reset documents button
      if st.button("🔄 Reset Documents"):
          st.session_state.documents_loaded = False
          st.session_state.rag_chain = None
          st.session_state.uploaded_files_info = []
          if hasattr(st.session_state, 'current_uploaded_files'):
              del st.session_state.current_uploaded_files
          st.rerun()

      # FAISS specific settings
      st.divider()
      st.subheader("🔍 FAISS Settings")
      st.info("FAISS is a fast similarity search library. It's more memory efficient than Chroma.")

    if not st.session_state.models_loaded:
        with st.spinner("🚀 Initializing AI models..."):
            st.session_state.embeddings = load_embeddings()
            st.session_state.llm = load_llm()
            st.session_state.models_loaded = True
        st.success("✅ Models ready!")
        time.sleep(1)
        st.rerun()

    if st.session_state.models_loaded and not st.session_state.documents_loaded:
      with st.spinner("📚 Loading documents into FAISS vector store..."):
        if st.session_state.pdf_source == "github":
          rag_chain, num_chunks, loaded_files = load_pdfs_from_github(st.session_state.github_repo_url)
          source_desc = "GitHub repository"
        elif st.session_state.pdf_source == "local":
            rag_chain, num_chunks, loaded_files = load_pdfs_from_folder(st.session_state.local_folder_path)
            source_desc = "local folder"
        else:  # upload
            if hasattr(st.session_state, 'current_uploaded_files'):
                rag_chain, num_chunks, loaded_files = load_uploaded_pdfs(st.session_state.current_uploaded_files)
                source_desc = "uploaded files"
            else:
                st.error("❌ No files uploaded. Please select PDF files first.")
                st.stop()

        if rag_chain:
            st.session_state.rag_chain = rag_chain
            st.session_state.documents_loaded = True

            st.markdown(f"""
            <div class="document-info">
                <h4>📄 Successfully loaded {len(loaded_files)} PDF documents from {source_desc} into FAISS:</h4>
                <ul>
                    {"".join([f"<li>{file}</li>" for file in loaded_files])}
                </ul>
                <p><strong>Total chunks:</strong> {num_chunks}</p>
                <p><strong>Vector Store:</strong> FAISS (Fast similarity search)</p>
                <p><strong>Source:</strong> {source_desc.title()}</p>
            </div>
            """, unsafe_allow_html=True)

            st.success("✅ Documents ready for Q&A with FAISS!")
            
            # Clean up uploaded files from session state
            if st.session_state.pdf_source == "upload" and hasattr(st.session_state, 'current_uploaded_files'):
                del st.session_state.current_uploaded_files
                
            time.sleep(2)
            st.rerun()
        else:
            st.error("❌ Failed to load documents. Please check your configuration.")

    if st.session_state.rag_chain:
      st.markdown("<div class = 'chat-container'>", unsafe_allow_html= True)

      for message in st.session_state.chat_history:
        display_chat_message(message["content"], message["is_user"])

      if st.session_state.processing_query:
        display_thinking_indicator()

      st.markdown("</div>", unsafe_allow_html = True)

      st.markdown("<div class = 'chat-input-container'>", unsafe_allow_html = True)

      with st.form(key ="chat_form", clear_on_submit=True):
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

      if st.session_state.processing_query and len(st.session_state.chat_history)>0:
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
      current_source = {
          "github": "GitHub Repository",
          "local": "Local Folder", 
          "upload": "Upload Files"
      }.get(st.session_state.pdf_source, "GitHub Repository")
      
      st.markdown(f"""
      <div style='text-align: center; padding: 2rem;'>
        <h3>👋 Welcome to PDF RAG Assistant with FAISS!</h3>
        <p>Current source: <strong>{current_source}</strong></p>
        <br>
        <div class="upload-section">
            <h4>📚 Choose your preferred document source:</h4>
            <br>
            <div style='display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap;'>
                <div style='text-align: left; max-width: 280px;'>
                    <h5>🌐 GitHub Repository (Default):</h5>
                    <ul>
                        <li>Uses pre-configured repository</li>
                        <li>Automatic download from GitHub</li>
                        <li>Good for shared documents</li>
                    </ul>
                </div>
                <div style='text-align: left; max-width: 280px;'>
                    <h5>📁 Local Folder:</h5>
                    <ul>
                        <li>Use documents from local machine</li>
                        <li>Specify folder path</li>
                        <li>Good for private documents</li>
                    </ul>
                </div>
                <div style='text-align: left; max-width: 280px;'>
                    <h5>📤 Upload Files:</h5>
                    <ul>
                        <li>Upload PDF files directly</li>
                        <li>Multiple files supported</li>
                        <li>Good for quick testing</li>
                    </ul>
                </div>
            </div>
        </div>
        <br>
        <p><strong>Steps to get started:</strong></p>
        <ol style='text-align: left; max-width: 500px; margin: 0 auto;'>
            <li><strong>Choose your document source</strong> from the sidebar</li>
            <li><strong>Configure the source</strong> (URL, path, or upload files)</li>
            <li><strong>Click "Load Documents"</strong> to process into FAISS vector store</li>
            <li><strong>Start asking questions!</strong> The AI will answer based on your documents</li>
        </ol>
        <br>
        <p><strong>Default Repository:</strong><br>
        <code>https://github.com/Jennifer1907/Time-Series-Team-Hub/tree/main/assets/pdf</code></p>
        <br>
        <p><strong>FAISS Benefits:</strong> Faster similarity search, lower memory usage, better performance for large document collections.</p>
      </div>
      """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()