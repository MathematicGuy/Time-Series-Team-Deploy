import streamlit as st
import os
import requests
import tempfile
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import time

st.set_page_config(
    page_title="Lightweight Document RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simplified CSS
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
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    margin-bottom: 20px;
    background-color: #fafafa;
  }
  .user-message{
    background-color: #e3f2fd;
    border-radius: 15px;
    padding: 10px 15px;
    margin: 8px 0;
    margin-left: 15%;
    text-align: left;
  }
  .assistant-message{
    background-color: #f1f8e9;
    border-radius: 15px;
    padding: 10px 15px;
    margin: 8px 0;
    margin-right: 15%;
    text-align: left;
  }
  .thinking-indicator {
    background-color: #fff3e0;
    border-radius: 15px;
    padding: 10px 15px;
    margin: 8px 0;
    margin-right: 15%;
    text-align: left;
    animation: pulse 1.5s ease-in-out infinite;
  }
  @keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
  }
  .upload-area {
    border: 2px dashed #cccccc;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin: 10px 0;
  }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'processing_query' not in st.session_state:
    st.session_state.processing_query = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

@st.cache_resource
def load_embeddings():
    """Load lightweight embeddings model"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_document(file, file_type):
    """Load document based on file type"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.name)
    
    # Save uploaded file temporarily
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    
    try:
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == "docx":
            loader = Docx2txtLoader(file_path)
        elif file_type in ["xlsx", "xls"]:
            loader = UnstructuredExcelLoader(file_path)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return []
            
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading {file.name}: {str(e)}")
        return []
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)

def get_github_pdf_files(repo_url):
    """Get PDF files from GitHub repository"""
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

def load_github_documents(repo_url):
    """Load documents from GitHub repository"""
    pdf_files = get_github_pdf_files(repo_url)
    
    if not pdf_files:
        return [], []
    
    all_documents = []
    loaded_files = []
    
    for pdf_file in pdf_files:
        try:
            # Download PDF
            response = requests.get(pdf_file['download_url'])
            if response.status_code == 200:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(response.content)
                    temp_path = temp_file.name
                
                # Load PDF
                loader = PyPDFLoader(temp_path)
                documents = loader.load()
                all_documents.extend(documents)
                loaded_files.append(pdf_file['name'])
                
                # Clean up
                os.unlink(temp_path)
                
        except Exception as e:
            st.error(f"Error processing {pdf_file['name']}: {str(e)}")
    
    return all_documents, loaded_files

def create_rag_chain(documents, openai_api_key):
    """Create RAG chain from documents"""
    if not documents:
        return None
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    docs = text_splitter.split_documents(documents)
    
    # Create vector store
    vector_db = FAISS.from_documents(documents=docs, embedding=st.session_state.embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Create LLM with environment variable for API key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    llm = OpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo-instruct",  # Use completion model instead of chat
        openai_api_key=openai_api_key
    )
    
    # Create custom prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}
    
    Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def display_chat_message(message, is_user=True):
    """Display chat message"""
    if is_user:
        st.markdown(f"""
        <div class="user-message">
            <strong>You:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>AI:</strong> {message}
        </div>
        """, unsafe_allow_html=True)

def display_thinking_indicator():
    """Display thinking indicator"""
    st.markdown("""
    <div class="thinking-indicator">
        <strong>AI:</strong> 🤔 Thinking...
    </div>
    """, unsafe_allow_html=True)

def process_user_query(question):
    """Process user query"""
    try:
        output = st.session_state.rag_chain.invoke(question)
        return output
    except Exception as e:
        return f"Sorry, an error occurred: {str(e)}"

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Lightweight Document RAG Assistant</h1>
        <p>Upload PDF, Word, Excel files or use GitHub repository for Q&A</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key
        openai_api_key = st.text_input(
            "OpenAI API Key:",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key to continue")
            st.info("Get your API key from: https://platform.openai.com/api-keys")
        
        st.divider()
        
        # Document source selection
        st.subheader("📁 Document Source")
        
        doc_source = st.radio(
            "Choose source:",
            ["Upload Files", "GitHub Repository"],
            key="doc_source_radio"
        )
        
        if doc_source == "Upload Files":
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Upload your documents:",
                type=['pdf', 'docx', 'xlsx', 'xls'],
                accept_multiple_files=True,
                help="Supported formats: PDF, Word (.docx), Excel (.xlsx, .xls)"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_files and st.button("📤 Process Uploaded Files", type="primary"):
                if openai_api_key:
                    with st.spinner("Processing uploaded files..."):
                        all_documents = []
                        loaded_files = []
                        
                        for file in uploaded_files:
                            file_type = file.name.split('.')[-1].lower()
                            documents = load_document(file, file_type)
                            if documents:
                                all_documents.extend(documents)
                                loaded_files.append(file.name)
                                st.success(f"✅ Processed: {file.name}")
                        
                        if all_documents:
                            # Load embeddings
                            if not st.session_state.embeddings:
                                st.session_state.embeddings = load_embeddings()
                            
                            rag_chain = create_rag_chain(all_documents, openai_api_key)
                            if rag_chain:
                                st.session_state.rag_chain = rag_chain
                                st.session_state.documents_loaded = True
                                st.success(f"✅ Loaded {len(loaded_files)} files successfully!")
                                st.rerun()
                else:
                    st.error("Please enter your OpenAI API key first")
        
        else:  # GitHub Repository
            github_url = st.text_input(
                "GitHub Repository URL:",
                value="https://github.com/Jennifer1907/Time-Series-Team-Hub/tree/main/assets/pdf",
                help="URL to GitHub folder containing PDF files"
            )
            
            if st.button("📥 Load from GitHub", type="primary"):
                if openai_api_key:
                    with st.spinner("Loading documents from GitHub..."):
                        documents, loaded_files = load_github_documents(github_url)
                        
                        if documents:
                            # Load embeddings
                            if not st.session_state.embeddings:
                                st.session_state.embeddings = load_embeddings()
                            
                            rag_chain = create_rag_chain(documents, openai_api_key)
                            if rag_chain:
                                st.session_state.rag_chain = rag_chain
                                st.session_state.documents_loaded = True
                                st.success(f"✅ Loaded {len(loaded_files)} files from GitHub!")
                                st.rerun()
                        else:
                            st.error("No documents found or failed to load from GitHub")
                else:
                    st.error("Please enter your OpenAI API key first")
        
        st.divider()
        
        # Status
        if st.session_state.documents_loaded:
            st.success("📄 Documents loaded and ready!")
        else:
            st.info("📄 No documents loaded yet")
        
        # Clear chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.processing_query = False
            st.rerun()

    # Main chat interface
    if st.session_state.rag_chain and openai_api_key:
        # Chat container
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            display_chat_message(message["content"], message["is_user"])
        
        # Show thinking indicator
        if st.session_state.processing_query:
            display_thinking_indicator()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_question = st.text_input(
                    "Ask a question about your documents:",
                    placeholder="What would you like to know?",
                    disabled=st.session_state.processing_query,
                    label_visibility="collapsed"
                )
            
            with col2:
                send_button = st.form_submit_button(
                    "📤 Send",
                    type="primary",
                    disabled=st.session_state.processing_query
                )
        
        # Process input
        if send_button and user_question.strip() and not st.session_state.processing_query:
            st.session_state.processing_query = True
            st.session_state.chat_history.append({
                "content": user_question,
                "is_user": True
            })
            st.rerun()
        
        # Generate response
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
        # Welcome message
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h3>👋 Welcome to Document RAG Assistant!</h3>
            <p>This is a lightweight version that supports:</p>
            <ul style='text-align: left; max-width: 400px; margin: 0 auto;'>
                <li>📄 <strong>PDF files</strong> - Research papers, reports, manuals</li>
                <li>📝 <strong>Word documents</strong> - .docx files</li>
                <li>📊 <strong>Excel files</strong> - .xlsx, .xls spreadsheets</li>
                <li>🔗 <strong>GitHub repositories</strong> - PDF collections</li>
            </ul>
            <br>
            <p><strong>To get started:</strong></p>
            <ol style='text-align: left; max-width: 400px; margin: 0 auto;'>
                <li>Enter your OpenAI API key in the sidebar</li>
                <li>Upload files or use the GitHub repository</li>
                <li>Start asking questions!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()