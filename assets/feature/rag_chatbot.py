import streamlit as st
import os
import torch
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface.llms import HuggingFacePipeline
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
import time
import tempfile
import shutil

st.set_page_config(
    page_title="PDF RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }

    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }

    .user-message {
        background-color: #000000;
        border-radius: 18px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-left: 20%;
        text-align: left;
    }

    .assistant-message {
        background-color: #006400;
        border-radius: 18px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-right: 20%;
        text-align: left;
    }

    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
        z-index: 1000;
    }

    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 12px 20px;
    }

    .document-info {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }

    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-ready {
        background-color: #28a745;
    }

    .status-loading {
        background-color: #ffc107;
    }

    .status-error {
        background-color: #dc3545;
    }

    .upload-section {
        background-color: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }

    .method-tab {
        padding: 0.5rem 1rem;
        margin: 0.2rem;
        border-radius: 8px;
        cursor: pointer;
        display: inline-block;
        transition: all 0.3s;
    }

    .method-tab.active {
        background-color: #007bff;
        color: white;
    }

    .method-tab.inactive {
        background-color: #e9ecef;
        color: #6c757d;
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
if 'pdf_folder_path' not in st.session_state:
    st.session_state.pdf_folder_path = "./knowledge_base"
if 'upload_method' not in st.session_state:
    st.session_state.upload_method = "folder"
if 'uploaded_files_info' not in st.session_state:
    st.session_state.uploaded_files_info = []

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

@st.cache_resource
def load_llm():
    MODEL_NAME = "lmsys/vicuna-7b-v1.5"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )
    return HuggingFacePipeline(pipeline=model_pipeline)

def save_uploaded_files(uploaded_files):
    """Save uploaded files to temporary directory and return paths"""
    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        # Create temp file path
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save uploaded file
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
    
    # Save uploaded files temporarily
    temp_paths, temp_dir = save_uploaded_files(uploaded_files)
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        for i, (temp_path, uploaded_file) in enumerate(zip(temp_paths, uploaded_files)):
            try:
                status_text.text(f"Đang xử lý: {uploaded_file.name}")
                
                loader = PyPDFLoader(temp_path)
                documents = loader.load()
                all_documents.extend(documents)
                loaded_files.append(uploaded_file.name)
                progress_bar.progress((i + 1) / len(temp_paths))
                
                st.success(f"✅ Đã xử lý: {uploaded_file.name} ({len(documents)} pages)")
                
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý {uploaded_file.name}: {str(e)}")
                continue
    
    finally:
        # Clean up temporary files
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
        progress_bar.empty()
        status_text.empty()

    if not all_documents:
        return None, 0, loaded_files

    # Create semantic chunks
    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )

    docs = semantic_splitter.split_documents(all_documents)
    vector_db = Chroma.from_documents(documents=docs, embedding=st.session_state.embeddings)
    retriever = vector_db.as_retriever()

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

def load_pdfs_from_folder(folder_path):
    """Load all PDF files from the specified folder"""
    cleaned_path = folder_path.strip().strip('"').strip("'")
    folder = Path(cleaned_path)

    st.write(f"🔍 **Debug info:**")
    st.write(f"- Đường dẫn gốc: `{folder_path}`")
    st.write(f"- Đường dẫn đã làm sạch: `{cleaned_path}`")
    st.write(f"- Đường dẫn tuyệt đối: `{folder.absolute()}`")
    st.write(f"- Folder tồn tại: `{folder.exists()}`")

    if not folder.exists():
        st.error(f"❌ Folder không tồn tại: `{cleaned_path}`")
        st.info("💡 **Gợi ý khắc phục:**")
        st.info("1. Kiểm tra đường dẫn có đúng không")
        st.info("2. Tạo thư mục nếu chưa có")
        st.info("3. Sử dụng đường dẫn ngắn hơn (ví dụ: `C:\\knowledge_base`)")
        return None, 0, []

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        st.warning(f"Không thể tìm thấy file PDF nào trong folder: {cleaned_path}")
        return None, 0, []

    all_documents = []
    loaded_files = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, pdf_file in enumerate(pdf_files):
        try:
            status_text.text(f"Đang xử lý: {pdf_file.name}")
            st.write(f"🔍 Processing file: {str(pdf_file)}")

            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            all_documents.extend(documents)
            loaded_files.append(pdf_file.name)
            progress_bar.progress((i + 1) / len(pdf_files))

            st.success(f"✅ Đã xử lý: {pdf_file.name} ({len(documents)} pages)")

        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý {pdf_file.name}: {str(e)}")
            st.write(f"🔍 Chi tiết lỗi: {type(e).__name__}")
            continue

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
    vector_db = Chroma.from_documents(documents=docs, embedding=st.session_state.embeddings)
    retriever = vector_db.as_retriever()

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

def display_chat_message(message, is_user=True):
    """Display a chat message with proper styling"""
    if is_user:
        st.markdown(f"""
        <div class="user-message">
            <strong>Bạn:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>AI Assistant:</strong> {message}
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 PDF RAG Assistant</h1>
        <p>Trợ lý AI thông minh - Hỏi đáp với tài liệu PDF bằng tiếng Việt</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Cấu hình")

        # Model loading status
        if st.session_state.models_loaded:
            st.markdown('<span class="status-indicator status-ready"></span>**Models:** Đã sẵn sàng', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-loading"></span>**Models:** Đang tải...', unsafe_allow_html=True)

        # Document loading status
        if st.session_state.documents_loaded:
            st.markdown('<span class="status-indicator status-ready"></span>**Tài liệu:** Đã tải', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-error"></span>**Tài liệu:** Chưa tải', unsafe_allow_html=True)

        st.divider()

        # Method selection
        st.subheader("📚 Chọn phương thức tải tài liệu")
        
        # Create tabs for different methods
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📁 Từ thư mục", use_container_width=True, 
                        type="primary" if st.session_state.upload_method == "folder" else "secondary"):
                st.session_state.upload_method = "folder"
                st.session_state.documents_loaded = False
                st.rerun()
        
        with col2:
            if st.button("📤 Upload trực tiếp", use_container_width=True,
                        type="primary" if st.session_state.upload_method == "upload" else "secondary"):
                st.session_state.upload_method = "upload"
                st.session_state.documents_loaded = False
                st.rerun()

        st.divider()

        # Configuration based on selected method
        if st.session_state.upload_method == "folder":
            st.subheader("📁 Cấu hình thư mục PDF")

            # Preset folder options
            preset_options = [
                "./knowledge_base",
                "C:/knowledge_base",
                "D:/pdf_docs",
                "Tùy chỉnh..."
            ]

            selected_preset = st.selectbox(
                "Chọn thư mục có sẵn:",
                preset_options
            )

            if selected_preset == "Tùy chỉnh...":
                folder_path = st.text_input(
                    "Đường dẫn thư mục chứa PDF:",
                    value=st.session_state.pdf_folder_path,
                    help="Nhập đường dẫn đến thư mục chứa các file PDF"
                )
            else:
                folder_path = selected_preset
                st.text_input(
                    "Đường dẫn hiện tại:",
                    value=folder_path,
                    disabled=True
                )

            if st.button("🔄 Tải từ thư mục", type="primary", use_container_width=True):
                st.session_state.pdf_folder_path = folder_path
                st.session_state.documents_loaded = False
                st.rerun()

        else:  # upload method
            st.subheader("📤 Upload file PDF")
            
            uploaded_files = st.file_uploader(
                "Chọn file PDF để upload:",
                type=['pdf'],
                accept_multiple_files=True,
                help="Bạn có thể chọn nhiều file PDF cùng lúc"
            )
            
            if uploaded_files:
                st.write(f"📋 **Đã chọn {len(uploaded_files)} file:**")
                for file in uploaded_files:
                    file_size = len(file.getbuffer()) / 1024 / 1024  # MB
                    st.write(f"- {file.name} ({file_size:.1f} MB)")
                
                if st.button("🚀 Xử lý file đã upload", type="primary", use_container_width=True):
                    st.session_state.uploaded_files_info = [(f.name, len(f.getbuffer())) for f in uploaded_files]
                    st.session_state.documents_loaded = False
                    # Store uploaded files in session state temporarily
                    st.session_state.current_uploaded_files = uploaded_files
                    st.rerun()

        st.divider()

        # Clear chat history
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        # Reset documents
        if st.button("🔄 Reset tài liệu", use_container_width=True):
            st.session_state.documents_loaded = False
            st.session_state.rag_chain = None
            st.session_state.uploaded_files_info = []
            if hasattr(st.session_state, 'current_uploaded_files'):
                del st.session_state.current_uploaded_files
            st.rerun()

    # Load models first
    if not st.session_state.models_loaded:
        with st.spinner("🚀 Đang khởi tạo AI models..."):
            st.session_state.embeddings = load_embeddings()
            st.session_state.llm = load_llm()
            st.session_state.models_loaded = True
        st.success("✅ Models đã sẵn sàng!")
        st.rerun()

    # Load documents based on selected method
    if st.session_state.models_loaded and not st.session_state.documents_loaded:
        if st.session_state.upload_method == "folder":
            with st.spinner("📚 Đang tải tài liệu từ thư mục..."):
                rag_chain, num_chunks, loaded_files = load_pdfs_from_folder(st.session_state.pdf_folder_path)
                
                if rag_chain:
                    st.session_state.rag_chain = rag_chain
                    st.session_state.documents_loaded = True
                    
                    # Display document info
                    st.markdown(f"""
                    <div class="document-info">
                        <h4>📄 Đã tải thành công {len(loaded_files)} tài liệu PDF từ thư mục:</h4>
                        <ul>
                            {"".join([f"<li>{file}</li>" for file in loaded_files])}
                        </ul>
                        <p><strong>Tổng số chunks:</strong> {num_chunks}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("✅ Tài liệu đã sẵn sàng cho việc hỏi đáp!")
                else:
                    st.error("❌ Không thể tải tài liệu. Vui lòng kiểm tra lại đường dẫn thư mục.")
        
        elif st.session_state.upload_method == "upload" and hasattr(st.session_state, 'current_uploaded_files'):
            with st.spinner("📚 Đang xử lý file đã upload..."):
                rag_chain, num_chunks, loaded_files = load_uploaded_pdfs(st.session_state.current_uploaded_files)
                
                if rag_chain:
                    st.session_state.rag_chain = rag_chain
                    st.session_state.documents_loaded = True
                    
                    # Display document info
                    st.markdown(f"""
                    <div class="document-info">
                        <h4>📄 Đã xử lý thành công {len(loaded_files)} file PDF đã upload:</h4>
                        <ul>
                            {"".join([f"<li>{file}</li>" for file in loaded_files])}
                        </ul>
                        <p><strong>Tổng số chunks:</strong> {num_chunks}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("✅ Tài liệu đã sẵn sàng cho việc hỏi đáp!")
                    
                    # Clean up
                    del st.session_state.current_uploaded_files
                else:
                    st.error("❌ Không thể xử lý file đã upload. Vui lòng thử lại.")

    # Chat interface
    if st.session_state.rag_chain:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

        # Display chat history
        for message in st.session_state.chat_history:
            display_chat_message(message["content"], message["is_user"])

        st.markdown("</div>", unsafe_allow_html=True)

        # Chat input (at the bottom)
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

        # Create two columns for input and button
        col1, col2 = st.columns([4, 1])

        with col1:
            user_question = st.text_input(
                "Nhập câu hỏi của bạn...",
                key="user_input",
                placeholder="Hỏi bất cứ điều gì về tài liệu...",
                on_change=None
            )

        with col2:
            send_button = st.button("📤 Gửi", type="primary")

        # Process user input
        if (send_button or user_question) and user_question.strip():
            # Add user message to chat history
            st.session_state.chat_history.append({
                "content": user_question,
                "is_user": True
            })

            # Generate response
            with st.spinner("🤔 Đang suy nghĩ..."):
                try:
                    output = st.session_state.rag_chain.invoke(user_question)
                    answer = output.split('Answer:')[1].strip() if 'Answer:' in output else output.strip()

                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "content": answer,
                        "is_user": False
                    })

                except Exception as e:
                    error_message = f"Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi: {str(e)}"
                    st.session_state.chat_history.append({
                        "content": error_message,
                        "is_user": False
                    })

            # Clear input and refresh
            st.rerun()

    else:
        # Welcome message when no documents are loaded
        current_method = "upload file trực tiếp" if st.session_state.upload_method == "upload" else "tải từ thư mục"
        
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem;'>
            <h3>👋 Chào mừng bạn đến với PDF RAG Assistant!</h3>
            <p>Hiện tại bạn đang sử dụng phương thức: <strong>{current_method}</strong></p>
            <br>
            <div class="upload-section">
                <h4>📚 Để bắt đầu, vui lòng chọn một trong hai cách:</h4>
                <br>
                <div style='display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;'>
                    <div style='text-align: left; max-width: 300px;'>
                        <h5>📁 Từ thư mục:</h5>
                        <ol>
                            <li>Chọn phương thức "Từ thư mục"</li>
                            <li>Cập nhật đường dẫn thư mục</li>
                            <li>Nhấn "Tải từ thư mục"</li>
                            <li>Bắt đầu hỏi đáp!</li>
                        </ol>
                    </div>
                    <div style='text-align: left; max-width: 300px;'>
                        <h5>📤 Upload trực tiếp:</h5>
                        <ol>
                            <li>Chọn phương thức "Upload trực tiếp"</li>
                            <li>Chọn file PDF từ máy tính</li>
                            <li>Nhấn "Xử lý file đã upload"</li>
                            <li>Bắt đầu hỏi đáp!</li>
                        </ol>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()