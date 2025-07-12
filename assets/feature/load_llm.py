import streamlit as st
import tempfile
import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline

from langchain.memory import ConversationBufferMemory  # Deprecated
from langchain_community.chat_message_histories import ChatMessageHistory  # Deprecated
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain  # Deprecated
from langchain_experimental.text_splitter import SemanticChunker

from langchain.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.prompts import PromptTemplate
import json

# Đọc HuggingFace token từ file
@st.cache_resource
def get_hg_token():
    return "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"

# Khởi tạo session state
if 'llm' not in st.session_state:
    st.session_state.llm = None

# ✅ Load LLM mà KHÔNG dùng quantization_config
@st.cache_resource
def load_llm():
    MODEL_NAME = "google/gemma-2b-it"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # optional
        token=get_hg_token(),
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_pipeline = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        device_map="auto"
    )

    return HuggingFacePipeline(pipeline=model_pipeline)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

# Tải models nếu chưa có
if not st.session_state.get("models_loaded", False):
    st.info("Đang tải models...")
    st.session_state.embeddings = load_embeddings()
    st.session_state.llm = load_llm()
    st.session_state.models_loaded = True
    st.success("Models đã sẵn sàng!")
    st.rerun()
