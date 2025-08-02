import streamlit as st
import os
import json
import faiss
import numpy as np
from spam_model import load_dataset, run_enhanced_pipeline, enhanced_spam_classifier_pipeline, model, tokenizer, device

# Set up the Streamlit page
st.set_page_config(page_title="Spam Slayer", layout="wide")
st.title("Spam Slayer: Multilingual Spam Classifier")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
language = st.sidebar.radio("Select Language", ('English', 'Vietnamese'))
data_source = 'gdrive' if language == 'English' else 'kaggle'

# Model directory and file paths
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

FAISS_INDEX = os.path.join(MODEL_DIR, f"{language.lower()}_faiss.index")
TRAIN_METADATA_PATH = os.path.join(MODEL_DIR, f"{language.lower()}_train_metadata.json")
CLASS_WEIGHTS_PATH = os.path.join(MODEL_DIR, f"{language.lower()}_class_weights.json")
BEST_ALPHA_PATH = os.path.join(MODEL_DIR, f"{language.lower()}_best_alpha.txt")

# Check if model files exist
model_exists = (
    os.path.exists(FAISS_INDEX) and
    os.path.exists(TRAIN_METADATA_PATH) and
    os.path.exists(CLASS_WEIGHTS_PATH) and
    os.path.exists(BEST_ALPHA_PATH)
)

# Training button logic
if st.sidebar.button("Train Model", disabled=False):
    with st.spinner(f"Training model for {language}... This might take a few minutes."):
        try:
            messages, labels = load_dataset(source=data_source)
            pipeline_results = run_enhanced_pipeline(messages, labels, use_augmentation=True)

            faiss.write_index(pipeline_results['index'], FAISS_INDEX)
            with open(TRAIN_METADATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(pipeline_results['train_metadata'], f, ensure_ascii=False)
            with open(CLASS_WEIGHTS_PATH, 'w', encoding='utf-8') as f:
                json.dump(pipeline_results['class_weights'], f, ensure_ascii=False)
            with open(BEST_ALPHA_PATH, 'w') as f:
                f.write(str(pipeline_results['best_alpha']))
            
            st.success(f"Model for {language} trained and saved successfully!")
            st.rerun() # Rerun to update the model_exists flag
        except Exception as e:
            st.error(f"An error occurred during training: {e}")

if not model_exists:
    st.sidebar.warning("Click 'Train Model' to begin.")
else:
    st.sidebar.success(f"Model for {language} is ready!")

# --- Main Inference Interface ---
st.header("Spam Classifier")

if not model_exists:
    st.warning("Please train a model first using the sidebar.")
else:
    # Load model artifacts using caching
    @st.cache_resource
    def load_model_artifacts():
        st.info(f"Loading {language} model artifacts...")
        loaded_index = faiss.read_index(FAISS_INDEX)
        with open(TRAIN_METADATA_PATH, 'r', encoding='utf-8') as f:
            loaded_train_metadata = json.load(f)
        with open(CLASS_WEIGHTS_PATH, 'r', encoding='utf-8') as f:
            loaded_class_weights = json.load(f)
        with open(BEST_ALPHA_PATH, 'r') as f:
            loaded_best_alpha = float(f.read())
        st.success("Model loaded successfully!")
        return loaded_index, loaded_train_metadata, loaded_class_weights, loaded_best_alpha

    try:
        index, train_metadata, class_weights, best_alpha = load_model_artifacts()

        user_input = st.text_area("Enter your message here...", height=150)
        
        if st.button("Classify"):
            if not user_input:
                st.error("Please enter a message to classify.")
            else:
                with st.spinner("Classifying message..."):
                    result = enhanced_spam_classifier_pipeline(
                        user_input, 
                        index, 
                        train_metadata, 
                        class_weights, 
                        best_alpha, 
                        k=5, 
                        explain=True
                    )
                
                st.subheader("Classification Result")
                prediction = result['prediction'].upper()
                if prediction == 'SPAM':
                    st.markdown(f"### 🚨 The message is likely **<span style='color:red'>SPAM</span>**.", unsafe_allow_html=True)
                    if result['subcategory']:
                        st.info(f"Spam Subcategory: `{result['subcategory']}`")
                else:
                    st.markdown(f"### ✅ The message is **<span style='color:green'>HAM</span>**.", unsafe_allow_html=True)
                
                st.markdown(f"**Vote Scores:** Ham={result['vote_scores']['ham']:.3f}, Spam={result['vote_scores']['spam']:.3f}")
                
                # --- Explainability Heatmap ---
                st.subheader("Explainability Heatmap")
                st.write("Words highlighted in red contributed most to the model's decision.")
                
                tokens = result['tokens']
                saliency_scores = result['saliency_scores']
                
                heatmap_html = ""
                for token, score in zip(tokens, saliency_scores):
                    score = max(0, score)
                    red_intensity = int(score * 255)
                    
                    if red_intensity > 0:
                        heatmap_html += f"<span style='background-color:rgba(255, 0, 0, {score}); padding: 2px; border-radius: 4px;'>{token}</span> "
                    else:
                        heatmap_html += f"<span style='padding: 2px; border-radius: 4px;'>{token}</span> "
                
                st.markdown(f'<div style="line-height: 2;">{heatmap_html}</div>', unsafe_allow_html=True)
    
                # --- Top Neighbors ---
                st.subheader("Top 5 Nearest Neighbors")
                for i, neighbor in enumerate(result['neighbors'], 1):
                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
                        st.markdown(f"**{i}.**")
                    with col2:
                        label_color = "green" if neighbor['label'] == 'ham' else "red"
                        st.markdown(f"**<span style='color:{label_color}'>{neighbor['label'].upper()}</span>**", unsafe_allow_html=True)
                    with col3:
                        st.write(f"{neighbor['message']}")

    except Exception as e:
        st.error(f"Error loading model artifacts: {e}. Please try retraining the model.")
