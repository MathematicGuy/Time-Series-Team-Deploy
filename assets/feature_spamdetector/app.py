import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import json
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Import your training modules
from spam_model import SpamClassifier
from utils import load_model_artifacts, save_model_artifacts

# Configure Streamlit page
st.set_page_config(
    page_title="🛡️ Spam Slayer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .spam-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        color: #c62828;
    }
    .ham-box {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        color: #2e7d32;
    }
    .saliency-word {
        display: inline-block;
        margin: 2px;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    if 'training_progress' not in st.session_state:
        st.session_state.training_progress = 0
    if 'training_status' not in st.session_state:
        st.session_state.training_status = ""
    if 'current_language' not in st.session_state:
        st.session_state.current_language = None

def render_saliency_heatmap(tokens, saliency_scores):
    """Render saliency heatmap using Plotly"""
    if not tokens or not saliency_scores:
        return None

    # Normalize saliency scores to 0-1 range
    if len(saliency_scores) > 1:
        min_score = min(saliency_scores)
        max_score = max(saliency_scores)
        if max_score > min_score:
            normalized_scores = [(s - min_score) / (max_score - min_score) for s in saliency_scores]
        else:
            normalized_scores = [0.5] * len(saliency_scores)
    else:
        normalized_scores = [0.5] * len(saliency_scores)

    # Create HTML with colored spans
    html_content = ""
    for token, score in zip(tokens, normalized_scores):
        # Color intensity based on saliency score
        red_intensity = int(255 * score)
        color = f"rgba(255, {255 - red_intensity}, {255 - red_intensity}, 0.7)"
        html_content += f'<span class="saliency-word" style="background-color: {color};">{token}</span> '

    return html_content

def load_trained_model(language):
    """Load pre-trained model artifacts if they exist"""
    model_resource_path = model_language_code(language) # model_resources/vi or model_resources/en
    print("model_resource_path:", model_resource_path)
    model_files = [
        f'model_resources/{model_resource_path}/model_artifacts.pkl',
        f'model_resources/{model_resource_path}/faiss_index.bin',
        f'model_resources/{model_resource_path}/train_metadata.json',
        f'model_resources/{model_resource_path}/class_weights.json',
        f'model_resources/{model_resource_path}/model_config.json'
    ]
    print('model_files:', model_files)


    classifier = SpamClassifier(classification_language=model_resource_path) # initialize SpamClassifier class from spam_model.py to use .load_from_files function

    if all(os.path.exists(f) for f in model_files):
        try:
            st.session_state.classifier = classifier.load_from_files() # load model_config.json from model_resources
            st.session_state.model_trained = True
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False

    return False

def train_model_callback(classification_language):
    """Callback function for model training"""
    try:
        # Initialize classifier
        classifier = SpamClassifier(classification_language=classification_language) #? Define Classification Language for Model

        # Update progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Loading dataset...")
        progress_bar.progress(10)

        # Load data based on classification_language selection
        data_source = 'kaggle' if classification_language == 'Vietnamese' else 'gdrive'
        messages, labels = classifier.load_dataset(source=data_source)

        status_text.text(f"Loaded {len(messages)} messages. Starting training...")
        progress_bar.progress(30)

        # Train the model
        results = classifier.train(
            messages, labels,
            progress_callback=lambda p, msg: (
                progress_bar.progress(min(30 + int(p * 0.6), 90)),
                status_text.text(msg)
            )
        )

        # Save model artifacts
        classifier.save_to_files()

        # Update session state
        st.session_state.classifier = classifier
        st.session_state.model_trained = True

        progress_bar.progress(100)
        status_text.text("Training completed successfully!")

        # Show training results
        st.success("✅ Model training completed!")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Size", len(messages))
        with col2:
            st.metric("Best Alpha", f"{results['best_alpha']:.2f}")
        with col3:
            best_accuracy = max(results['accuracy_results'].values())
            st.metric("Best Accuracy", f"{best_accuracy:.1%}")

        return True

    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return False


def reset_to_welcome():
    """Reset session state to show the Welcome screen."""
    st.session_state.model_trained = False
    st.session_state.current_language = None
    st.session_state.classifier = None

def model_language_code(classification_language):
    match classification_language:
        case 'English':
            return 'en'
        case 'Vietnamese':
            return 'vi'
        case _:
            return 'None'

def check_model_ready(model_path):
    #? Check if Embedding Model is train or not
    if os.path.isfile(model_path):
        print("Trained True")
        st.session_state.model_trained = True
    else:
        print(f'Trained False')
        st.session_state.model_trained = False

def main():
    """Main Streamlit application"""
    initialize_session_state()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Spam Slayer</h1>
        <p>Advanced Multilingual Spam Detection with Explainable AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        # Language selection
        classification_language = st.selectbox(
            "Select Language",
            ["None", "Vietnamese", "English"],
            help="Vietnamese: Loads Kaggle dataset\nEnglish: Loads Google Drive dataset"
        )
        st.write(f"**Selected Language:** {classification_language}")

        st.markdown("---")

        if not os.path.exists('model_resources/'):
            os.makedirs('model_resources/vi', exist_ok=True)
            os.makedirs('model_resources/en', exist_ok=True)
            print(f"Created directory: {'model_resources/'}")

        model_path = f"model_resources/{model_language_code(classification_language)}/model_config.json"
        print(model_path)

        # check model trained or not, if trained st.session_state.model_trained = True else False
        check_model_ready(model_path)

        st.subheader("📊 Model Status")
        
        print('Model Status:', st.session_state.model_trained)
        if st.session_state.model_trained:
            #? Update Embedding model each time a new language get chosen
            if st.session_state.current_language != classification_language:
                st.session_state.current_language = classification_language
                print('current_languages:', st.session_state.current_language)

                with st.spinner(f"Loading {classification_language} model..."): # add loading icon when function still running
                    if load_trained_model(language=classification_language):
                        st.success("Model Ready !")

            # Always display model status metrics when model is trained
            with open(model_path, 'r', encoding='utf-8') as f:
                train_result = json.load(f)
                model_info = train_result['model_info']
                print('Model Infor:', model_info)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dataset Size", model_info['dataset_size'])
            with col2:
                st.metric("Best Alpha", f"{model_info['best_alpha']:.2f}")
            with col3:
                best_accuracy = max(model_info['accuracy_results'].values())
                st.metric("Best Accuracy", f"{best_accuracy:.1%}")
        else:
            st.warning("⏳ Model Not Trained")

        st.markdown("---")

        # Training section
        st.subheader("🎯 Model Training")
        if st.button("🚀 Train New Model", disabled=False):
            st.info(f"Starting training with {classification_language} dataset...")
            train_model_callback(classification_language)

    # Main content
    if not st.session_state.model_trained:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to Spam Slayer!

        Get started by training a model:

        1. **Select Language** in the sidebar (Vietnamese or English)
        2. **Click "Train New Model"** to start training
        3. **Wait for training to complete** (this may take a few minutes)
        4. **Start classifying messages!**

        ### 🌟 Features:
        - **Multilingual Support**: Vietnamese and English datasets
        - **Advanced AI**: Uses multilingual E5 embeddings with weighted KNN
        - **Explainable AI**: Saliency heatmaps show which words influence predictions
        - **Spam Subcategorization**: Detailed spam type classification
        - **Real-time Processing**: Instant classification results
        """)

        # Show training demo
        with st.expander("📖 How It Works"):
            st.markdown("""
            **Spam Slayer** uses state-of-the-art machine learning techniques:

            1. **Text Embedding**: Converts messages to numerical representations using multilingual E5
            2. **Similarity Search**: Uses FAISS for efficient nearest neighbor search
            3. **Weighted Classification**: Combines similarity scores with class weights and saliency
            4. **Explainability**: Computes token-level importance scores
            5. **Subcategorization**: Classifies spam into specific types (promotional, system alerts, etc.)
            """)

    else:
        # Classification interface
        st.markdown("## 🔍 Message Classification")

        # Input section
        col1, col2 = st.columns([3, 1])

        with col1:
            user_message = st.text_area(
                "Enter message to classify:",
                placeholder="Type your message here...",
                height=100
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            classify_button = st.button("🔍 Classify", type="primary")

            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                k_neighbors = st.slider("K Neighbors", 1, 20, 5)
                show_neighbors = st.checkbox("Show Similar Messages", False)
                explain_prediction = st.checkbox("Show Explainability", True)

        # Classification results
        if classify_button and user_message.strip():
            with st.spinner("Analyzing message..."):
                try:
                    # Get prediction
                    result = st.session_state.classifier.classify_message(
                        user_message,
                        k=k_neighbors,
                        explain=explain_prediction
                    )

                    # Display prediction
                    prediction = result['prediction']
                    vote_scores = result['vote_scores']

                    if prediction == 'spam':
                        st.markdown(f"""
                        <div class="prediction-box spam-box">
                            🚨 SPAM DETECTED 🚨
                        </div>
                        """, unsafe_allow_html=True)

                        # Show subcategory
                        if 'subcategory' in result and result['subcategory']:
                            subcategory_map = {
                                'spam_quangcao': '📢 Promotional/Advertisement',
                                'spam_hethong': '⚠️ System Alert/Phishing',
                                'spam_khac': '🔍 Other Spam Type'
                            }
                            subcategory_name = subcategory_map.get(
                                result['subcategory'],
                                result['subcategory']
                            )
                            st.info(f"**Spam Type:** {subcategory_name}")

                    else:
                        st.markdown(f"""
                        <div class="prediction-box ham-box">
                            ✅ LEGITIMATE MESSAGE ✅
                        </div>
                        """, unsafe_allow_html=True)

                    # Show confidence scores
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Ham Score", f"{vote_scores['ham']:.3f}")
                    with col2:
                        st.metric("Spam Score", f"{vote_scores['spam']:.3f}")
                    with col3:
                        confidence = max(vote_scores.values()) / sum(vote_scores.values())
                        st.metric("Confidence", f"{confidence:.1%}")

                    # Explainability section
                    if explain_prediction and 'tokens' in result and 'saliency_scores' in result:
                        st.markdown("### 🔬 Explainability Analysis")

                        # Saliency heatmap
                        heatmap_html = render_saliency_heatmap(
                            result['tokens'],
                            result['saliency_scores']
                        )

                        if heatmap_html:
                            st.markdown("**Word Importance Heatmap:**")
                            st.markdown(f'<div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #fafafa;">{heatmap_html}</div>', unsafe_allow_html=True)
                            st.caption("Darker red indicates higher influence on spam classification")

                    # Similar messages section
                    if show_neighbors and 'neighbors' in result:
                        st.markdown("### 📋 Similar Training Messages")

                        for i, neighbor in enumerate(result['neighbors'][:5], 1):
                            with st.expander(f"Similar Message #{i} - {neighbor['label'].upper()}"):
                                st.write(f"**Similarity:** {neighbor['score']:.3f}")
                                st.write(f"**Weight:** {neighbor['weight']:.3f}")
                                st.write(f"**Message:** {neighbor['message']}")

                except Exception as e:
                    st.error(f"Classification error: {str(e)}")

        elif classify_button:
            st.warning("Please enter a message to classify.")

        # Usage statistics
        with st.expander("📊 Model Information"):
            if hasattr(st.session_state.classifier, 'model_info'):
                info = st.session_state.classifier.model_info
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Training Dataset:**", info.get('dataset_size', 'N/A'))
                    st.write("**Model Type:**", info.get('model_name', 'multilingual-e5-base'))
                with col2:
                    st.write("**Best Alpha:**", info.get('best_alpha', 'N/A'))
                    st.write("**Training Date:**", info.get('training_date', 'N/A'))

if __name__ == "__main__":
    # SPAM:  "Hey John, you might get $500 cashback if you install."
    main()