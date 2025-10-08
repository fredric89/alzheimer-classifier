import subprocess
import sys
import os

# Install PyTorch CPU versions if not available
try:
    import torch
    import torchvision
except ImportError:
    print("Installing PyTorch CPU versions...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "torch==2.0.1+cpu", "torchvision==0.15.2+cpu", 
        "-f", "https://download.pytorch.org/whl/torch_stable.html"
    ])
    # Restart to load the new packages
    os.execv(sys.executable, [sys.executable] + sys.argv)

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
import requests

# Rest of your app code...
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
import requests
import os

# Set page config
st.set_page_config(
    page_title="Alzheimer's MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .confidence-bar {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        height: 25px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .download-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Define the model architecture (must match your training)
class AlzheimerClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(AlzheimerClassifier, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=False)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

def download_model():
    """Download model file from Google Drive"""
    model_path = 'mobilenetv2_model.pth'
    
    # Your actual Google Drive download URL
    model_url = "https://drive.google.com/uc?export=download&id=18GarlSYHEPHfAQtzynxJqLT6yY8shOfZ"
    
    try:
        st.info("📥 Downloading model file... This may take a few minutes for first-time setup.")
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress tracking
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = downloaded_size / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"Downloaded {downloaded_size/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB")
        
        progress_bar.progress(1.0)
        status_text.text("Download complete!")
        st.success("✅ Model downloaded successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Error downloading model: {e}")
        st.error("""
        Please make sure:
        1. Your Google Drive link is correct and publicly accessible
        2. The file exists in Google Drive
        3. You have internet connection
        """)
        return False

@st.cache_resource
def load_model():
    """Load the trained model - with auto-download if needed"""
    model_path = 'mobilenetv2_model.pth'
    
    # Download model if it doesn't exist
    if not os.path.exists(model_path):
        success = download_model()
        if not success:
            return None
    
    try:
        model = AlzheimerClassifier(num_classes=4)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_class_info():
    """Load class names and descriptions"""
    try:
        with open('class_info.json', 'r') as f:
            return json.load(f)
    except:
        # Fallback if file doesn't exist
        return {
            'class_names': ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'],
            'class_descriptions': {
                'NonDemented': 'No signs of Alzheimer\'s disease detected',
                'VeryMildDemented': 'Very mild cognitive decline',
                'MildDemented': 'Mild cognitive decline - early stage Alzheimer\'s',
                'ModerateDemented': 'Moderate cognitive decline - middle stage Alzheimer\'s'
            }
        }

def get_transform():
    """Get the same transforms used during training"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

def predict_image(image, model):
    """Make prediction on uploaded image"""
    transform = get_transform()
    
    # Preprocess image
    image_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probabilities, 1).item()
        confidence = probabilities[0][predicted_class_idx].item()
    
    return predicted_class_idx, confidence, probabilities[0].numpy()

def plot_probabilities(probabilities, class_names):
    """Create a bar plot of probabilities"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
    bars = ax.barh(class_names, probabilities, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title('Classification Probabilities')
    
    # Add probability values on bars
    for bar, prob in zip(bars, probabilities):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Alzheimer\'s Disease MRI Classification</h1>', unsafe_allow_html=True)
    
    # Load model and class info
    model = load_model()
    class_info = load_class_info()
    class_names = class_info['class_names']
    class_descriptions = class_info['class_descriptions']
    
    if model is None:
        st.error("""
        ❌ Model could not be loaded. 
        
        Please refresh the page and try again. If the problem persists, check your internet connection.
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload MRI Image")
        uploaded_file = st.file_uploader(
            "Choose an MRI brain scan",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        st.header("ℹ️ About")
        st.markdown("""
        This AI tool classifies MRI brain scans into four Alzheimer's disease stages:
        
        - **NonDemented**: No signs detected
        - **VeryMildDemented**: Very mild decline  
        - **MildDemented**: Early stage
        - **ModerateDemented**: Middle stage
        
        **Note**: For research/educational purposes only.
        """)
        
        # Model download info
        if not os.path.exists('mobilenetv2_model.pth'):
            st.markdown("""
            **🔧 First-time Setup:**
            The model will download automatically when you first use the app.
            This may take 1-2 minutes.
            """)
        else:
            st.success("✅ Model loaded and ready!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if uploaded_file is not None:
            # Display uploaded image
            try:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
                
                # Analyze button
                if st.button("🔍 Analyze MRI Scan", type="primary", use_container_width=True):
                    with st.spinner("🔄 Analyzing the MRI scan..."):
                        predicted_idx, confidence, all_probabilities = predict_image(image, model)
                    
                    predicted_class = class_names[predicted_idx]
                    
                    # Display results in a nice card
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    
                    # Result header based on confidence
                    if confidence > 0.8:
                        st.success(f"## 🎯 Prediction: {predicted_class}")
                    elif confidence > 0.6:
                        st.warning(f"## 📊 Prediction: {predicted_class}")
                    else:
                        st.info(f"## 🤔 Prediction: {predicted_class}")
                    
                    st.metric("Confidence Score", f"{confidence:.2%}")
                    
                    # Description
                    st.write("**Description:**")
                    st.info(class_descriptions.get(predicted_class, "No description available"))
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Probability distribution
                    st.subheader("📈 Probability Distribution")
                    fig = plot_probabilities(all_probabilities, class_names)
                    st.pyplot(fig)
                    
                    # Detailed probabilities
                    st.subheader("🔢 Detailed Probabilities")
                    for i, (class_name, prob) in enumerate(zip(class_names, all_probabilities)):
                        col_prog, col_text = st.columns([3, 1])
                        with col_prog:
                            st.progress(float(prob), text=f"{class_name}")
                        with col_text:
                            st.write(f"{prob:.2%}")
            
            except Exception as e:
                st.error(f"Error processing image: {e}")
        
        else:
            # Welcome message when no image uploaded
            st.info("👆 Please upload an MRI brain scan image to get started")
            
            # Sample layout expectation
            st.subheader("🖼️ Image Guidelines")
            st.markdown("""
            For accurate results, ensure your MRI images:
            - Are **clear** and **well-focused**
            - Show **brain anatomy** clearly
            - Have **good contrast**
            - Are in **JPG, PNG, or BMP** format
            - Minimum size: **224×224 pixels**
            
            **Example MRI slices are ideal for classification.**
            """)
    
    with col2:
        if uploaded_file is not None:
            # Additional information and interpretation guide
            st.subheader("📋 Interpretation Guide")
            
            # Confidence level interpretation
            st.write("**Confidence Levels:**")
            confidence_info = {
                "🟢 High (80-100%)": "Strong indication of predicted stage",
                "🟡 Medium (60-80%)": "Moderate indication", 
                "🟠 Low (40-60%)": "Weak indication - consider re-evaluation",
                "🔴 Very Low (<40%)": "Uncertain prediction"
            }
            
            for level, desc in confidence_info.items():
                st.write(f"- {level}: {desc}")
            
            # Medical context
            st.subheader("🏥 Clinical Context")
            st.markdown("""
            **Alzheimer's Disease Progression:**
            
            1. **NonDemented**: Normal aging, no significant cognitive decline
            2. **VeryMildDemented**: Minor memory lapses, often unnoticed
            3. **MildDemented**: Noticeable memory loss, difficulty with complex tasks  
            4. **ModerateDemented**: Significant memory loss, requires assistance
            
            **Early detection can help manage symptoms and plan care.**
            """)
            
        else:
            # Model information
            st.subheader("🔬 Model Information")
            st.markdown("""
            **Technical Details:**
            - **Architecture**: MobileNetV2
            - **Training Data**: 6,400 MRI images
            - **Classes**: 4 Alzheimer's stages
            - **Input Size**: 224×224 pixels
            - **Framework**: PyTorch
            
            **Dataset Composition:**
            - NonDemented: 3,200 images
            - VeryMildDemented: 2,240 images  
            - MildDemented: 896 images
            - ModerateDemented: 64 images
            """)
            
            # Statistics from your dataset
            st.subheader("📊 Dataset Statistics")
            class_distribution = {
                'NonDemented': 3200,
                'VeryMildDemented': 2240, 
                'MildDemented': 896,
                'ModerateDemented': 64
            }
            
            st.bar_chart(class_distribution)
            
            # Setup instructions
            if os.path.exists('mobilenetv2_model.pth'):
                file_size = os.path.getsize('mobilenetv2_model.pth') / (1024 * 1024)
                st.success(f"✅ Model ready! ({file_size:.1f} MB)")
            else:
                st.info("🚀 Ready for first use - model will download automatically")
    
    # Disclaimer (always visible)
    st.markdown("""
    <div class="disclaimer">
    <h4>⚠️ Important Medical Disclaimer</h4>
    <p>This tool is for <strong>educational and research purposes only</strong>. It is <strong>NOT a medical device</strong> and should <strong>NOT</strong> be used for clinical diagnosis.</p>
    <p>Always consult qualified healthcare professionals for medical advice and diagnosis. Do not make healthcare decisions based solely on this tool's output.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
