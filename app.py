import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow.keras.models as models

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Eye Disease AI (Champion Model)",
    page_icon="👁️",
    layout="wide"
)

# Constants
IMG_SIZE = 224
CLASS_NAMES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
MODEL_PATH = 'xception_final_boost.h5' # The 88.85% Champion Model

# ==========================================
# CORE LOGIC
# ==========================================

@st.cache_resource
def load_model():
    """Load the Champion Xception Model."""
    try:
        # Xception (H5 format) is robust and loads reliably
        model = models.load_model(MODEL_PATH, compile=False)
        return model, None
    except Exception as e:
        return None, str(e)

def predict(model, image):
    """
    Run inference using the Xception model.
    """
    # Preprocess for Xception
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized)
    
    # Normalize: Xception model HAS internal Rescaling layer
    # So we pass raw [0-255] values.
    # img_array = img_array / 127.5 - 1  <-- REMOVED (Double Normalization)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array, verbose=0)[0]
    return preds

# ==========================================
# UI LAYOUT
# ==========================================
st.title("🔬 Eye Disease Classification")

# Sidebar for controls
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.info("Upload a retinal fundus image to detect diseases.")
    st.markdown("---")
    st.markdown("**Status:** Local Mode (Xception)")

# Load Model
model, error = load_model()

if model is None:
    st.error(f"❌ Error: Could not load '{MODEL_PATH}'.")
    if error:
        st.error(f"Details: {error}")
    st.stop()
else:
    st.success("✅ System Ready: Champion Model Active")

# Main Content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Patient Scan', use_column_width=True)

with col2:
    st.subheader("2. Analysis Results")
    
    if uploaded_file is not None:
        if st.button("Run Diagnostics", type="primary"):
            with st.spinner("Analyzing retina patterns..."):
                preds = predict(model, image)
                
                # Get Top Prediction
                idx = np.argmax(preds)
                diagnosis = CLASS_NAMES[idx]
                confidence = preds[idx] * 100
                
                # Display
                color = "green" if diagnosis == 'normal' else "red"
                st.markdown(f"### Diagnosis: :{color}[{diagnosis.upper().replace('_', ' ')}]")
                st.metric("Confidence Score", f"{confidence:.2f}%")
                
                # Chart
                st.write("---")
                st.write("**Probability Distribution:**")
                for i, cls in enumerate(CLASS_NAMES):
                    st.progress(float(preds[i]), text=f"{cls.title()} ({preds[i]*100:.1f}%)")
                    
                st.info(f"Model used: Xception (Best Performing Single Model)")
