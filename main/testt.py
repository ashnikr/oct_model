import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import onnxruntime as ort

# ---------------- CONFIG ----------------
class Config:
    IMG_SIZE = (224, 224)
    CLASS_NAMES = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']
    MODEL_PATH = r"model/ensemble_model.onnx"
    CLASS_DESCRIPTIONS = {
        'AMD': 'Age-related Macular Degeneration: affects central vision.',
        'CNV': 'Choroidal Neovascularization: abnormal blood vessel growth under retina.',
        'CSR': 'Central Serous Retinopathy: fluid buildup under retina.',
        'DME': 'Diabetic Macular Edema: swelling in the retina due to diabetes.',
        'DR': 'Diabetic Retinopathy: retinal blood vessel damage from diabetes.',
        'DRUSEN': 'Drusen: yellow deposits under retina, often in AMD.',
        'MH': 'Macular Hole: a small break in the central part of the retina.',
        'NORMAL': 'No apparent abnormalities detected.'
    }
    SEVERITY_MAPPING = {
        'NORMAL': 'Low', 'DRUSEN': 'Low', 'CSR': 'Medium', 'MH': 'Medium',
        'AMD': 'High', 'CNV': 'High', 'DME': 'High', 'DR': 'High'
    }

# -------------- MODEL LOADER --------------
class OCTModel:
    def __init__(self, path):  # ✅ Fixed constructor
        self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image):
        out = self.session.run([self.output_name], {self.input_name: image})
        return out[0][0]

# -------------- IMAGE PROCESSING --------------
def preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(Config.IMG_SIZE)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img

# -------------- REPORT GENERATOR --------------
def generate_report(name, diagnosis, confidence, prob_dict):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    risk = Config.SEVERITY_MAPPING[diagnosis]
    desc = Config.CLASS_DESCRIPTIONS[diagnosis]
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

    report = f"""
OCT EYE SCAN REPORT
====================

Date: {ts}
Patient Image: {name}

Diagnosis: {diagnosis}
AI Confidence: {confidence:.2%}
Risk Level: {risk}

Description:
{desc}

Other Probabilities:
---------------------
"""
    for cls, prob in sorted_probs:
        report += f"{cls}: {prob:.2%}\n"
    return report

# -------------- UI SETUP --------------
st.set_page_config(page_title="OCT Scan Diagnosis", layout="centered")

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
    }
    .sub-text {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🩺 OCT Eye Scan Diagnosis</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Upload an OCT image to get a diagnosis with confidence and risk level.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload OCT Image", type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Scan"):
        with st.spinner("Analyzing scan..."):
            img_batch, processed = preprocess_image(image)
            model = OCTModel(Config.MODEL_PATH)  # ✅ Now works
            preds = model.predict(img_batch)

            idx = np.argmax(preds)
            diagnosis = Config.CLASS_NAMES[idx]
            confidence = preds[idx]
            risk = Config.SEVERITY_MAPPING[diagnosis]

            st.success(f"✅ Diagnosis: *{diagnosis}*")
            st.info(f"📊 Confidence: *{confidence:.2%}*")
            st.warning(f"⚠ Risk Level: *{risk}*")
            st.markdown(f"*Description:* {Config.CLASS_DESCRIPTIONS[diagnosis]}")

            # Probabilities
            prob_dict = {Config.CLASS_NAMES[i]: preds[i] for i in range(len(preds))}

            # Report
            report = generate_report(uploaded_file.name, diagnosis, confidence, prob_dict)
            st.download_button("📄 Download Report", report, file_name="oct_scan_report.txt")

# Footer
st.markdown("""
---
⚠ This is an AI-based tool intended for research and support. Always consult a specialist for medical decisions.
""")
