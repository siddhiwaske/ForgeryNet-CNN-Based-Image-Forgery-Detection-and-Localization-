import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from efficientnet_pytorch import EfficientNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

# App title
st.set_page_config(page_title="ForgeryNet", layout="wide")
st.title("🔍 ForgeryNet: Image Forgery Detection & Localization")
st.markdown("Upload an image to detect tampering and view localization results.")

# Load model once
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNet.from_pretrained('efficientnet-b3')
    model._fc = nn.Linear(model._fc.in_features, 2)
    model.load_state_dict(torch.load("efficientnet_b3_forgerynet.pth", map_location=device))
    model.to(device).eval()
    return model, device
model, device = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and prepare image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224,224)) / 255.0
    input_tensor = torch.tensor(img_resized).permute(2,0,1).unsqueeze(0).float().to(device)

    # Grad-CAM
    target_layer = model._blocks[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    heatmap = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

    # Outline mask
    mask_pred = (grayscale_cam > 0.45).astype(np.uint8) * 255
    mask_pred = cv2.resize(mask_pred, (img_rgb.shape[1], img_rgb.shape[0]))
    contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined = img_rgb.copy()
    cv2.drawContours(outlined, contours, -1, (0,255,0), 3)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)  # 0 = fake, 1 = real
        pred_class = "FAKE" if pred_idx == 0 else "REAL"
        confidence = probs[pred_idx]

    # Layout
    col1, col2, col3, col4 = st.columns(4)
    col1.image(img_rgb, caption="Original Image", use_container_width=True)
    col2.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True)
    col3.image(outlined, caption="Tampered Region (Model)", use_container_width=True)

    # Show ground truth only if predicted class = FAKE
    if pred_class == "FAKE":
        GROUNDTRUTH_DIR = r"C:\Users\hp\Downloads\ForgeryNet\CASIA 2 Groundtruth"
        file_name = os.path.basename(uploaded_file.name)
        base_name = os.path.splitext(file_name)[0]
        mask_filename = base_name + "_gt.png"
        mask_path = os.path.join(GROUNDTRUTH_DIR, mask_filename)

        if os.path.exists(mask_path):
            gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            col4.image(gt, caption="Ground Truth Mask", use_container_width=True)
        else:
            col4.error("❌ Ground Truth Mask Not Found")
    else:
        col4.info("✅ No forgery detected — ground truth mask not applicable.")

    st.markdown("---")
    st.markdown(f"### 🧠 Prediction: **{pred_class}** (Confidence: {confidence:.2f})")


