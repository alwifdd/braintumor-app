import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    model_effnet = tf.keras.models.load_model(
        "model/effnet_finetuned.h5", compile=False
    )
    model_effnet_base = tf.keras.models.load_model(
        "model/effnet_baseline.h5", compile=False
    )
    model_mobilenet = tf.keras.models.load_model(
        "model/mobilenet_finetuned.h5", compile=False
    )
    return model_effnet, model_effnet_base, model_mobilenet


model_effnet, model_effnet_base, model_mobilenet = load_models()

class_names = ["glioma", "meningioma", "pituitary", "notumor"]

# =========================
# PREPROCESS
# =========================
def preprocess(img, model_type):
    img = cv2.resize(img, (224, 224))

    if model_type == "effnet":
        img = img / 255.0
    else:
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    return np.expand_dims(img, axis=0)

# =========================
# GRAD-CAM
# =========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap.numpy()


def overlay_gradcam(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay

# =========================
# UI
# =========================
st.title("🧠 Brain Tumor Classification")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.image(img, caption="Input Image", use_container_width=True)

    model_choice = st.selectbox(
        "Choose Model",
        ["EfficientNet FT", "EfficientNet Baseline", "MobileNet"]
    )

    if st.button("Predict"):
        if model_choice == "EfficientNet FT":
            model = model_effnet
            model_type = "effnet"
            layer = "top_conv"

        elif model_choice == "EfficientNet Baseline":
            model = model_effnet_base
            model_type = "effnet"
            layer = "top_conv"

        else:
            model = model_mobilenet
            model_type = "mobilenet"
            layer = "Conv_1"

        img_input = preprocess(img, model_type)

        pred = model.predict(img_input)
        pred_class = np.argmax(pred)
        confidence = float(pred[0][pred_class])

        heatmap = make_gradcam_heatmap(img_input, model, layer)
        gradcam = overlay_gradcam(img, heatmap)

        st.subheader("📊 Result")
        st.image(gradcam, caption="Grad-CAM", use_container_width=True)

        st.success(f"Prediction: {class_names[pred_class]}")
        st.write(f"Confidence: {confidence:.2f}")

# =========================
# FOOTER
# =========================
st.warning("⚠️ Ini hanya alat bantu, bukan diagnosis medis.")