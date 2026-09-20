import cv2
import io
import numpy as np
import pandas as pd
from PIL import Image
from inference_sdk import InferenceHTTPClient
import streamlit as st

st.set_page_config(page_title="Hard Hat & Safety Detector", page_icon="👷", layout="wide")

st.title("👷 Site Safety & Hard Hat Detector")

# Class color mapping (BGR format)
CLASS_COLORS = {
    "WEARING HARD-HAT": (0, 255, 0),      # Green
    "NO HARD-HAT": (0, 0, 255),           # Red
    "wearing hard-hat": (0, 255, 0),
    "no hard-hat": (0, 0, 255),
    "helmet": (0, 255, 0),
    "no-helmet": (0, 0, 255),
}

# Sidebar controls
st.sidebar.header("⚙️ Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.4, step=0.05)

# Input methods
camera_file = st.camera_input("Take a live photo")
uploaded_file = st.file_uploader("Or upload an image file", type=["jpg", "jpeg", "png"])

selected_file = camera_file if camera_file is not None else uploaded_file

if selected_file is not None:
    image = Image.open(selected_file)

    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="Rhe3HdHgQKYFx7aatoOx"
    )

    # Execute workflow
    workflow_response = client.run_workflow(
        workspace_name="chinyama-chilila",
        workflow_id="custom-workflow-10",
        images={"image": image}
    )

    # Parse predictions
    predictions = []
    if isinstance(workflow_response, list) and len(workflow_response) > 0:
        res_dict = workflow_response[0]
        preds = res_dict.get("predictions", [])
        if isinstance(preds, dict) and "predictions" in preds:
            predictions = preds["predictions"]
        elif isinstance(preds, list):
            predictions = preds

    img_np = np.array(image)
    img_h, img_w = img_np.shape[:2]

    box_thickness = max(2, int(img_w / 400))
    font_scale = max(0.6, img_w / 800)
    font_thickness = max(2, int(img_w / 500))

    filtered_predictions = [p for p in predictions if isinstance(p, dict) and p.get("confidence", 0) >= conf_threshold]

    class_counts = {}
    violations = 0
    table_data = []

    for idx, pred in enumerate(filtered_predictions):
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        cls_name = pred["class"]
        box_color = CLASS_COLORS.get(cls_name, (255, 255, 0))
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        if "no" in cls_name.lower():
            violations += 1

        table_data.append({
            "ID": idx + 1,
            "Class": cls_name,
            "Confidence": f"{pred['confidence'] * 100:.1f}%",
            "Bounding Box": f"[{x1}, {y1}, {x2}, {y2}]"
        })

        cv2.rectangle(img_np, (x1, y1), (x2, y2), box_color, box_thickness)
        label = f"{cls_name} ({pred['confidence']:.2f})"

        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        bg_y1 = max(0, y1 - text_h - 10)
        bg_y2 = max(text_h + 10, y1)
        cv2.rectangle(img_np, (x1, bg_y1), (x1 + text_w + 10, bg_y2), box_color, -1)

        cv2.putText(
            img_np,
            label,
            (x1 + 5, bg_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )

    # Show safety alert banner
    if violations > 0:
        st.error(f"🚨 **SAFETY VIOLATION ALERT:** Detected {violations} person(s) without a hard hat!")
    elif len(filtered_predictions) > 0:
        st.success("✅ **ALL COMPLIANT:** All detected personnel are wearing hard hats.")

    st.image(img_np, caption="Processed Image", use_container_width=True)

    # Safety Metrics Summary
    st.markdown("### 📊 Safety Analytics Summary")
    total_detected = len(filtered_predictions)
    
    if total_detected > 0:
        compliance_rate = ((total_detected - violations) / total_detected) * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Personnel", total_detected)
        m2.metric("Safety Violations", violations, delta_color="inverse")
        m3.metric("Compliance Rate", f"{compliance_rate:.1f}%")

        # Table audit
        st.markdown("### 📋 Detection Logs")
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    else:
        st.info("No detections found above the selected confidence threshold.")

    # Download button
    result_img = Image.fromarray(img_np)
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download Labeled Image",
        data=byte_im,
        file_name="safety_audit_result.png",
        mime="image/png",
    )
