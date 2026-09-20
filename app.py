import cv2
import numpy as np
from PIL import Image
from inference_sdk import InferenceHTTPClient
import streamlit as st

st.title("👷 Hard Hat Detection App")

# Allow live camera capture or file upload
camera_file = st.camera_input("Take a live photo")
uploaded_file = st.file_uploader("Or upload an image file", type=["jpg", "jpeg", "png"])

# Select whichever image source is active
selected_file = camera_file if camera_file is not None else uploaded_file

if selected_file is not None:
    # Load image
    image = Image.open(selected_file)

    # Run Roboflow model
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="Rhe3HdHgQKYFx7aatoOx"
    )

    result = client.infer(image, model_id="hard-hat-detector-l0uba/4")

    # Draw bounding boxes
    img_np = np.array(image)
    img_h, img_w = img_np.shape[:2]

    # Calculate dynamic sizes based on image resolution
    box_thickness = max(2, int(img_w / 400))
    font_scale = max(0.6, img_w / 800)
    font_thickness = max(2, int(img_w / 500))

    predictions = result.get("predictions", [])

    for pred in predictions:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        # Draw bounding box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)

        # Format label text
        label = f"{pred['class']} ({pred['confidence']:.2f})"

        # Draw filled background badge for text
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        bg_y1 = max(0, y1 - text_h - 10)
        bg_y2 = max(text_h + 10, y1)
        cv2.rectangle(img_np, (x1, bg_y1), (x1 + text_w + 10, bg_y2), (0, 255, 0), -1)

        # Draw text inside the badge
        cv2.putText(
            img_np,
            label,
            (x1 + 5, bg_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )

    # Display result
  # Display result
st.image(img_np, caption="Detections", use_container_width=True)
st.success(f"Found {len(predictions)} objects!")
