import streamlit as st
import cv2
import numpy as np
from PIL import Image
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

st.title("👷 Hard Hat Detection App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load uploaded image
    image = Image.open(uploaded_file)
    image.save("temp_input.jpg")

    # Run Roboflow model
  # Run Roboflow model
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="Rhe3HdHgQKYFx7aatoOx"
    )

    result = client.infer(image, model_id="hard-hat-detector-l0uba/4")

    # Draw bounding boxes
    img_np = np.array(image)
    predictions = result.get("predictions", [])

    for pred in predictions:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_np, 
            f"{pred['class']} ({pred['confidence']:.2f})", 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (0, 255, 0), 2
        )

    # Display result
    st.image(img_np, caption="Detections", use_container_width=True)
    st.success(f"Found {len(predictions)} objects!")
