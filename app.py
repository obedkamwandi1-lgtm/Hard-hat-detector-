import cv2
import io
import numpy as np
import pandas as pd
from PIL import Image
from inference_sdk import InferenceHTTPClient
import streamlit as st
import smtplib
from email.mime.text import MIMEText
import streamlit.components.v1 as components

st.set_page_config(page_title="Hard Hat & Safety Detector", page_icon="👷", layout="wide")
st.title("👷 Site Safety & Hard Hat Detector")

def play_offline_siren():
    siren_js = """
    <script>
    (function() {
        try {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtx) return;
            const ctx = new AudioCtx();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sawtooth';
            const now = ctx.currentTime;
            osc.frequency.setValueAtTime(880, now);
            osc.frequency.setValueAtTime(440, now + 0.25);
            osc.frequency.setValueAtTime(880, now + 0.50);
            osc.frequency.setValueAtTime(440, now + 0.75);
            gain.gain.setValueAtTime(0.5, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 1.0);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(now);
            osc.stop(now + 1.0);
        } catch (e) {
            console.log("Audio playback blocked or unsupported:", e);
        }
    })();
    </script>
    """
    components.html(siren_js, height=0, width=0)

# Class color mapping (RGB format since Streamlit renders NumPy arrays in RGB)
CLASS_COLORS = {
    "WEARING HARD-HAT": (0, 255, 0),      # Green
    "NO HARD-HAT": (255, 0, 0),           # Red
    "NOT WEARING HARD-HAT": (255, 0, 0),  
    "not wearing hard-hat": (255, 0, 0),
    "wearing hard-hat": (0, 255, 0),
    "no hard-hat": (255, 0, 0),
    "helmet": (0, 255, 0),
    "no-helmet": (255, 0, 0),
}

# Sidebar controls
st.sidebar.header("⚙️ Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.4, step=0.05)

# Input methods
camera_file = st.camera_input("Take a live photo")
uploaded_file = st.file_uploader("Or upload an image file", type=["jpg", "jpeg", "png"])

selected_file = camera_file if camera_file is not None else uploaded_file

if selected_file is not None:
    # Load image
    image = Image.open(selected_file)

    # Initialize Roboflow Client
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="Rhe3HdHgQKYFx7aatoOx"
    )

    try:
        # Directly run model inference on the image
        result = client.infer(image, model_id="hard-hat-detector-l0uba/4")
    except Exception as e:
        st.error(f"Error connecting to Roboflow API. Please check your internet or API key. Details: {e}")
        st.stop()

    # Extract predictions list
    predictions = result.get("predictions", [])
    img_np = np.array(image)
    img_h, img_w = img_np.shape[:2]

    # Dynamic styling sizes (Scaled down to prevent overlap on group photos)
    box_thickness = max(1, int(img_w / 600))
    font_scale = max(0.3, img_w / 1500)
    font_thickness = max(1, int(img_w / 1000))

    # 1. Filter initial predictions based on slider confidence threshold
    initial_predictions = [p for p in predictions if isinstance(p, dict) and p.get("confidence", 0) >= conf_threshold]

    # 2. Apply Non-Maximum Suppression (NMS) to remove overlapping double-detections
    boxes_for_nms = []
    scores_for_nms = []
    
    for p in initial_predictions:
        # Convert Roboflow center x,y coordinates to OpenCV top-left x,y for NMS
        x, y, w, h = p["x"], p["y"], p["width"], p["height"]
        boxes_for_nms.append([int(x - w / 2), int(y - h / 2), int(w), int(h)])
        scores_for_nms.append(float(p["confidence"]))

    # 0.4 is the overlap threshold (If two boxes overlap by 40%+, delete the weaker one)
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_for_nms, conf_threshold, 0.4)
    
    # Create the final list of clean predictions
    filtered_predictions = [initial_predictions[i] for i in np.array(indices).flatten()] if len(indices) > 0 else []

    class_counts = {}
    violations = 0
    table_data = []

    # Create shorter display labels to reduce visual clutter
    short_labels = {
        "WEARING HARD-HAT": "SAFE",
        "NOT WEARING HARD-HAT": "NO HAT",
        "wearing hard-hat": "SAFE",
        "no hard-hat": "NO HAT",
        "NO HARD-HAT": "NO HAT",
        "helmet": "SAFE",
        "no-helmet": "NO HAT"
    }

    for idx, pred in enumerate(filtered_predictions):
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        
        cls_name = pred["class"]
        box_color = CLASS_COLORS.get(cls_name, (255, 255, 0)) # Default to yellow if class is unlisted
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        # Check if the class name contains "no" or "not" to correctly trigger violations
        cls_lower = cls_name.lower()
        if "no" in cls_lower or "not" in cls_lower:
            violations += 1
            
        table_data.append({
            "ID": idx + 1, 
            "Class": cls_name, 
            "Confidence": f"{pred['confidence'] * 100:.1f}%", 
            "Bounding Box": f"[{x1}, {y1}, {x2}, {y2}]"
        })
        
        display_name = short_labels.get(cls_name, cls_name)
        label = f"{display_name} ({pred['confidence']:.2f})"
        
        # Draw bounding box for each individual worker
        cv2.rectangle(img_np, (x1, y1), (x2, y2), box_color, box_thickness)
        
        # Draw tighter text background badge
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        bg_y1 = max(0, y1 - text_h - 6) # Reduced padding
        bg_y2 = y1 # Snapped exactly to the top of the bounding box
        cv2.rectangle(img_np, (x1, bg_y1), (x1 + text_w + 4, bg_y2), box_color, -1)
        
        # Draw text inside badge with tighter margins
        cv2.putText(img_np, label, (x1 + 2, bg_y2 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

    # Real-time safety alert banner + Automatic Email Dispatch
    if violations > 0:
        st.error(f"🚨 **SAFETY VIOLATION ALERT:** Detected {violations} person(s) without a hard hat!")
        play_offline_siren()
        
        # Check session state to prevent duplicate emails for the same detection
        if "last_alert_sent" not in st.session_state or st.session_state["last_alert_sent"] != violations:
            try:
                # Retrieve credentials from Streamlit Secrets
                sender = st.secrets["email"]["sender"]
                password = st.secrets["email"]["password"]
                receiver = st.secrets["email"]["receiver"]
                
                # Compose the email message
                msg = MIMEText(
                    f"🚨 AUTOMATED SITE SAFETY ALERT\n\n"
                    f"The vision monitoring system detected {violations} person(s) without required safety helmets on site.\n\n"
                    f"Please log in to your dashboard to review the capture."
                )
                msg['Subject'] = '🚨 Urgent: Safety Violation Detected'
                msg['From'] = sender
                msg['To'] = receiver
                
                # Send email via Gmail's SSL server
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                    server.login(sender, password)
                    server.send_message(msg)
                
                st.success("✉️ **Automated Alert Sent:** Notification email dispatched to site manager.")
                st.session_state["last_alert_sent"] = violations
            except Exception as e:
                st.error(f"Failed to send email alert. Check Streamlit Secrets. Error: {e}")
                
    elif len(filtered_predictions) > 0:
        st.success("✅ **ALL COMPLIANT:** All detected personnel are wearing hard hats.")
        st.session_state["last_alert_sent"] = 0

    # Display image with bounding boxes
    st.image(img_np, caption="Processed Image", use_container_width=True)

    # Display Safety Analytics Dashboard
    st.markdown("### 📊 Safety Analytics Summary")
    total_detected = len(filtered_predictions)
    
    if total_detected > 0:
        compliance_rate = ((total_detected - violations) / total_detected) * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Personnel", total_detected)
        m2.metric("Safety Violations", violations, delta_color="inverse")
        m3.metric("Compliance Rate", f"{compliance_rate:.1f}%")
        
        # Table audit log
        st.markdown("### 📋 Detection Logs")
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    else:
        st.info("No detections found above the selected confidence threshold.")

    # Prepare download button
    result_img = Image.fromarray(img_np)
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    st.download_button(
        label="📥 Download Labeled Image",
        data=byte_im,
        file_name="safety_audit_result.png",
        mime="image/png"
    )
