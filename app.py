import io
import tempfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from inference_sdk import InferenceHTTPClient
import streamlit as st
import smtplib
from email.mime.text import MIMEText
import streamlit.components.v1 as components
import qrcode

st.set_page_config(
    page_title="Hard Hat & Safety Detector", page_icon="👷", layout="wide"
)

st.title("👷 Site Safety & Hard Hat Detector")


def play_offline_siren():
  siren_js = """
    <div style="background-color: #ff4b4b; color: white; padding: 12px; border-radius: 8px; text-align: center; font-family: sans-serif; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h4 style="margin: 0 0 4px 0;">🚨 CONTINUOUS SAFETY ALARM ACTIVE 🚨</h4>
        <p style="margin: 0 0 10px 0; font-size: 13px;">Hard hat violation detected! Repeating siren and voice warning...</p>
        <button onclick="stopAlarm()" style="background-color: white; color: #ff4b4b; border: none; padding: 8px 20px; font-weight: bold; border-radius: 4px; cursor: pointer; font-size: 14px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">🛑 Stop Alarm</button>
    </div>
    <script>
    let isAlarmActive = true;
    let audioCtx = null;

    function playSirenLoop() {
        if (!isAlarmActive) return;
        try {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                audioCtx = new AudioCtx();
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.type = 'sawtooth';
                const now = audioCtx.currentTime;
                osc.frequency.setValueAtTime(880, now);
                osc.frequency.setValueAtTime(440, now + 0.25);
                osc.frequency.setValueAtTime(880, now + 0.50);
                osc.frequency.setValueAtTime(440, now + 0.75);
                gain.gain.setValueAtTime(0.5, now);
                gain.gain.exponentialRampToValueAtTime(0.01, now + 1.0);
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start(now);
                osc.stop(now + 1.0);
            }
        } catch (e) {
            console.log("Audio blocked:", e);
        }
        
        if (isAlarmActive) {
            setTimeout(playSirenLoop, 1500);
        }
    }

    function speakLoop() {
        if (!isAlarmActive) return;
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance("Warning! Hard hat violation detected!");
            utterance.rate = 1.1;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            utterance.onend = function() {
                if (isAlarmActive) {
                    setTimeout(speakLoop, 500);
                }
            };
            window.speechSynthesis.speak(utterance);
        }
    }

    function stopAlarm() {
        isAlarmActive = false;
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
        if (audioCtx && audioCtx.state !== 'closed') {
            audioCtx.close();
        }
        const container = document.currentScript ? document.currentScript.parentElement : document.body;
        const alertBox = container.querySelector('div') || container;
        alertBox.style.display = 'none';
    }

    playSirenLoop();
    speakLoop();
    </script>
    """
  components.html(siren_js, height=130)


# Class color mapping (RGB format)
CLASS_COLORS = {
    "WEARING HARD-HAT": (0, 255, 0),  # Green
    "NO HARD-HAT": (255, 0, 0),  # Red
    "NOT WEARING HARD-HAT": (255, 0, 0),
    "not wearing hard-hat": (255, 0, 0),
    "wearing hard-hat": (0, 255, 0),
    "no hard-hat": (255, 0, 0),
    "helmet": (0, 255, 0),
    "no-helmet": (255, 0, 0),
}

# Sidebar controls
st.sidebar.header("⚙️ Detection Settings")
app_mode = st.sidebar.radio(
    "Select Operation Mode", ["Image / Snapshot", "Video File"]
)
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.1, max_value=1.0, value=0.4, step=0.05
)

# Sidebar QR Code for Mobile Access
st.sidebar.markdown("---")
st.sidebar.header("📱 Scan to Mobile")
app_url = "https://iyu76icob8pbcc5jpfrz95.streamlit.app/"
qr_img = qrcode.make(app_url)
st.sidebar.image(
    qr_img.get_image(), caption="Scan to open on mobile", use_container_width=True
)

if app_mode == "Image / Snapshot":
  camera_file = st.camera_input("Take a live photo")
  uploaded_file = st.file_uploader(
      "Or upload an image file", type=["jpg", "jpeg", "png"]
  )

  selected_file = camera_file if camera_file is not None else uploaded_file

  if selected_file is not None:
    image = Image.open(selected_file)

    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com", api_key="Rhe3HdHgQKYFx7aatoOx"
    )

    try:
      result = client.infer(image, model_id="hard-hat-detector-l0uba/4")
    except Exception as e:
      st.error(
          f"Error connecting to Roboflow API. Please check your internet or API"
          f" key. Details: {e}"
      )
      st.stop()

    predictions = result.get("predictions", [])
    img_np = np.array(image)
    img_h, img_w = img_np.shape[:2]

    box_thickness = max(1, int(img_w / 600))
    font_scale = max(0.3, img_w / 1500)
    font_thickness = max(1, int(img_w / 1000))

    initial_predictions = [
        p
        for p in predictions
        if isinstance(p, dict) and p.get("confidence", 0) >= conf_threshold
    ]

    boxes_for_nms = []
    scores_for_nms = []
    for p in initial_predictions:
      x, y, w, h = p["x"], p["y"], p["width"], p["height"]
      boxes_for_nms.append([int(x - w / 2), int(y - h / 2), int(w), int(h)])
      scores_for_nms.append(float(p["confidence"]))

    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms, scores_for_nms, conf_threshold, 0.4
    )
    filtered_predictions = (
        [initial_predictions[i] for i in np.array(indices).flatten()]
        if len(indices) > 0
        else []
    )

    class_counts = {}
    violations = 0
    table_data = []

    for idx, pred in enumerate(filtered_predictions):
      x, y, w, h = (
          int(pred["x"]),
          int(pred["y"]),
          int(pred["width"]),
          int(pred["height"]),
      )
      x1, y1 = int(x - w / 2), int(y - h / 2)
      x2, y2 = int(x + w / 2), int(y + h / 2)

      cls_name = pred["class"]
      box_color = CLASS_COLORS.get(cls_name, (255, 255, 0))
      class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

      cls_lower = cls_name.lower()
      if "no" in cls_lower or "not" in cls_lower:
        violations += 1

      table_data.append({
          "ID": idx + 1,
          "Class": cls_name,
          "Confidence": f"{pred['confidence'] * 100:.1f}%",
          "Bounding Box": f"[{x1}, {y1}, {x2}, {y2}]",
      })

      label = f"{cls_name} ({pred['confidence']:.2f})"
      cv2.rectangle(img_np, (x1, y1), (x2, y2), box_color, box_thickness)
      (text_w, text_h), baseline = cv2.getTextSize(
          label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
      )
      bg_y1 = max(0, y1 - text_h - 6)
      bg_y2 = y1
      cv2.rectangle(
          img_np, (x1, bg_y1), (x1 + text_w + 4, bg_y2), box_color, -1
      )
      cv2.putText(
          img_np,
          label,
          (x1 + 2, bg_y2 - 2),
          cv2.FONT_HERSHEY_SIMPLEX,
          font_scale,
          (0, 0, 0),
          font_thickness,
      )

    if violations > 0:
      st.error(
          f"🚨 **SAFETY VIOLATION ALERT:** Detected {violations} person(s) without"
          " a hard hat!"
      )
      play_offline_siren()

      if (
          "last_alert_sent" not in st.session_state
          or st.session_state["last_alert_sent"] != violations
      ):
        try:
          sender = st.secrets["email"]["sender"]
          password = st.secrets["email"]["password"]
          receiver = st.secrets["email"]["receiver"]

          msg = MIMEText(
              "🚨 AUTOMATED SITE SAFETY ALERT\n\nThe vision monitoring system"
              f" detected {violations} person(s) without required safety helmets"
              " on site.\n\nPlease log in to your dashboard to review the"
              " capture."
          )
          msg["Subject"] = "🚨 Urgent: Safety Violation Detected"
          msg["From"] = sender
          msg["To"] = receiver

          with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)

          st.success(
              "✉️ **Automated Alert Sent:** Notification email dispatched to site"
              " manager."
          )
          st.session_state["last_alert_sent"] = violations
        except Exception as e:
          st.error(
              "Failed to send email alert. Check Streamlit Secrets. Error:"
              f" {e}"
          )

    elif len(filtered_predictions) > 0:
      st.success(
          "✅ **ALL COMPLIANT:** All detected personnel are wearing hard hats."
      )
      st.session_state["last_alert_sent"] = 0

    st.image(img_np, caption="Processed Image", use_container_width=True)

    st.markdown("### 📊 Safety Analytics Summary")
    total_detected = len(filtered_predictions)
    if total_detected > 0:
      compliance_rate = (
          (total_detected - violations) / total_detected
      ) * 100
      m1, m2, m3 = st.columns(3)
      m1.metric("Total Personnel", total_detected)
      m2.metric("Safety Violations", violations, delta_color="inverse")
      m3.metric("Compliance Rate", f"{compliance_rate:.1f}%")

    st.markdown("### 📋 Detection Logs")
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

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
  else:
    st.info("No photo captured or uploaded yet. Take or upload a photo above.")

elif app_mode == "Video File":
  st.markdown("### 🎥 Video Stream Compliance Analysis")
  video_file = st.file_uploader(
      "Upload a construction site video", type=["mp4", "avi", "mov"]
  )

  if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    st_frame = st.empty()

    if st.button("▶️ Start Processing Video"):
      client = InferenceHTTPClient(
          api_url="https://detect.roboflow.com", api_key="Rhe3HdHgQKYFx7aatoOx"
      )

      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        img_h, img_w = rgb_frame.shape[:2]

        box_thickness = max(1, int(img_w / 600))
        font_scale = max(0.3, img_w / 1500)
        font_thickness = max(1, int(img_w / 1000))

        try:
          result = client.infer(pil_img, model_id="hard-hat-detector-l0uba/4")
          predictions = result.get("predictions", [])

          initial_predictions = [
              p
              for p in predictions
              if isinstance(p, dict) and p.get("confidence", 0) >= conf_threshold
          ]

          boxes_for_nms = []
          scores_for_nms = []
          for p in initial_predictions:
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            boxes_for_nms.append([int(x - w / 2), int(y - h / 2), int(w), int(h)])
            scores_for_nms.append(float(p["confidence"]))

          indices = cv2.dnn.NMSBoxes(
              boxes_for_nms, scores_for_nms, conf_threshold, 0.4
          )
          filtered_predictions = (
              [initial_predictions[i] for i in np.array(indices).flatten()]
              if len(indices) > 0
              else []
          )

          frame_violations = 0
          for pred in filtered_predictions:
            x, y, w, h = (
                int(pred["x"]),
                int(pred["y"]),
                int(pred["width"]),
                int(pred["height"]),
            )
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            cls_name = pred["class"]
            box_color = CLASS_COLORS.get(cls_name, (255, 255, 0))

            cls_lower = cls_name.lower()
            if "no" in cls_lower or "not" in cls_lower:
              frame_violations += 1

            label = f"{cls_name} ({pred['confidence']:.2f})"
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), box_color, box_thickness)
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            bg_y1 = max(0, y1 - text_h - 6)
            bg_y2 = y1
            cv2.rectangle(
                rgb_frame, (x1, bg_y1), (x1 + text_w + 4, bg_y2), box_color, -1
            )
            cv2.putText(
                rgb_frame,
                label,
                (x1 + 2, bg_y2 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness,
            )

          if frame_violations > 0:
            play_offline_siren()

        except Exception as e:
          print("Inference error on frame:", e)

        st_frame.image(rgb_frame, channels="RGB", use_container_width=True)

      cap.release()
      st.success("✅ Video processing complete!")
  else:
    st.info("Please upload a video file above to begin video monitoring.")
