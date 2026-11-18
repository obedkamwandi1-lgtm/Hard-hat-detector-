import io
import os
import tempfile
import time
from datetime import datetime
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
from streamlit_js_eval import get_geolocation

# Ensure archive directory exists
os.makedirs("archive", exist_ok=True)

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


def render_google_map(lat: float, lon: float, zone_label: str):
    maps_link = f"https://www.google.com/maps?q={lat},{lon}"
    embed_url = f"https://maps.google.com/maps?q={lat},{lon}&z=16&output=embed"
    st.markdown(f"### 📍 Incident GPS Location — {zone_label}")
    st.markdown(
        f"**Coordinates:** `{lat:.6f}, {lon:.6f}` | 🔗 **[Open Exact Pin in"
        f" Google Maps]({maps_link})**"
    )
    components.iframe(embed_url, height=350)


def render_realtime_webcam_monitor(conf_threshold: float, zone_name: str, lat: float, lon: float):
    """Hands-free HTML5 real-time camera monitor with direct Roboflow inference & live alarm."""
    html_code = f"""
    <div style="font-family: sans-serif; color: #fafafa; background: #1e2129; padding: 16px; border-radius: 10px; border: 1px solid #31333f;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 10px;">
            <div>
                <span id="status-badge" style="background: #2e7d32; color: white; padding: 5px 12px; border-radius: 20px; font-weight: bold; font-size: 13px;">
                    🟢 LIVE CAMERA STARTING...
                </span>
                <span style="margin-left: 10px; font-size: 13px; color: #b0b3b8;">
                    Zone: <b>{zone_name}</b> ({lat:.5f}, {lon:.5f})
                </span>
            </div>
            <div>
                <button id="toggle-btn" onclick="toggleCamera()" style="background: #ff4b4b; color: white; border: none; padding: 8px 16px; border-radius: 6px; font-weight: bold; cursor: pointer;">
                    ⏹ Stop Monitoring
                </button>
            </div>
        </div>

        <div id="alarm-banner" style="display: none; background: #ff4b4b; color: white; padding: 12px; border-radius: 8px; text-align: center; margin-bottom: 12px;">
            <h4 style="margin: 0 0 4px 0;">🚨 CONTINUOUS SAFETY VIOLATION DETECTED! 🚨</h4>
            <p id="alarm-details" style="margin: 0 0 8px 0; font-size: 13px;">Worker detected without a hard hat!</p>
            <button onclick="muteAlarm()" style="background: white; color: #ff4b4b; border: none; padding: 6px 16px; border-radius: 4px; font-weight: bold; cursor: pointer;">
                🔇 Mute Siren
            </button>
        </div>

        <div style="position: relative; width: 100%; max-width: 720px; margin: 0 auto; border-radius: 8px; overflow: hidden; background: #000;">
            <video id="webcam" autoplay playsinline muted style="width: 100%; display: block;"></video>
            <canvas id="overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;"></canvas>
        </div>

        <div style="display: flex; justify-content: space-around; margin-top: 14px; background: #262730; padding: 10px; border-radius: 8px; text-align: center;">
            <div>
                <div style="font-size: 12px; color: #a3a8b8;">TOTAL DETECTED</div>
                <div id="stat-total" style="font-size: 22px; font-weight: bold;">0</div>
            </div>
            <div>
                <div style="font-size: 12px; color: #a3a8b8;">VIOLATIONS</div>
                <div id="stat-viol" style="font-size: 22px; font-weight: bold; color: #ff4b4b;">0</div>
            </div>
            <div>
                <div style="font-size: 12px; color: #a3a8b8;">COMPLIANCE</div>
                <div id="stat-comp" style="font-size: 22px; font-weight: bold; color: #00e676;">100%</div>
            </div>
        </div>
    </div>

    <script>
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    const statusBadge = document.getElementById('status-badge');
    const alarmBanner = document.getElementById('alarm-banner');
    const alarmDetails = document.getElementById('alarm-details');

    const MODEL_ENDPOINT = "https://detect.roboflow.com/chinyama-chilila/hard-hat-detector-l0uba-9-rfdetr-small-t1?api_key=Rhe3HdHgQKYFx7aatoOx&confidence={int(conf_threshold * 100)}";
    let isRunning = true;
    let isBusy = false;
    let alarmMuted = false;
    let streamRef = null;
    let lastSpeakTime = 0;

    async function startWebcam() {{
        try {{
            streamRef = await navigator.mediaDevices.getUserMedia({{
                video: {{ facingMode: "environment", width: {{ ideal: 640 }}, height: {{ ideal: 480 }} }},
                audio: false
            }});
            video.srcObject = streamRef;
            statusBadge.innerText = "🟢 LIVE MONITORING ACTIVE";
            statusBadge.style.background = "#2e7d32";
            setInterval(processFrame, 1200);
        }} catch (err) {{
            statusBadge.innerText = "🔴 CAMERA ACCESS BLOCKED: " + err.message;
            statusBadge.style.background = "#c62828";
        }}
    }}

    function triggerSiren() {{
        if (alarmMuted) return;
        try {{
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {{
                const audioCtx = new AudioCtx();
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.type = 'sawtooth';
                const now = audioCtx.currentTime;
                osc.frequency.setValueAtTime(880, now);
                osc.frequency.setValueAtTime(440, now + 0.25);
                gain.gain.setValueAtTime(0.35, now);
                gain.gain.exponentialRampToValueAtTime(0.01, now + 0.5);
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start(now);
                osc.stop(now + 0.5);
            }}
        }} catch (e) {{}}

        const nowMs = Date.now();
        if ('speechSynthesis' in window && nowMs - lastSpeakTime > 4000) {{
            lastSpeakTime = nowMs;
            const u = new SpeechSynthesisUtterance("Warning! Hard hat violation detected at {zone_name}!");
            u.rate = 1.1;
            window.speechSynthesis.speak(u);
        }}
    }}

    function muteAlarm() {{
        alarmMuted = true;
        alarmBanner.style.display = "none";
        if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    }}

    function toggleCamera() {{
        const btn = document.getElementById('toggle-btn');
        if (isRunning) {{
            isRunning = false;
            if (streamRef) streamRef.getTracks().forEach(t => t.stop());
            statusBadge.innerText = "⏸ MONITORING PAUSED";
            statusBadge.style.background = "#616161";
            btn.innerText = "▶️ Resume Monitoring";
            btn.style.background = "#2e7d32";
        }} else {{
            isRunning = true;
            alarmMuted = false;
            btn.innerText = "⏹ Stop Monitoring";
            btn.style.background = "#ff4b4b";
            startWebcam();
        }}
    }}

    async function processFrame() {{
        if (!isRunning || isBusy || video.readyState < 2) return;
        isBusy = true;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const offscreen = document.createElement('canvas');
        offscreen.width = video.videoWidth;
        offscreen.height = video.videoHeight;
        offscreen.getContext('2d').drawImage(video, 0, 0);
        const base64Img = offscreen.toDataURL('image/jpeg', 0.7).split(',')[1];

        try {{
            const resp = await fetch(MODEL_ENDPOINT, {{
                method: "POST",
                headers: {{ "Content-Type": "application/x-www-form-urlencoded" }},
                body: base64Img
            }});
            const data = await resp.json();
            const preds = data.predictions || [];

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            let violations = 0;

            preds.forEach(p => {{
                const x1 = p.x - p.width / 2;
                const y1 = p.y - p.height / 2;
                const clsLower = p.class.toLowerCase();
                const isViol = clsLower.includes("no") || clsLower.includes("not");
                if (isViol) violations++;

                const color = isViol ? "#ff1744" : "#00e676";
                ctx.strokeStyle = color;
                ctx.lineWidth = 4;
                ctx.strokeRect(x1, y1, p.width, p.height);

                const label = `${{p.class}} (${{(p.confidence * 100).toFixed(0)}}%)`;
                ctx.font = "bold 15px sans-serif";
                const textW = ctx.measureText(label).width;
                ctx.fillStyle = color;
                ctx.fillRect(x1, Math.max(0, y1 - 24), textW + 10, 24);
                ctx.fillStyle = "#000000";
                ctx.fillText(label, x1 + 5, Math.max(17, y1 - 6));
            }});

            const total = preds.length;
            const comp = total > 0 ? Math.round(((total - violations) / total) * 100) : 100;
            document.getElementById('stat-total').innerText = total;
            document.getElementById('stat-viol').innerText = violations;
            document.getElementById('stat-comp').innerText = comp + "%";

            if (violations > 0) {{
                alarmBanner.style.display = "block";
                alarmDetails.innerText = `Detected ${{violations}} person(s) without a hard hat at {zone_name}!`;
                triggerSiren();
            }} else if (total > 0) {{
                alarmBanner.style.display = "none";
            }}
        }} catch (err) {{
            console.error("Inference frame error:", err);
        }} finally {{
            isBusy = false;
        }}
    }}

    startWebcam();
    </script>
    """
    components.html(html_code, height=680)


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


def process_and_display_image(
    image: Image.Image,
    conf_threshold: float,
    active_lat: float,
    active_lon: float,
    active_zone_name: str,
    key_prefix: str = "img",
):
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="Rhe3HdHgQKYFx7aatoOx",
    )

    try:
        result = client.infer(
            image,
            model_id="chinyama-chilila/hard-hat-detector-l0uba-9-rfdetr-small-t1",
        )
    except Exception as e:
        st.error(
            "Error connecting to Roboflow API. Please check your internet or API"
            f" key. Details: {e}"
        )
        return

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
            "Zone": active_zone_name,
            "GPS": f"{active_lat:.5f}, {active_lon:.5f}",
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
            f"🚨 **SAFETY VIOLATION ALERT:** Detected {violations} person(s)"
            f" without a hard hat at **{active_zone_name}**!"
        )
        play_offline_siren()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.join(
            "archive", f"violation_{timestamp}_{violations}viol.png"
        )
        Image.fromarray(img_np).save(archive_path)

        google_maps_link = (
            f"https://www.google.com/maps?q={active_lat},{active_lon}"
        )

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
                    f" detected {violations} person(s) without required safety"
                    f" helmets on site.\n\nZone: {active_zone_name}\nGPS"
                    f" Coordinates: {active_lat:.6f}, {active_lon:.6f}\nGoogle Maps"
                    f" Pin: {google_maps_link}\n\nCaptured image archived as:"
                    f" {archive_path}"
                )
                msg["Subject"] = f"🚨 Urgent: Safety Violation at {active_zone_name}"
                msg["From"] = sender
                msg["To"] = receiver

                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(sender, password)
                    server.send_message(msg)

                st.success(
                    "✉️ **Automated Alert Sent & Archived:** Email with Google Maps"
                    " pin dispatched and capture saved to audit trail."
                )
                st.session_state["last_alert_sent"] = violations
            except Exception as e:
                st.error(
                    "Failed to send email alert. Check Streamlit Secrets. Error:"
                    f" {e}"
                )

    elif len(filtered_predictions) > 0:
        st.success(
            "✅ **ALL COMPLIANT
