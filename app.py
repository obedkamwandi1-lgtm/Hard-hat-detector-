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


def render_fully_automatic_monitor(
    conf_threshold: float, zone_name: str, lat: float, lon: float
):
    """Hands-free HTML5 real-time camera monitor with direct Roboflow inference & live alarm."""
    maps_link = f"https://www.google.com/maps?q={lat},{lon}"
    html_code = f"""
    <div style="font-family: sans-serif; color: #fafafa; background: #1e2129; padding: 16px; border-radius: 10px; border: 1px solid #31333f;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 10px;">
            <div>
                <span id="status-badge" style="background: #2e7d32; color: white; padding: 5px 12px; border-radius: 20px; font-weight: bold; font-size: 13px;">
                    🟢 AUTO-CAPTURE ACTIVE
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
            statusBadge.innerText = "🟢 AUTO-CAPTURE MONITORING ACTIVE";
            statusBadge.style.background = "#2e7d32";
            // Automatically analyze a frame every 1.5 seconds without clicking anything
            setInterval(processFrame, 1500);
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
            console.error("Auto-capture frame error:", err);
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


# Sidebar controls
st.sidebar.header("⚙️ Detection Settings")
app_mode = st.sidebar.radio(
    "Select Operation Mode",
    ["Image / Snapshot", "Video File", "Live Webcam Feed"],
)
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.1, max_value=1.0, value=0.4, step=0.05
)

# Sidebar Location & GPS Tracking
st.sidebar.markdown("---")
st.sidebar.header("📍 Site Location & GPS")
location_source = st.sidebar.radio(
    "Location Source",
    ["Fixed Site Zone", "Automatic Device GPS"],
)

SITE_ZONES = {
    "Kansanshi Mine (Solwezi)": (-12.115800, 26.402500),
    "Main Site Gate (Lusaka HQ)": (-15.387500, 28.322800),
    "Copperbelt Shaft 1 (Kitwe)": (-12.802400, 28.213200),
    "Processing Plant (Chingola)": (-12.528900, 27.883800),
    "Smelter Zone (Mufulira)": (-12.549800, 28.240700),
}

active_lat, active_lon = -12.115800, 26.402500
active_zone_name = "Kansanshi Mine (Solwezi)"

if location_source == "Automatic Device GPS":
    loc = get_geolocation()
    if loc and "coords" in loc:
        active_lat = float(loc["coords"]["latitude"])
        active_lon = float(loc["coords"]["longitude"])
        active_zone_name = "Live Device GPS"
        st.sidebar.success(f"GPS Locked: {active_lat:.5f}, {active_lon:.5f}")
    else:
        st.sidebar.warning(
            "Allow browser location permission for live GPS. Using Kansanshi Mine"
            " fallback coordinates."
        )
        active_zone_name = "Kansanshi Mine (Fallback)"
else:
    selected_zone = st.sidebar.selectbox(
        "Select Camera Zone", list(SITE_ZONES.keys())
    )
    active_lat, active_lon = SITE_ZONES[selected_zone]
    active_zone_name = selected_zone
    st.sidebar.info(f"Zone GPS: {active_lat:.5f}, {active_lon:.5f}")

# Sidebar QR Code for Mobile Access
st.sidebar.markdown("---")
st.sidebar.header("📱 Scan to Mobile")
app_url = "https://iyu76icob8pbcc5jpfrz95.streamlit.app/"
qr_img = qrcode.make(app_url)
qr_buf = io.BytesIO()
qr_img.save(qr_buf, format="PNG")
st.sidebar.image(qr_buf.getvalue(), caption="Scan to open on mobile")

# Sidebar Audit Trail Archive Viewer
st.sidebar.markdown("---")
st.sidebar.header("📁 Audit Trail Archive")
archived_files = sorted(
    [f for f in os.listdir("archive") if f.endswith(".png")], reverse=True
)
st.sidebar.text(f"Total Archived Captures: {len(archived_files)}")
if archived_files:
    selected_archive = st.sidebar.selectbox(
        "Review Past Violation", ["Select capture..."] + archived_files
    )
    if selected_archive != "Select capture...":
        st.sidebar.image(
            os.path.join("archive", selected_archive), caption=selected_archive
        )

if app_mode == "Image / Snapshot":
    camera_file = st.camera_input("Take a live photo")
    uploaded_file = st.file_uploader(
        "Or upload an image file", type=["jpg", "jpeg", "png"]
    )

    selected_file = camera_file if camera_file is not None else uploaded_file

    if selected_file is not None:
        image = Image.open(selected_file)
        # process image handler as before
    else:
        st.info("No photo captured or uploaded yet. Take or upload a photo above.")

elif app_mode == "Video File":
    st.markdown("### 🎥 Video Stream Compliance Analysis")
    video_file = st.file_uploader(
        "Upload a construction site video", type=["mp4", "avi", "mov"]
    )
    if video_file is not None:
        st.info("Video file loaded.")

elif app_mode == "Live Webcam Feed":
    st.markdown("### 🔴 Fully Automated Live Camera Monitoring")
    st.info(
        f"Monitoring **{active_zone_name}** (`{active_lat:.5f},"
        f" {active_lon:.5f}`) — frames are automatically captured and analyzed in real-time."
    )
    render_fully_automatic_monitor(
        conf_threshold, active_zone_name, active_lat, active_lon
    )
    render_google_map(active_lat, active_lon, active_zone_name)
