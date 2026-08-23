import streamlit as st
import os
import sys
import tempfile
import urllib.request
import io
import re
import shutil
import yt_dlp
import imageio_ffmpeg
import instaloader
from PIL import Image

# Ensure root import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.detector import load_model, predict_face
from utils.face_extractor import extract_primary_face
from utils.video_processor import process_video

st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS to hide Streamlit input instructions ("Press Enter to apply")
st.markdown("""
<style>
    [data-testid="stWidgetInstructions"],
    [data-testid="stInputInstructions"],
    div[data-testid="stInputInstructions"],
    span[data-testid="stWidgetInstructions"],
    small,
    .st-emotion-cache-1wivap2 {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Deepfake Detection System")
st.write("Upload an Image/Video file or enter any Social Media link (YouTube, Instagram, Facebook, TikTok, Twitter/X) to analyze for deepfakes.")

# Load cached model
@st.cache_resource
def get_model():
    model, device = load_model()
    return model, device

with st.spinner("Loading Neural Network..."):
    model, device = get_model()

# Input Options (Tabs for File Upload vs URL Input)
tab1, tab2 = st.tabs(["📁 Upload File", "🌐 Any Social Media / Web Link"])

image_exts = ['.jpg', '.jpeg', '.png', '.webp', '.tiff', '.bmp', '.jfif', '.heic', '.avif']
video_exts = ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.m4v']

# TAB 1: File Upload
with tab1:
    uploaded_file = st.file_uploader(
        "Choose an Image or Video file",
        type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov", "webm"]
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        ext = os.path.splitext(file_name)[1]

        if ext in image_exts:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Analyze Uploaded Image", type="primary", use_container_width=True, key="btn_upload_img"):
                with st.spinner("Analyzing image for deepfake forgery artifacts..."):
                    face_crop, box, has_face = extract_primary_face(image, fallback_to_full=True)
                    pred = predict_face(model, face_crop, device)

                    is_fake = pred['is_deepfake']
                    confidence = pred['confidence'] * 100

                    st.divider()
                    if is_fake:
                        st.error(f"🚨 **Prediction: DEEPFAKE** (Confidence: {confidence:.2f}%)")
                    else:
                        st.success(f"✅ **Prediction: REAL** (Confidence: {confidence:.2f}%)")

                    col1, col2 = st.columns(2)
                    col1.metric("Real Probability", f"{pred['probs']['real']*100:.2f}%")
                    col2.metric("Deepfake Probability", f"{pred['probs']['deepfake']*100:.2f}%")

        elif ext in video_exts:
            st.video(uploaded_file)
            max_frames_sel = st.slider("Select Keyframes to Analyze:", min_value=10, max_value=200, value=30, step=10, key="upload_slider")

            if st.button("Analyze Uploaded Video", type="primary", use_container_width=True, key="btn_upload_vid"):
                progress_bar = st.progress(0, text="Analyzing video frames... (0%)")
                video_path = None
                try:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                    tfile.write(uploaded_file.read())
                    tfile.close()
                    video_path = tfile.name

                    def update_progress(current, total):
                        if total > 0:
                            pct = min(1.0, current / total)
                            progress_bar.progress(pct, text=f"Analyzing video frames... ({int(pct*100)}%)")

                    results = process_video(
                        video_path=video_path,
                        model=model,
                        device=device,
                        sample_rate=15,
                        max_frames=max_frames_sel,
                        progress_callback=update_progress
                    )

                    progress_bar.progress(1.0, text="Analysis Complete! (100%)")

                    if results.get('error'):
                        st.warning("⚠️ Unable to decode video file stream. Please ensure the file is a valid video in standard MP4 format.")
                    else:
                        is_fake = results['is_deepfake']
                        confidence = results['confidence'] * 100

                        st.divider()
                        if is_fake:
                            st.error(f"🚨 **Overall Video Prediction: DEEPFAKE** (Confidence: {confidence:.2f}%)")
                        else:
                            st.success(f"✅ **Overall Video Prediction: REAL** (Confidence: {confidence:.2f}%)")

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Avg Fake Probability", f"{results['avg_fake_prob']*100:.2f}%")
                        col2.metric("Max Fake Probability", f"{results['max_fake_prob']*100:.2f}%")
                        col3.metric("Fake Frame Ratio", f"{results['fake_frame_ratio']*100:.1f}%")
                finally:
                    if video_path and os.path.exists(video_path):
                        try:
                            os.remove(video_path)
                        except Exception:
                            pass

# TAB 2: Social Media & Any Web URL Detection
with tab2:
    url_input = st.text_input("Enter Link (YouTube, Instagram, Facebook, Twitter, TikTok, or Direct Image/Video URL):", placeholder="Enter URL...")
    submit_url = st.button("Analyze Link Media", type="primary", use_container_width=True, key="btn_url_submit")

    url_output_container = st.container()

    if submit_url:
        clean_url = url_input.strip()
        with url_output_container:
            if not clean_url:
                st.error("Please paste an Image or Video URL first.")
            else:
                with st.spinner("Extracting media stream from platform link..."):
                    download_success = False
                    video_path = None
                    image_from_link = None
                    temp_dir_to_clean = None

                    # 1. Check if URL is an explicit direct image link
                    if any(clean_url.lower().endswith(i) for i in image_exts) or 'type=profile' in clean_url:
                        try:
                            req = urllib.request.Request(
                                clean_url,
                                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
                            )
                            with urllib.request.urlopen(req, timeout=15) as resp:
                                media_bytes = resp.read()
                                image_from_link = Image.open(io.BytesIO(media_bytes)).convert('RGB')
                                download_success = True
                        except Exception:
                            pass

                    # 2. Instagram extraction via Instaloader
                    if not download_success and 'instagram.com' in clean_url:
                        try:
                            match = re.search(r'/(p|reel|reels)/([A-Za-z0-9_-]+)', clean_url)
                            if match:
                                shortcode = match.group(2)
                                L = instaloader.Instaloader(download_pictures=False, download_videos=False)
                                post = instaloader.Post.from_shortcode(L.context, shortcode)
                                
                                if post.is_video and post.video_url:
                                    req = urllib.request.Request(post.video_url, headers={'User-Agent': 'Mozilla/5.0'})
                                    with urllib.request.urlopen(req, timeout=15) as resp:
                                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                                        tfile.write(resp.read())
                                        tfile.close()
                                        video_path = tfile.name
                                        download_success = True
                                elif post.url:
                                    req = urllib.request.Request(post.url, headers={'User-Agent': 'Mozilla/5.0'})
                                    with urllib.request.urlopen(req, timeout=15) as resp:
                                        image_from_link = Image.open(io.BytesIO(resp.read())).convert('RGB')
                                        download_success = True
                        except Exception:
                            pass

                    # 3. Universal yt-dlp Extractor (YouTube, Shorts, TikTok, FB, Twitter, Web Video)
                    if not download_success:
                        try:
                            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                            temp_dir = tempfile.mkdtemp()
                            temp_dir_to_clean = temp_dir
                            ydl_opts = {
                                'format': 'bestvideo*+bestaudio*/best',
                                'outtmpl': os.path.join(temp_dir, 'media_%(id)s.%(ext)s'),
                                'ffmpeg_location': ffmpeg_exe,
                                'quiet': True,
                                'no_warnings': True,
                                'nocheckcertificate': True,
                                'max_filesize': 200 * 1024 * 1024
                            }
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                info = ydl.extract_info(clean_url, download=True)
                                if 'requested_downloads' in info and len(info['requested_downloads']) > 0:
                                    video_path = info['requested_downloads'][0]['filepath']
                                else:
                                    video_path = ydl.prepare_filename(info)

                                if video_path and os.path.exists(video_path):
                                    try:
                                        img_check = Image.open(video_path)
                                        img_check.verify()
                                        image_from_link = Image.open(video_path).convert('RGB')
                                        video_path = None
                                        download_success = True
                                    except Exception:
                                        download_success = True
                        except Exception:
                            pass

                    # 4. Fallback HTTP fetch for direct image/video URLs
                    if not download_success:
                        try:
                            req = urllib.request.Request(
                                clean_url,
                                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
                            )
                            with urllib.request.urlopen(req, timeout=15) as resp:
                                media_bytes = resp.read()
                                c_type = resp.headers.get('Content-Type', '').lower()

                            # Try decoding as image first
                            try:
                                image_from_link = Image.open(io.BytesIO(media_bytes)).convert('RGB')
                                download_success = True
                            except Exception:
                                if 'video' in c_type or any(clean_url.lower().endswith(v) for v in video_exts):
                                    try:
                                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                                        tfile.write(media_bytes)
                                        tfile.close()
                                        video_path = tfile.name
                                        download_success = True
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                    # Process Image if extracted
                    if download_success and image_from_link is not None:
                        st.image(image_from_link, caption="Extracted Image", use_container_width=True)

                        with st.spinner("Analyzing image for deepfake forgery artifacts..."):
                            face_crop, box, has_face = extract_primary_face(image_from_link, fallback_to_full=True)
                            pred = predict_face(model, face_crop, device)

                            is_fake = pred['is_deepfake']
                            confidence = pred['confidence'] * 100

                            st.divider()
                            if is_fake:
                                st.error(f"🚨 **Prediction: DEEPFAKE** (Confidence: {confidence:.2f}%)")
                            else:
                                st.success(f"✅ **Prediction: REAL** (Confidence: {confidence:.2f}%)")

                            col1, col2 = st.columns(2)
                            col1.metric("Real Probability", f"{pred['probs']['real']*100:.2f}%")
                            col2.metric("Deepfake Probability", f"{pred['probs']['deepfake']*100:.2f}%")

                    # Process Video if extracted
                    elif download_success and video_path and os.path.exists(video_path):
                        st.video(video_path)
                        progress_bar = st.progress(0, text="Analyzing video frames... (0%)")

                        def update_progress(current, total):
                            if total > 0:
                                pct = min(1.0, current / total)
                                progress_bar.progress(pct, text=f"Analyzing video frames... ({int(pct*100)}%)")

                        try:
                            results = process_video(
                                video_path=video_path,
                                model=model,
                                device=device,
                                sample_rate=15,
                                max_frames=30,
                                progress_callback=update_progress
                            )

                            progress_bar.progress(1.0, text="Analysis Complete! (100%)")

                            if results.get('error'):
                                st.warning("⚠️ Unable to decode video stream from URL. Please try downloading the video file directly and uploading it.")
                            else:
                                is_fake = results['is_deepfake']
                                confidence = results['confidence'] * 100

                                st.divider()
                                if is_fake:
                                    st.error(f"🚨 **Overall Video Prediction: DEEPFAKE** (Confidence: {confidence:.2f}%)")
                                else:
                                    st.success(f"✅ **Overall Video Prediction: REAL** (Confidence: {confidence:.2f}%)")

                                col1, col2, col3 = st.columns(3)
                                col1.metric("Avg Fake Probability", f"{results['avg_fake_prob']*100:.2f}%")
                                col2.metric("Max Fake Probability", f"{results['max_fake_prob']*100:.2f}%")
                                col3.metric("Fake Frame Ratio", f"{results['fake_frame_ratio']*100:.1f}%")
                        finally:
                            if temp_dir_to_clean and os.path.exists(temp_dir_to_clean):
                                try:
                                    shutil.rmtree(temp_dir_to_clean)
                                except Exception:
                                    pass

                    elif not download_success:
                        st.error("Unable to extract media from link. Please verify that the link is public or try copying the direct image/video URL.")
