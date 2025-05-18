import streamlit as st
import cv2
import time
import os
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
import re
from itertools import groupby
import tempfile
import platform
import traceback
import requests
import base64
import io
from datetime import datetime
import pytz

# Page configuration
st.set_page_config(
    page_title="License Plate Reader",
    page_icon="🚗",
    layout="wide"
)

# Initialize session state variables
if "stop_webcam" not in st.session_state:
    st.session_state.stop_webcam = True
if "last_license_text" not in st.session_state:
    st.session_state.last_license_text = None
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "use_webcam" not in st.session_state:
    st.session_state.use_webcam = False
if "webcam_warning_shown" not in st.session_state:
    st.session_state.webcam_warning_shown = False
if "detections_saved" not in st.session_state:
    st.session_state.detections_saved = set()

# Supabase integration
SUPABASE_URL = "https://nyfaluazyaribgfqryxy.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im55ZmFsdWF6eWFyaWJnZnFyeXh5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NzA0NzQ1MCwiZXhwIjoyMDYyNjIzNDUwfQ.rbytQ-q5a8cN-A-LakAmtywl2VqXn-CiTeJXkhKJeIk"

def is_valid_license_plate(text):
    """
    Check if the detected text appears to be a valid Thai license plate.
    A valid license plate should have:
    1. At least 2 Thai characters (ก-๙)
    2. Exactly 4-5 numbers (0-9) 
    3. Total length of at least 6 characters
    """
    if not text or len(text) <= 5:
        return False
        
    thai_chars = re.findall(r'[ก-๙]', text)
    has_thai = len(thai_chars) == 2
    
    numbers = re.findall(r'[0-9]', text)
    has_four_numbers = 4 <= len(numbers) <= 5
    
    # Return True only if both criteria are met
    return has_thai and has_four_numbers

def insert_data_to_supabase(plate, city=None, image_data=None):
    """
    Save license plate data and captured image to Supabase database
    """
    if not plate:
        return False
    
    # Validate license plate format before saving
    if not is_valid_license_plate(plate):
        st.warning(f"'{plate}' does not appear to be a valid license plate (needs Thai characters and 4-5 numbers). Not saving to database.")
        return False
        
    url = f"{SUPABASE_URL}/rest/v1/car"
    headers = {
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }

    # Ensure we have a city value or use default
    if not city:
        city = "Unknown"

    # Get current timestamp in ISO format
    bangkok_tz = pytz.timezone('Asia/Bangkok')
    current_time = datetime.now(bangkok_tz).strftime("%Y-%m-%d %H:%M:%S")

    # Prepare base data
    data = {
        "plate_number": plate,
        "city": city,
        "detected_at": current_time
    }
    
    # Add image data if provided
    if image_data is not None:
        try:
            # Resize image to reduce storage size (adjust dimensions as needed)
            image_resized = cv2.resize(image_data, (640, 480))
            
            # Convert to JPEG format with compression
            is_success, buffer = cv2.imencode(".jpg", image_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if is_success:
                # Convert to base64 for storage
                img_str = base64.b64encode(buffer).decode('utf-8')
                data["image_data"] = img_str
            else:
                st.warning("Failed to encode image for storage")
        except Exception as img_error:
            st.warning(f"Could not process image for database storage: {img_error}")

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error saving to database: {e}")
        return False

# Cache the model loading to avoid reloading on every rerun
@st.cache_resource
def load_models():
    try:
        # Load OCR models with use_fast=False to fix the warning
        processor = TrOCRProcessor.from_pretrained(
            "spykittichai/th-character-ocr", 
            use_fast=False,
            trust_remote_code=True
        )
        
        model = VisionEncoderDecoderModel.from_pretrained(
            "spykittichai/th-character-ocr",
            trust_remote_code=True
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Load detection model with better path handling
        try:
            # First try the absolute path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            detection_model_path = os.path.join(script_dir, "detection/model/best.pt")
            
            # If not found, try relative paths
            if not os.path.exists(detection_model_path):
                candidate_paths = [
                    "detection/model/best.pt",
                    "./detection/model/best.pt",
                    "../detection/model/best.pt",
                    "best.pt"
                ]
                
                for path in candidate_paths:
                    if os.path.exists(path):
                        detection_model_path = path
                        st.success(f"Found model at {path}")
                        break
                else:
                    # If model still not found, try downloading a default model
                    st.warning("Model not found locally, downloading a default YOLO model...")
                    detection_model = YOLO("yolov8n.pt")
                    return processor, model, device, detection_model
            
            detection_model = YOLO(detection_model_path)
            return processor, model, device, detection_model
            
        except Exception as model_error:
            st.error(f"Error loading detection model: {model_error}")
            st.info("Falling back to default YOLO model...")
            detection_model = YOLO("yolov8n.pt")
            return processor, model, device, detection_model
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.code(traceback.format_exc())
        return None, None, None, None

def predict(processor, model, device, image_path):
    try:
        image = image_path.resize((384, 384))
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_length=20)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return ""

def clean_and_filter(text):
    # 1. ลบตัวซ้ำที่ติดกัน เช่น '777777' → '7'
    text = ''.join(k for k, _ in groupby(text))
    # 2. ใช้ regex เพื่อดึงเฉพาะ "ตัวเลข" และ "อักษรไทย"
    filtered = re.findall(r'[ก-๙0-9]', text)
    return ''.join(filtered)

def check_img(processor, model, device, img):
    if img is None:
        return None

    # เช็คว่า img เป็น numpy array หรือไม่
    if not isinstance(img, np.ndarray):
        return None
    
    try:
        img = cv2.resize(img, (348, 348))  # Resize ขนาดภาพ
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilation = cv2.dilate(thresh, kernel, iterations=1)

        erosion = cv2.erode(dilation, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cropped_images = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cropped = erosion[y:y+h, x:x+w]  # Crop ภาพ
            cropped_images.append((x, y, cropped))

        line_threshold = 150
        lines = []

        cropped_images.sort(key=lambda item: item[1]) 

        for item in cropped_images:
            x, y, cropped = item
            placed = False
            for line in lines:
                if abs(line[0][1] - y) < line_threshold:
                    line.append(item)
                    placed = True
                    break
            if not placed:
                lines.append([item]) 
                
        for line in lines:
            line.sort(key=lambda item: item[0])

        cropped_images = [item for line in lines for item in line]
        
        passed_img = []
        
        for _, _, cropped in cropped_images:
            height, width = cropped.shape

            white_pixels = cv2.countNonZero(cropped)

            if 60 < height <= 170 and 12 < width <= 60 and white_pixels > 700:
                passed_img.append(cropped)
        
        word = []
        word_no_clean = []
        for img in passed_img:
            if len(img.shape) == 2: 
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(img)

            result = predict(processor, model, device, pil_img)
            word_no_clean.append(result)
            cleaned = clean_and_filter(result)
            if cleaned:
                word.append(cleaned)
        
        license_text = "".join(word)
        return license_text
    except Exception as e:
        st.error(f"Error in check_img: {e}")
        return None

def detect_and_crop_license_plates(image, model):
    """
    ตรวจจับและ crop ป้ายทะเบียนจากภาพโดยใช้ YOLO model (1 รอบ)
    """
    if image is None:
        return [], []
    
    try:    
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb_image)

        cropped_images = []
        boxes = []

        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                x1, y1, x2, y2 = box
                # Ensure indices are within bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                
                # Skip invalid boxes
                if x1 >= x2 or y1 >= y2:
                    continue
                    
                cropped = image[y1:y2, x1:x2]
                cropped_images.append(cropped)

        return cropped_images, boxes
    except Exception as e:
        st.error(f"Error in detect_and_crop_license_plates: {e}")
        return [], []

def process_frame(frame, detection_model, processor, model, device):
    if frame is None:
        return None, None, False
        
    try:
        # First detection
        cropped_plates, boxes = detect_and_crop_license_plates(frame, detection_model)
        
        license_text = None
        annotated_frame = frame.copy()
        
        # Draw boxes on the frame
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Process first plate if found
        if cropped_plates:
            best_plate = cropped_plates[0]
            
            # Try second detection on cropped plate for better accuracy
            second_crop, _ = detect_and_crop_license_plates(best_plate, detection_model)
            
            if second_crop and len(second_crop) > 0:
                final_crop = second_crop[0]
                license_text = check_img(processor, model, device, final_crop)
            else:
                license_text = check_img(processor, model, device, best_plate)
                
        return annotated_frame, license_text, bool(cropped_plates)
    except Exception as e:
        st.error(f"Error processing frame: {e}")
        return frame, None, False

def start_stop_webcam():
    st.session_state.camera_active = not st.session_state.camera_active
    if not st.session_state.camera_active:
        st.session_state.stop_webcam = True

def check_webcam_available():
    """Check if any webcam is available on the system."""
    available_cameras = []
    max_to_check = 3  # Check cameras 0, 1, 2
    
    for i in range(max_to_check):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                cap.release()
        except:
            pass
    
    return available_cameras

def use_uploaded_image(image_bytes, detection_model, processor, model, device, status_indicator, image_placeholder, result_placeholder):
    """Process an uploaded image."""
    try:
        # Convert to numpy array
        file_bytes = np.asarray(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            status_indicator.error("Could not read uploaded image.")
            return
        
        status_indicator.info("Processing image...")
        
        # Process the image
        annotated_frame, license_text, plate_found = process_frame(
            image, detection_model, processor, model, device
        )
        
        # Convert to RGB for display
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Display results
        image_placeholder.image(annotated_frame_rgb, channels="RGB", caption="Uploaded Image")
        
        if license_text:
            # Get the city from user input
            city = st.text_input("Enter city (optional):", key="city_input")
            
            # Check if it's a valid license plate format
            is_valid = is_valid_license_plate(license_text)
            
            # Display validation status
            if is_valid:
                validation_status = "✅ Valid license plate format detected"
            else:
                validation_status = "⚠️ Not a valid license plate format (missing characters or numbers)"
            
            # Add a save button (enabled only if the format is valid)
            save_clicked = st.button("Save to Database", disabled=not is_valid)
            
            result_placeholder.markdown(f"""
            ### Detected License Plate
            
            **Text:** {license_text}
            
            **Validation:** {validation_status}
            """)
            
            status_indicator.success("License plate detected and recognized!")
            
            # Save to database if button clicked and format is valid
            if save_clicked and is_valid:
                if insert_data_to_supabase(license_text, city, annotated_frame):
                    status_indicator.success(f"License plate {license_text} saved to database with image!")
                    st.session_state.detections_saved.add(license_text)
                else:
                    status_indicator.error("Failed to save to database.")
        else:
            if plate_found:
                result_placeholder.warning("License plate detected, but text couldn't be recognized.")
                status_indicator.warning("License plate detected, but couldn't read the text.")
            else:
                result_placeholder.error("No license plate detected in the image.")
                status_indicator.error("No license plate detected.")
    except Exception as e:
        status_indicator.error(f"Error processing image: {e}")

def main():
    st.title("🚗 Thai License Plate Recognition")
    
    # Always provide image upload option regardless of webcam
    # Check available cameras in background
    available_cameras = []
    
    try:
        available_cameras = check_webcam_available()
    except Exception as e:
        st.sidebar.warning(f"Error checking for cameras: {e}")
    
    webcam_available = len(available_cameras) > 0
    
    # If running on Streamlit Cloud, webcams might be disabled
    is_cloud_env = os.environ.get('STREAMLIT_SHARING') or os.environ.get('STREAMLIT_CLOUD')
    
    # Load models
    with st.spinner("Loading models..."):
        processor, model, device, detection_model = load_models()
        if processor is None or model is None or device is None or detection_model is None:
            st.error("Failed to load models. Please check your model files and dependencies.")
            st.markdown("""
            ### Troubleshooting:
            1. Make sure you have the detection model in `detection/model/best.pt`
            2. Make sure your transformers and torch versions are compatible
            3. Check your internet connection for downloading models
            """)
            return
        st.success("Models loaded successfully! Ready to detect license plates.")
    
    # Sidebar options
    st.sidebar.title("Options")
    
    # Always provide Demo mode
    demo_mode = st.sidebar.checkbox("Use Demo Mode (sample images)", False)
    
    # Database options
    st.sidebar.markdown("### Database Options")
    save_to_db = st.sidebar.checkbox("Auto-save valid detections to database", False)
    save_images = st.sidebar.checkbox("Save images with detections", True)
    default_city = st.sidebar.text_input("Default city for detections", "กรุงเทพมหานคร")
    
    # Determine available input sources
    input_sources = ["Upload Image", "Upload Video"]
    if webcam_available and not is_cloud_env:
        input_sources.insert(0, "Webcam")
        
    # Show warning if webcam not available
    if not webcam_available and not st.session_state.webcam_warning_shown and not demo_mode:
        st.warning("⚠️ No webcams detected on this system. Only upload options will be available.")
        st.session_state.webcam_warning_shown = True
        
    input_option = st.sidebar.radio(
        "Select input source:",
        input_sources
    )

    # Results placeholders
    col1, col2 = st.columns(2)
    with col1:
        image_placeholder = st.empty()
    with col2:
        result_placeholder = st.empty()
        
    # Create a status indicator
    status_indicator = st.empty()
    
    if demo_mode:
        st.info("Using demo mode with test images.")
        # List of demo images
        demo_images = {
            "Thai License Plate 1": "https://raw.githubusercontent.com/zunzureal/license-plate-demo-images/main/plate1.jpg",
            "Thai License Plate 2": "https://raw.githubusercontent.com/zunzureal/license-plate-demo-images/main/plate2.jpg", 
            "Demo Car": "https://raw.githubusercontent.com/zunzureal/license-plate-demo-images/main/car.jpg"
        }
        
        selected_image = st.selectbox("Select a demo image:", list(demo_images.keys()))
        
        if st.button("Process Demo Image"):
            status_indicator.info(f"Processing demo image: {selected_image}")
            
            try:
                import urllib.request
                image_url = demo_images[selected_image]
                with urllib.request.urlopen(image_url) as response:
                    image_bytes = bytearray(response.read())
                
                use_uploaded_image(image_bytes, detection_model, processor, model, device, 
                                  status_indicator, image_placeholder, result_placeholder)
            except Exception as e:
                status_indicator.error(f"Error loading demo image: {e}")
    
    elif input_option == "Webcam" and webcam_available:
        # Camera selector for available cameras
        if len(available_cameras) > 1:
            camera_index = st.sidebar.selectbox(
                "Select Camera", 
                available_cameras,
                index=0
            )
        else:
            camera_index = available_cameras[0]
            st.sidebar.info(f"Using camera at index {camera_index} (only one camera available)")
            
        # Create start/stop button for webcam
        button_text = "Stop Camera" if st.session_state.camera_active else "Start Camera"
        st.button(button_text, on_click=start_stop_webcam)
            
        if st.session_state.camera_active:
            status_indicator.info("Starting webcam...")
            
            # Initialize webcam capture
            try:
                # Try different webcam access methods
                cap = None
                
                # Try different methods to open the camera
                methods = [
                    lambda: cv2.VideoCapture(camera_index),
                    lambda: cv2.VideoCapture(camera_index, cv2.CAP_DSHOW),  # For Windows
                    lambda: cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)  # For Mac
                ]
                
                for method in methods:
                    try:
                        cap = method()
                        if cap.isOpened():
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None:
                                break
                            else:
                                cap.release()
                                cap = None
                    except:
                        if cap:
                            cap.release()
                        cap = None
                
                # Check if webcam opened successfully
                                # After successfully opening the camera
                if cap is None or not cap.isOpened():
                    status_indicator.error(f"Could not open webcam with index {camera_index}.")
                    st.session_state.camera_active = False
                    return
                
                # Set smaller resolution for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                status_indicator.success("Webcam started! Processing frames...")
                
                # Reset stopping flag
                st.session_state.stop_webcam = False
                
                # For cooldown between detections
                                # Reset stopping flag
                st.session_state.stop_webcam = False
                
                # For cooldown between detections
                last_detection_time = 0
                cooldown_period = 1  # 1 second between detections
                frame_counter = 0  # Add frame counter
                
                # Create a loop for camera frames
                frame_placeholder = st.empty()
                
                while not st.session_state.stop_webcam:
                    try:
                        # Read frame
                        ret, frame = cap.read()
                        
                        if not ret or frame is None:
                            status_indicator.warning("Failed to capture frame from camera. Retrying...")
                            time.sleep(0.5)  # Wait a bit and try again
                            continue
                        
                        # Display every frame but only process every 3rd frame
                        frame_counter += 1
                        
                        # Process frame for detection only sometimes (every 3rd frame)
                        if frame_counter % 3 == 0:
                            # Process every frame but only run detection on cooldown
                            current_time = time.time()
                            if current_time - last_detection_time > cooldown_period:
                                # Resize frame for processing to improve performance
                                processed_frame = cv2.resize(frame, (640, 480))
                                annotated_frame, license_text, plate_found = process_frame(
                                    processed_frame, detection_model, processor, model, device
                                )
                                
                                if license_text:
                                    st.session_state.last_license_text = license_text
                                    last_detection_time = current_time
                                    
                                    # Save to database if auto-save is enabled
                                    if save_to_db and license_text not in st.session_state.detections_saved and is_valid_license_plate(license_text):
                                        image_data = annotated_frame if save_images else None
                                        if insert_data_to_supabase(license_text, default_city, image_data):
                                            status_indicator.success(f"License plate {license_text} saved to database with image!")
                                            st.session_state.detections_saved.add(license_text)
                            else:
                                # Just copy the frame without processing
                                annotated_frame = frame.copy()
                        else:
                            # Use the frame directly without detection for smoother video
                            annotated_frame = frame.copy()
                            
                        # Always add any previously detected text to the current frame
                        if st.session_state.last_license_text:
                            cv2.putText(
                                annotated_frame, 
                                f"License: {st.session_state.last_license_text}", 
                                (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (0, 0, 255), 
                                2
                            )
                        
                        # Convert to RGB for Streamlit
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display frame
                        image_placeholder.image(annotated_frame_rgb, channels="RGB", caption="Webcam Feed")
                        # Display results
                        if st.session_state.last_license_text:
                            is_valid = is_valid_license_plate(st.session_state.last_license_text)
                            validation_status = "✅ Valid" if is_valid else "⚠️ Invalid format"
                            
                            result_placeholder.markdown(f"""
                            ### Detected License Plate
                            
                            **Text:** {st.session_state.last_license_text}
                            
                            **Format:** {validation_status}
                            """)
                        
                        # Small delay to reduce CPU usage
                        time.sleep(0.03)
                    except Exception as frame_error:
                        st.error(f"Error processing frame: {frame_error}")
                        time.sleep(0.5)
                        
                    # Check if session state is still active (button might have been pressed)
                    if not st.session_state.camera_active:
                        st.session_state.stop_webcam = True
                
                # Release webcam
                if cap and cap.isOpened():
                    cap.release()
                status_indicator.info("Webcam stopped.")
            
            except Exception as e:
                status_indicator.error(f"Error with webcam: {e}")
                st.session_state.camera_active = False
                st.session_state.stop_webcam = True
        
    elif input_option == "Upload Image":
        status_indicator.info("Please upload an image.")
        
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image_bytes = uploaded_file.getvalue()
                use_uploaded_image(image_bytes, detection_model, processor, model, device, 
                                status_indicator, image_placeholder, result_placeholder)
            except Exception as e:
                status_indicator.error(f"Error processing uploaded image: {e}")
    
    elif input_option == "Upload Video":
        status_indicator.info("Please upload a video file.")
        
        uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
        
        if uploaded_file is not None:
            try:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_file_path = tmp_file.name
                
                status_indicator.info("Processing video...")
                
                # Open the video file
                cap = cv2.VideoCapture(temp_file_path)
                
                if not cap.isOpened():
                    status_indicator.error("Error opening video file.")
                    # Clean up and return
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                    return
                    
                # Get video info for the progress bar
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Create a progress bar
                progress_bar = st.progress(0)
                frame_counter = 0
                
                # Container for displaying detection results
                all_detections = []
                valid_detections = []  # Track which detections have valid format
                frames_with_detections = {}  # Store frames for valid detections
                
                # Process the video
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every 5th frame for speed (adjust as needed)
                    if frame_counter % 25 == 0:
                        annotated_frame, license_text, plate_found = process_frame(
                            frame, detection_model, processor, model, device
                        )
                        
                        if license_text:
                            # Check if it's a new detection or same as previous
                            if not all_detections or all_detections[-1][1] != license_text:
                                is_valid = is_valid_license_plate(license_text)
                                all_detections.append((frame_counter, license_text, is_valid))
                                
                                # Track valid detections separately
                                if is_valid:
                                    valid_detections.append((frame_counter, license_text))
                                    # Keep the frame if we want to save images
                                    if save_images:
                                        frames_with_detections[frame_counter] = annotated_frame
                                
                                # Save to database if auto-save is enabled and format is valid
                                if save_to_db and license_text not in st.session_state.detections_saved and is_valid:
                                    image_data = annotated_frame if save_images else None
                                    if insert_data_to_supabase(license_text, default_city, image_data):
                                        st.session_state.detections_saved.add(license_text)
                            
                        # Display the current frame
                        if frame_counter % 30 == 0:  # Update display less frequently
                            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            image_placeholder.image(annotated_frame_rgb, channels="RGB", caption=f"Frame {frame_counter}")
                    
                    # Update progress
                    frame_counter += 1
                    progress_bar.progress(min(frame_counter / total_frames, 1.0))
                
                cap.release()
                
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                
                # Show results
                status_indicator.success(f"Video processing complete! Found {len(all_detections)} potential license plates ({len(valid_detections)} valid format).")
                
                if all_detections:
                    result_text = "### Detected License Plates\n\n"
                    for frame_num, text, is_valid in all_detections:
                        time_sec = frame_num / fps
                        validation_mark = "✅" if is_valid else "⚠️"
                        result_text += f"- At {time_sec:.2f}s (frame {frame_num}): **{text}** {validation_mark}\n"
                    
                    result_placeholder.markdown(result_text)
                    
                    # Show save all button if auto-save is not enabled and there are valid plates
                    if not save_to_db and valid_detections:
                        save_all = st.button("Save Valid Detections to Database")
                        if save_all:
                            success_count = 0
                            for frame_num, plate in valid_detections:
                                if plate not in st.session_state.detections_saved:
                                    # Use the stored frame if available
                                    image_data = frames_with_detections.get(frame_num) if save_images else None
                                    if insert_data_to_supabase(plate, default_city, image_data):
                                        success_count += 1
                                        st.session_state.detections_saved.add(plate)
                            
                            status_indicator.success(f"Saved {success_count} valid license plates to database!")
                else:
                    result_placeholder.warning("No license plates detected in the video.")
            except Exception as e:
                status_indicator.error(f"Error processing video: {e}")
                # Clean up if necessary
                if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

    # Add instructions at the bottom
    st.markdown("""
    ### How to use:
    1. Select an input source from the sidebar (Webcam, Upload Image, or Upload Video).
    2. For webcam, press Start Camera and wait for detections.
    3. For images and videos, upload a file and wait for processing.
    4. Use the database options to automatically save valid license plate detections.
    
    The system will detect license plates, attempt to read the text, validate the format, and save valid plates to the database.
    
    **Note:** Only license plates with at least 2 Thai characters and 4-5 numbers will be saved to the database. Images will also be stored for monitoring purposes.
    """)

    # Add database section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Database Status")
    saved_count = len(st.session_state.detections_saved)
    st.sidebar.markdown(f"• Plates saved this session: **{saved_count}**")
    
    if saved_count > 0:
        st.sidebar.markdown("• Saved plates:")
        for plate in st.session_state.detections_saved:
            st.sidebar.markdown(f"  - {plate}")

    # Add deployment info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Deployment Information")
    st.sidebar.markdown(f"• Running on: **{platform.system()} {platform.release()}**")
    st.sidebar.markdown(f"• Using device: **{device}**")
    
    # Add webcam status
    if webcam_available:
        st.sidebar.markdown(f"• Available cameras: **{len(available_cameras)}** (indices: {', '.join(map(str, available_cameras))})")
    else:
        st.sidebar.markdown("• No cameras detected")
    
    # Add versions info
    st.sidebar.markdown(f"• Python version: **{platform.python_version()}**")
    st.sidebar.markdown(f"• OpenCV version: **{cv2.__version__}**")

if __name__ == "__main__":
    main()