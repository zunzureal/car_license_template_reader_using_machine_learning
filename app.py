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

# Cache the model loading to avoid reloading on every rerun
@st.cache_resource
def load_models():
    try:
        # Load OCR models
        processor = TrOCRProcessor.from_pretrained("spykittichai/th-character-ocr")
        model = VisionEncoderDecoderModel.from_pretrained("spykittichai/th-character-ocr")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Load detection model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        detection_model_path = os.path.join(script_dir, "detection/model/best.pt")
        
        if not os.path.exists(detection_model_path):
            st.warning(f"Model file not found at {detection_model_path}, trying relative path")
            detection_model_path = "detection/model/best.pt"
            
            if not os.path.exists(detection_model_path):
                st.error(f"Model file not found at {detection_model_path} either")
                return None, None, None, None
        
        detection_model = YOLO(detection_model_path)
        
        return processor, model, device, detection_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def predict(processor, model, device, image_path):
    try:
        image = image_path.resize((384, 384))
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
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
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    
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
            result_placeholder.markdown(f"""
            ### Detected License Plate
            
            **Text:** {license_text}
            """)
            status_indicator.success("License plate detected and recognized!")
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
    
    # Check available cameras first
    available_cameras = check_webcam_available()
    webcam_available = len(available_cameras) > 0
    
    # If running on Streamlit Cloud, webcams might be disabled
    is_cloud_env = os.environ.get('STREAMLIT_SHARING') or os.environ.get('STREAMLIT_CLOUD')
    
    # Load models
    with st.spinner("Loading models..."):
        processor, model, device, detection_model = load_models()
        if processor is None or model is None or device is None or detection_model is None:
            st.error("Failed to load models. Please check your model files and dependencies.")
            return
        st.success("Models loaded successfully! Ready to detect license plates.")
    
    # Sidebar options
    st.sidebar.title("Options")
    
    # Determine available input sources
    input_sources = ["Upload Image", "Upload Video"]
    if webcam_available and not is_cloud_env:
        input_sources.insert(0, "Webcam")
        
    # Show warning if webcam not available
    if not webcam_available and not st.session_state.webcam_warning_shown:
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
    
    if input_option == "Webcam" and webcam_available:
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
            
        # Demo image toggle - for environments where webcam doesn't work
        use_demo_mode = st.sidebar.checkbox("Use Demo Mode (for environments without camera access)", False)
        
        if use_demo_mode:
            st.info("Using demo mode with test images instead of webcam feed.")
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
        else:
            # Create start/stop button for webcam
            button_text = "Stop Camera" if st.session_state.camera_active else "Start Camera"
            st.button(button_text, on_click=start_stop_webcam)
                
            if st.session_state.camera_active:
                status_indicator.info("Starting webcam...")
                
                # Initialize webcam capture
                try:
                    # Try different webcam access methods
                    cap = cv2.VideoCapture(camera_index)
                    
                    # Check if webcam opened successfully - we already verified it works above,
                    # but double-check in case another app grabbed it
                    if not cap.isOpened():
                        status_indicator.error(f"Could not open webcam with index {camera_index}. It may have been claimed by another application.")
                        st.session_state.camera_active = False
                        return
                    
                    status_indicator.success("Webcam started! Processing frames...")
                    
                    # Reset stopping flag
                    st.session_state.stop_webcam = False
                    
                    # For cooldown between detections
                    last_detection_time = 0
                    cooldown_period = 1  # 1 second between detections
                    
                    # Create a loop for camera frames
                    frame_placeholder = st.empty()
                    
                    while not st.session_state.stop_webcam:
                        # Read frame
                        ret, frame = cap.read()
                        
                        if not ret:
                            status_indicator.warning("Failed to capture frame from camera")
                            time.sleep(0.5)  # Wait a bit and try again
                            continue
                        
                        # Process every frame but only run detection on cooldown
                        current_time = time.time()
                        if current_time - last_detection_time > cooldown_period:
                            annotated_frame, license_text, plate_found = process_frame(
                                frame, detection_model, processor, model, device
                            )
                            
                            if license_text:
                                st.session_state.last_license_text = license_text
                                last_detection_time = current_time
                        else:
                            # Just add the previous detection results to the frame
                            annotated_frame = frame.copy()
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
                            result_placeholder.markdown(f"""
                            ### Detected License Plate
                            
                            **Text:** {st.session_state.last_license_text}
                            """)
                        
                        # Small delay to reduce CPU usage
                        time.sleep(0.03)
                        
                        # Check if session state is still active (button might have been pressed)
                        if not st.session_state.camera_active:
                            st.session_state.stop_webcam = True
                    
                    # Release webcam
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
            image_bytes = uploaded_file.getvalue()
            use_uploaded_image(image_bytes, detection_model, processor, model, device, 
                             status_indicator, image_placeholder, result_placeholder)
    
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
                
                # Process the video
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every 5th frame for speed (adjust as needed)
                    if frame_counter % 5 == 0:
                        annotated_frame, license_text, plate_found = process_frame(
                            frame, detection_model, processor, model, device
                        )
                        
                        if license_text:
                            # Check if it's a new detection or same as previous
                            if not all_detections or all_detections[-1][1] != license_text:
                                all_detections.append((frame_counter, license_text))
                            
                        # Display the current frame
                        if frame_counter % 15 == 0:  # Update display less frequently
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
                status_indicator.success(f"Video processing complete! Found {len(all_detections)} license plates.")
                
                if all_detections:
                    result_text = "### Detected License Plates\n\n"
                    for frame_num, text in all_detections:
                        time_sec = frame_num / fps
                        result_text += f"- At {time_sec:.2f}s (frame {frame_num}): **{text}**\n"
                    
                    result_placeholder.markdown(result_text)
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
    
    The system will detect license plates and attempt to read the text.
    """)

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

if __name__ == "__main__":
    main()