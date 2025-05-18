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

# Page configuration
st.set_page_config(
    page_title="License Plate Reader",
    page_icon="🚗",
    layout="wide"
)

# Cache the model loading to avoid reloading on every rerun
@st.cache_resource
def load_models():
    # Load OCR models
    processor = TrOCRProcessor.from_pretrained("spykittichai/th-character-ocr")
    model = VisionEncoderDecoderModel.from_pretrained("spykittichai/th-character-ocr")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load detection model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detection_model_path = os.path.join(script_dir, "detection/model/best.pt")
    
    if not os.path.exists(detection_model_path):
        st.error(f"Model file not found: {detection_model_path}")
        detection_model_path = "detection/model/best.pt"  # Try relative path
        
    detection_model = YOLO(detection_model_path)
    
    return processor, model, device, detection_model

# Functions from your original code
def predict(processor, model, device, image_path):
    image = image_path.resize((384, 384))
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

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

def detect_and_crop_license_plates(image, model):
    """
    ตรวจจับและ crop ป้ายทะเบียนจากภาพโดยใช้ YOLO model (1 รอบ)
    """
    if image is None:
        return [], []
        
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb_image)

    cropped_images = []
    boxes = []

    if results and results[0].boxes:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        for box in boxes:
            x1, y1, x2, y2 = box
            cropped = image[y1:y2, x1:x2]
            cropped_images.append(cropped)

    return cropped_images, boxes

def process_frame(frame, detection_model, processor, model, device):
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
        
        if second_crop:
            final_crop = second_crop[0]
            license_text = check_img(processor, model, device, final_crop)
        else:
            license_text = check_img(processor, model, device, best_plate)
            
    return annotated_frame, license_text, bool(cropped_plates)

def main():
    st.title("🚗 Thai License Plate Recognition")
    
    # Load models
    with st.spinner("Loading models..."):
        processor, model, device, detection_model = load_models()
        st.success("Models loaded successfully! Ready to detect license plates.")
    
    # Sidebar options
    st.sidebar.title("Options")
    input_option = st.sidebar.radio(
        "Select input source:",
        ["Webcam", "Upload Image", "Upload Video"]
    )

    # Results placeholders
    col1, col2 = st.columns(2)
    with col1:
        image_placeholder = st.empty()
    with col2:
        result_placeholder = st.empty()
        
    # Create a status indicator
    status_indicator = st.empty()
    
    if input_option == "Webcam":
        status_indicator.info("Starting webcam...")
        
        # Start webcam button
        start_button = st.button("Start Webcam")
        stop_button = st.button("Stop")
        
        if start_button:
            # Access webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not open webcam. Please check your camera connection.")
            else:
                status_indicator.success("Webcam started! Processing frames...")
                
                last_detection_time = 0
                cooldown_period = 1  # 1 second between detections
                last_license_text = None
                
                # Use session_state to control the webcam loop
                st.session_state.stop_webcam = False
                
                while not st.session_state.stop_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every frame but only run detection on cooldown
                    current_time = time.time()
                    if current_time - last_detection_time > cooldown_period:
                        annotated_frame, license_text, plate_found = process_frame(
                            frame, detection_model, processor, model, device
                        )
                        
                        if license_text:
                            last_license_text = license_text
                            last_detection_time = current_time
                    else:
                        # Just add the previous detection results to the frame
                        annotated_frame = frame.copy()
                        if last_license_text:
                            cv2.putText(
                                annotated_frame, 
                                f"License: {last_license_text}", 
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
                    if last_license_text:
                        result_placeholder.markdown(f"""
                        ### Detected License Plate
                        
                        **Text:** {last_license_text}
                        """)
                    
                    # Check if stop was pressed (needs to be implemented as a callback)
                    if stop_button:
                        st.session_state.stop_webcam = True
                        break
                
                cap.release()
                status_indicator.info("Webcam stopped.")
        
        if stop_button:
            st.session_state.stop_webcam = True
            status_indicator.info("Stopping webcam...")
    
    elif input_option == "Upload Image":
        status_indicator.info("Please upload an image.")
        
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Convert to numpy array
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
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
    
    elif input_option == "Upload Video":
        status_indicator.info("Please upload a video file.")
        
        uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
        
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            temp_file = "temp_video.mp4"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.read())
            
            status_indicator.info("Processing video...")
            
            # Open the video file
            cap = cv2.VideoCapture(temp_file)
            
            if not cap.isOpened():
                status_indicator.error("Error opening video file.")
            else:
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
                os.remove(temp_file)
                
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

    # Add some instructions at the bottom
    st.markdown("""
    ### How to use:
    1. Select an input source from the sidebar (Webcam, Upload Image, or Upload Video).
    2. For webcam, press Start and wait for detections.
    3. For images and videos, upload a file and wait for processing.
    
    The system will detect license plates and attempt to read the text.
    """)

if __name__ == "__main__":
    main()