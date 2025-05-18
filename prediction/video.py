from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2
import re
from itertools import groupby
from ultralytics import YOLO
import os
import numpy as np

# Load model and processor
model_path = "./th_character_process_v4"
processor = TrOCRProcessor.from_pretrained(model_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(image_path):
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

def check_img(img):
        if img is None:
            print("Error: ไม่สามารถโหลดภาพได้ (img is None)")
            return

        # เช็คว่า img เป็น numpy array หรือไม่

        if not isinstance(img, np.ndarray):
            print("Error: img ไม่ใช่ numpy array")
            return
        
        img = cv2.resize(img, (348, 348))  # Resize ขนาดภาพ
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilation = cv2.dilate(thresh, kernel, iterations=1)  # ใช้ edges แทน thresh

        erosion = cv2.erode(dilation, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cropped_images = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cropped = erosion[y:y+h, x:x+w]  # Crop ภาพ
            """ height, width = cropped.shape
            print(height, width)
            cv2.imshow("Cropped Contour", cropped)
            cv2.waitKey(0)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            white_pixels = cv2.countNonZero(cropped)
            black_pixels = total_pixels - white_pixels
            print(f"จำนวนพิกเซลสีขาว: {white_pixels}")
            print(f"จำนวนพิกเซลสีดำ: {black_pixels}") """
            
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
                """ white_pixels = cv2.countNonZero(cropped)
                total_pixels = thresh.shape[0] * thresh.shape[1]
                black_pixels = total_pixels - white_pixels

                print(f"จำนวนพิกเซลสีขาว: {white_pixels}")
                print(f"จำนวนพิกเซลสีดำ: {black_pixels}")
                print(height, width) """
                """ cv2.imshow("Cropped Contour", cropped)
                cv2.waitKey(0) """
                passed_img.append(cropped)
        
        word = []
        word_no_clean = []
        for img in passed_img:
            if len(img.shape) == 2: 
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(img)

            result = predict(pil_img)
            word_no_clean.append(result)
            cleaned = clean_and_filter(result)
            if cleaned:
                word.append(cleaned)
        print(50 * "-")
        print("License plate:", "".join(word))
        print(50 * "-")
        
def detect_and_crop_license_plates(image, model_path='../detection/model/best.pt'):
    """
    ตรวจจับและ crop ป้ายทะเบียนจากภาพโดยใช้ YOLO model (1 รอบ)
    """
    try:
        model = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"ไม่สามารถโหลด YOLO model ได้: {e}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb_image)

    cropped_images = []

    if results and results[0].boxes:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        for box in boxes:
            x1, y1, x2, y2 = box
            cropped = image[y1:y2, x1:x2]
            cropped_images.append(cropped)

    return cropped_images



def process_video_and_crop(detection_model, video_path, output_crop_dir):
    """ประมวลผลวิดีโอ ตรวจจับป้ายทะเบียน Crop ภาพแรกที่เจอ อ่าน OCR ด้วย Tesseract และหยุด."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("ไม่สามารถเปิดวิดีโอได้")

        frame_count = 0
        plate_cropped = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detection_model.predict(frame_rgb)

            if results and results[0].boxes and not plate_cropped:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()

                cropped_images = []
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    confidence = confidences[i]

                    if confidence:
                        # Crop ป้ายทะเบียน
                        cropped_plate = frame[y1:y2, x1:x2]
                        cropped_images.append(cropped_plate)
                        
                        plate_cropped = True

                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        """ cv2.imshow('First Plate Detected', frame)
                        cv2.waitKey(0)
                        cap.release()
                        cv2.destroyAllWindows() """
                        return cropped_images


            cv2.imshow('Video with Plate Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        
        return []

    except IOError as e:
        print(f"เกิดข้อผิดพลาดในการเปิดวิดีโอ: {e}")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการประมวลผลวิดีโอ: {e}")


if __name__ == "__main__":
    detection_model_path = "../detection/model/best.pt"
    video_path = "./video/KhoLo1106.mp4"
    video_list = os.listdir("./video")
    dir_path = "./video/"
    
    try:
        detection_model = YOLO(detection_model_path)
        detection_model.eval()
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลด Model ตรวจจับ: {e}")
        exit()

    for video in video_list:
        full_video_path = dir_path + video 
        
        first_crop_images = process_video_and_crop(detection_model, full_video_path, None)
        
        final_cropped_images = []
        
        for crop_img in first_crop_images:
            second_crop = detect_and_crop_license_plates(crop_img, detection_model_path)
            final_cropped_images.extend(second_crop)

        if final_cropped_images:
            for idx, img in enumerate(final_cropped_images):
                idx = idx + 1
                if img is not None and isinstance(img, np.ndarray):
                    if idx != len(final_cropped_images):
                        continue  
                    check_img(img)
                    cv2.destroyAllWindows()
                else:
                    print(f"พบภาพว่าง (None) หรือชนิดไม่ถูกต้อง ในลิสต์ index {idx}")
        else:
            print("ไม่พบป้ายทะเบียนในวิดีโอ หรือมีบางอย่างผิดพลาด")