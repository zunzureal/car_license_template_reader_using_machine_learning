from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2
import re
from itertools import groupby
from ultralytics import YOLO

processor = TrOCRProcessor.from_pretrained("spykittichai/th-character-ocr")
model = VisionEncoderDecoderModel.from_pretrained("spykittichai/th-character-ocr")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def detect_and_crop_license_plates(image, model_path='./model/best.pt'):
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

def clean_and_filter(text):
    # 1. ลบตัวซ้ำที่ติดกัน เช่น '777777' → '7'
    text = ''.join(k for k, _ in groupby(text))
    # 2. ใช้ regex เพื่อดึงเฉพาะ "ตัวเลข" และ "อักษรไทย"
    filtered = re.findall(r'[ก-๙0-9]', text)
    return ''.join(filtered)


def predict(image_path):
    image = image_path.resize((384, 384))
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

def check_img(img):
    if img is None:
        print("Error: ไม่สามารถโหลดภาพได้")
    else:
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
            """height, width = cropped.shape
            print(height, width)
            cv2.imshow("Cropped Contour", cropped)
            cv2.waitKey(0)"""
            
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
            total_pixels = thresh.shape[0] * thresh.shape[1]
            black_pixels = total_pixels - white_pixels
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

        print(80 * "-")
        print("License plate:", "".join(word))
        print(80 * "-")

if __name__ == "__main__":
    image_path = "./image/img1.png"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"ไม่พบภาพที่ path: {image_path}")

    first_crop_images = detect_and_crop_license_plates(image, "../detection/model/best.pt")

    final_cropped_images = []
    for i, crop_img in enumerate(first_crop_images):
        second_crop = detect_and_crop_license_plates(crop_img, "../detection/model/best.pt")
        final_cropped_images.extend(second_crop)

    for img in final_cropped_images:    
        check_img(img)
