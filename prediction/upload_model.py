from transformers import VisionEncoderDecoderModel, TrOCRProcessor

model = VisionEncoderDecoderModel.from_pretrained("./th_character_process_v4")
processor = TrOCRProcessor.from_pretrained("./th_character_process_v4")

model.push_to_hub("spykittichai/th-character-ocr")
processor.push_to_hub("spykittichai/th-character-ocr")
