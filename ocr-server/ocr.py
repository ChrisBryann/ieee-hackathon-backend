from paddleocr import PaddleOCR
import json
from typing import List
import ossd
  

class BaseOCR():
  def __init__(self, text_detection_model_dir: str, text_recognition_model_dir: str):
    self.ocr = PaddleOCR(use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False, lang='en',
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                    text_detection_model_dir=f"{text_detection_model_dir}\\PP-OCRv5_mobile_det",
                    text_recognition_model_dir=f"{text_recognition_model_dir}\\PP-OCRv5_mobile_rec",
                    )
    
  async def invoke(self, file_path: str) -> List[List[str]]:
    result = self.ocr.predict(file_path)

    for res in result:
        texts = res.get('rec_texts', [])
        boxes = res.get('rec_boxes', [])
        scores = res.get('rec_scores', [])
        query = []
        for i, (text, score) in enumerate(zip(texts, scores)):
            x1, y1, x2, y2 = boxes[i]
            # print(f'Text: {text} Score: {score} x1: {x1} y1: {y1} x2: {x2} y2: {y2}')
            query.append([text, score, x1, y1, x2, y2])
            
    return query