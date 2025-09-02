from paddleocr import PaddleOCR
import json
from typing import List
import os

from pydantic import BaseModel
from enum import Enum
from typing import List, Optional


# ---------- ENUMS ----------

class ProcessingConfidence(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class DocumentLayout(str, Enum):
    standard = "standard"
    complex = "complex"
    damaged = "damaged"


# ---------- BASE MESSAGES ----------

class VendorInformationData(BaseModel):
    text: Optional[str] = None
    confidence: Optional[float] = None


class FinancialDataLineItemData(BaseModel):
    class Description(BaseModel):
        text: Optional[str] = None

    class Quantity(BaseModel):
        text: Optional[str] = None
        numeric_value: Optional[int] = None

    class Amount(BaseModel):
        text: Optional[str] = None
        numeric_value: Optional[float] = None

    description: Optional[Description] = None
    quantity: Optional[Quantity] = None
    amount: Optional[Amount] = None


class ExtractionIssueData(BaseModel):
    issue_type: Optional[str] = None
    description: Optional[str] = None
    affected_fields: List[str] = []
    suggested_action: Optional[str] = None


# ---------- MAIN RESPONSE ----------

class UploadInvoiceResponseDTO(BaseModel):

    class ExtractionMetadata(BaseModel):
        processing_confidence: Optional[ProcessingConfidence] = None
        document_layout: Optional[DocumentLayout] = None
        total_text_elements: Optional[int] = None
        high_confidence_elements: Optional[int] = None

    class VendorInformation(BaseModel):
        company_name: Optional[VendorInformationData] = None
        address: Optional[VendorInformationData] = None

        class Contact(BaseModel):
            phone: Optional[VendorInformationData] = None
            email: Optional[VendorInformationData] = None

        contact: Optional[Contact] = None

    class InvoiceDetails(BaseModel):

        class InvoiceNumber(BaseModel):
            text: Optional[str] = None
            confidence: Optional[float] = None
            context_label: Optional[str] = None

        class InvoiceDate(BaseModel):
            text: Optional[str] = None
            confidence: Optional[float] = None
            original_format: Optional[str] = None

        class DueDate(BaseModel):
            text: Optional[str] = None
            confidence: Optional[float] = None
            context_label: Optional[str] = None

        class FinancialData(BaseModel):

            class TotalAmount(BaseModel):
                text: Optional[str] = None
                numeric_value: Optional[float] = None
                confidence: Optional[float] = None
                context_label: Optional[str] = None

            class Subtotal(BaseModel):
                text: Optional[str] = None
                numeric_value: Optional[float] = None

            class PaymentTerms(BaseModel):

                class EarlyPayDiscount(BaseModel):
                    found: Optional[bool] = None
                    text: Optional[str] = None
                    percentage: Optional[float] = None
                    days: Optional[int] = None

                terms_text: Optional[str] = None
                standardized: Optional[str] = None
                confidence: Optional[float] = None
                early_pay_discount: Optional[EarlyPayDiscount] = None

            total_amount: Optional[TotalAmount] = None
            line_items: List[FinancialDataLineItemData] = []
            subtotal: Optional[Subtotal] = None
            payment_terms: Optional[PaymentTerms] = None

        invoice_number: Optional[InvoiceNumber] = None
        invoice_date: Optional[InvoiceDate] = None
        due_date: Optional[DueDate] = None
        financial_data: Optional[FinancialData] = None
        extraction_issues: List[ExtractionIssueData] = []

    extraction_metadata: Optional[ExtractionMetadata] = None
    vendor_information: Optional[VendorInformation] = None
    invoice_details: Optional[InvoiceDetails] = None

  

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