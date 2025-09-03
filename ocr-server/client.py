import json, os
from dotenv import load_dotenv
import ocr_pb2, ocr_pb2_grpc
import grpc

load_dotenv()

addr = os.getenv('GRPC_SERVER_ADDRESS', 'localhost:50051')
creds = grpc.ssl_channel_credentials()
channel = grpc.insecure_channel(addr)
ocr = ocr_pb2_grpc.InvoiceOCRStub(channel)

content = ocr.UploadInvoice(ocr_pb2.UploadInvoiceRequest(user_id=100, invoice_file_path=os.path.join(os.getcwd(), 'datasets', 'batch2-0008.jpg')))
print(content)
# with open(os.path.join(os.getcwd(), 'llm_result', f'{filename_without_ext}_llm_result.json'), 'w', encoding='utf-8') as f:
#     json.dump(content, f)