import logging
import asyncio
from llm import BaseLLM

from concurrent import futures
import grpc
import ocr_pb2
import ocr_pb2_grpc
from ocr import BaseOCR



# Coroutines to be invoked when the event loop is shutting down.
_cleanup_coroutines = []
    
class InvoiceOCRServicer(ocr_pb2_grpc.InvoiceOCRServicer):
    def __init__(self):
        super().__init__()
        self.ocr = BaseOCR(text_detection_model_dir="C:\\Users\\chris\\.paddlex\\official_models", text_recognition_model_dir="C:\\Users\\chris\\.paddlex\\official_models")
        self.llm = BaseLLM()
    async def UploadInvoice(self, request: ocr_pb2.UploadInvoiceRequest, context: grpc.aio.ServicerContext) -> ocr_pb2.UploadInvoiceResponse:
        logging.info('Uploading Invoice...')
        invoice_data = await self.ocr.invoke(request.invoice_file_path)
        logging.info('Invoice processed by the OCR...')
        content = await self.llm.predict_invoice(invoice_data)
        print(content.model_dump())
        return ocr_pb2.UploadInvoiceResponse(**content.model_dump())
        
        
async def serve() -> None:
  server = grpc.aio.server()
  ocr_pb2_grpc.add_InvoiceOCRServicer_to_server(InvoiceOCRServicer(), server)
  listen_addr = '[::]:50051'
  server.add_insecure_port(listen_addr)
  logging.info('Starting server on %s', listen_addr)
  await server.start()
  
  # async def server_graceful_shutdown():
  #   logging.info('Starting graceful shutdown...')
  #   # Shuts down the server with 0 seconds of grace period. During the
  #   # grace period, the server won't accept new connections and allow
  #   # existing RPCs to continue within the grace period.
  #   await server.stop(5)
  
  # _cleanup_coroutines.append(server_graceful_shutdown())
  await server.wait_for_termination()
  
  
if __name__ == '__main__':
    asyncio.run(serve())