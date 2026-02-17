import io
from PyPDF2 import PdfReader
from utils.logger import logger
from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional
from schemas.request.predict_lifespan import PredictLifespanRequest
from dotenv import load_dotenv
import json


load_dotenv()
client = genai.Client()


class DataExtractionService:
    def __init__(self, file):
        self.file = file

    async def extract_data(self):
        byte_content = await self.file.read()

        # Create a BytesIO object from the byte content
        pdf_file_obj = io.BytesIO(byte_content)

        reader = PdfReader(pdf_file_obj)
        extracted_text = reader.pages[0].extract_text()
        logger.info(f"Extracted Text: {extracted_text}")
        return extracted_text

    def get_prompt(self, extracted_text: str):
        return f"""
        You are a data extraction expert. Your task is to extract information from the following text and return it in JSON format.
        
        Extracted Text:
        {extracted_text}
        
        Return the extracted information in the JSON format.
        """

    async def create_request_model(self) -> PredictLifespanRequest:
        extracted_text = await self.extract_data()
        prompt = self.get_prompt(extracted_text)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": PredictLifespanRequest.model_json_schema(),
            },
        )
        py_dict = json.loads(response.text)
        logger.info(f"Type of response: {type(py_dict)}")
        logger.info(f"Response: {py_dict}")
        return PredictLifespanRequest(**py_dict)
