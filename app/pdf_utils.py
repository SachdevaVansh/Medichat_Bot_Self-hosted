from pypdf import  PdfReader
from typing import List, Optional
from io import BytesIO
import re

def clean_text(text: str) -> str:
    """Clean text by removing HTML tags and normalizing whitespace"""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove extra line breaks
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ' '
    for page in reader.pages:
        text = text + page.extract_text() or ''
    
    # Clean the extracted text
    return clean_text(text)