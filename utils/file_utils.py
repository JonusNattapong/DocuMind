"""
Utilities for file operations and document handling.
"""
import os
from typing import Optional, List
import pypdf
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_pdf_text(file_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.lower().endswith('.pdf'):
        raise ValueError(f"File is not a PDF: {file_path}")
    
    try:
        logger.info(f"Reading PDF text from: {file_path}")
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        if not text.strip():
            logger.warning("No text extracted from PDF. The PDF might be scanned or contain images.")
            return ""
            
        return text
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise

def pdf_to_images(file_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Convert PDF pages to images.
    
    Args:
        file_path: Path to the PDF file
        dpi: DPI for the output images
        
    Returns:
        List of PIL Image objects
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        logger.info(f"Converting PDF to images: {file_path}")
        images = convert_from_path(file_path, dpi=dpi)
        return images
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        raise

def ocr_from_image(image: Image.Image) -> str:
    """
    Perform OCR on an image using pytesseract.
    
    Args:
        image: PIL Image object
        
    Returns:
        Extracted text from the image
    """
    try:
        logger.info("Performing OCR on image")
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logger.error(f"Error during OCR: {str(e)}")
        raise

def ocr_pdf(file_path: str, dpi: int = 300) -> str:
    """
    Extract text from a PDF using OCR if regular text extraction fails.
    
    Args:
        file_path: Path to the PDF file
        dpi: DPI for the image conversion
        
    Returns:
        Extracted text from the PDF
    """
    # First try normal text extraction
    text = read_pdf_text(file_path)
    
    # If no text was extracted, try OCR
    if not text.strip():
        logger.info("No text found in PDF, trying OCR")
        all_text = ""
        images = pdf_to_images(file_path, dpi)
        
        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)}")
            page_text = ocr_from_image(image)
            all_text += page_text + "\n\n"
            
        return all_text
    
    return text

def save_json(data: dict, output_path: str) -> None:
    """
    Save data as JSON file.
    
    Args:
        data: Dictionary data to save
        output_path: Path to save the JSON file
    """
    import json
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON data saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving JSON: {str(e)}")
        raise

def load_json(file_path: str) -> dict:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data
    """
    import json
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"JSON data loaded from: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON: {str(e)}")
        raise