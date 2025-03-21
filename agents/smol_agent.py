"""
SmolDocling Agent for OCR and document parsing.
This agent is responsible for extracting text and structure from documents
using the SmolDocling-256M-preview model loaded from a local directory.
"""
import os
import logging
from typing import Dict, Any, Union, List, Optional
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from functools import lru_cache

from ..utils.file_utils import ocr_pdf, read_pdf_text

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SmolDoclingAgent:
    """
    Agent for document OCR and parsing using locally deployed SmolDocling-256M-preview model.
    """
    
    def __init__(self, use_gpu: bool = False, quantize: bool = True, local_model_path: str = None):
        """
        Initialize the SmolDocling agent.
        
        Args:
            use_gpu: Whether to use GPU for inference if available
            quantize: Whether to use quantization to reduce memory usage
            local_model_path: Path to locally saved model (default: './models/smoldocling')
        """
        self.model_name = "google/smoldocling-256M-preview"
        self.local_model_path = local_model_path or os.path.join('.', 'models', 'smoldocling')
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.quantize = quantize
        self.model = None
        self.processor = None
        
        # Ensure model directory exists
        if not os.path.exists(self.local_model_path):
            os.makedirs(os.path.dirname(self.local_model_path), exist_ok=True)
            logger.warning(f"Local model path does not exist: {self.local_model_path}")
            logger.warning("You must download the model first. See README.md for instructions.")
        
        logger.info(f"SmolDocling Agent initialized on device: {self.device}")
    
    def load_model(self) -> None:
        """
        Load the SmolDocling model and processor from local directory or download if needed.
        """
        try:
            # Check if model exists in local path
            if os.path.exists(os.path.join(self.local_model_path, 'config.json')):
                logger.info(f"Loading SmolDocling model from local path: {self.local_model_path}")
                model_path = self.local_model_path
            else:
                logger.info(f"Local model not found. Downloading from HuggingFace: {self.model_name}")
                model_path = self.model_name
                
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_path)
            
            # Load model with quantization if enabled
            if self.quantize and self.device == "cuda":
                logger.info("Using 8-bit quantization to save memory")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    load_in_8bit=True
                )
            else:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path
                ).to(self.device)
                
            logger.info("SmolDocling model loaded successfully")
            
            # Save model locally if downloaded from HuggingFace
            if model_path == self.model_name and not os.path.exists(os.path.join(self.local_model_path, 'config.json')):
                logger.info(f"Saving model to local path for future use: {self.local_model_path}")
                self.model.save_pretrained(self.local_model_path)
                self.processor.save_pretrained(self.local_model_path)
                
        except Exception as e:
            logger.error(f"Error loading SmolDocling model: {str(e)}")
            raise
    
    @lru_cache(maxsize=5)  # Cache results for the last 5 document chunks
    def process_text_chunk(self, text_chunk: str, max_new_tokens: int = 1000) -> str:
        """
        Process a chunk of text through the SmolDocling model.
        
        Args:
            text_chunk: Text chunk to process
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Processed DocTags output
        """
        if self.model is None or self.processor is None:
            self.load_model()
            
        try:
            # Prepare input
            inputs = self.processor(text_chunk, return_tensors="pt").to(self.device)
            
            # Generate output
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    temperature=0.3,
                    do_sample=False
                )
                
            # Decode output
            doctags = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return doctags
        
        except Exception as e:
            logger.error(f"Error processing text with SmolDocling: {str(e)}")
            # Return the original text if processing fails
            return text_chunk
    
    def process_document(self, file_path: str, chunk_size: int = 500, overlap: int = 100) -> str:
        """
        Process a document with SmolDocling, chunking if necessary due to token limits.
        
        Args:
            file_path: Path to the document
            chunk_size: Size of text chunks to process
            overlap: Overlap between chunks
            
        Returns:
            Combined DocTags output
        """
        # Extract text from document, including OCR if needed
        try:
            logger.info(f"Processing document: {file_path}")
            document_text = ocr_pdf(file_path)
            
            if not document_text.strip():
                raise ValueError(f"Could not extract text from document: {file_path}")
                
            # Split into chunks to avoid token limit
            words = document_text.split()
            chunks = []
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)
                
            logger.info(f"Document split into {len(chunks)} chunks")
            
            # Process each chunk
            doctags_chunks = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                doctags = self.process_text_chunk(chunk)
                doctags_chunks.append(doctags)
                
            # Combine the processed chunks
            combined_doctags = "\n\n".join(doctags_chunks)
            return combined_doctags
            
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            raise