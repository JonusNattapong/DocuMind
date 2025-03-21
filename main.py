"""
DocuMind: Agent-based Document Processing System
This module coordinates the four specialized agents:
1. SmolDocling Agent: OCR and document parsing (local model)
2. Mistral Agent: Data structuring (API-based)
3. RAG Agent: Vector storage and retrieval
4. Gemma Agent: Analysis and question answering
"""
import os
import logging
import argparse
import time
from typing import Dict, Any, Optional, List
import json
from dotenv import load_dotenv

from agents.smol_agent import SmolDoclingAgent
from agents.mistral_agent import MistralAgent
from agents.rag_agent import RAGAgent
from agents.gemma_agent import GemmaAgent
from utils.file_utils import save_json, load_json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("documind.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DocuMindPipeline:
    """
    Main pipeline that coordinates all agents.
    """
    
    def __init__(
        self, 
        use_gpu: bool = False,
        use_api_for_gemma: bool = True,
        cache_dir: str = "cache",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        local_gemma_model: str = "google/gemma-2b-it",
        smoldocling_model_path: str = None,
        mistral_api_key: str = None,
        mistral_model: str = "mistral-large-latest"
    ):
        """
        Initialize the DocuMind pipeline.
        
        Args:
            use_gpu: Whether to use GPU for inference if available
            use_api_for_gemma: Whether to use Google API for Gemma
            cache_dir: Directory to cache intermediate results
            embedding_model: Name of the embedding model for RAG
            local_gemma_model: Local Gemma model if not using API
            smoldocling_model_path: Path to local SmolDocling model
            mistral_api_key: Mistral API key (defaults to env var)
            mistral_model: Mistral model to use
        """
        self.use_gpu = use_gpu
        self.use_api_for_gemma = use_api_for_gemma
        self.cache_dir = cache_dir
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize agents
        logger.info("Initializing agents...")
        
        # Initialize SmolDocling agent with local model path
        self.smol_agent = SmolDoclingAgent(
            use_gpu=use_gpu,
            local_model_path=smoldocling_model_path
        )
        
        # Initialize Mistral agent with API
        self.mistral_agent = MistralAgent(
            api_key=mistral_api_key,
            model=mistral_model
        )
        
        # Initialize RAG agent
        self.rag_agent = RAGAgent(embedding_model=embedding_model)
        
        # Initialize Gemma agent
        self.gemma_agent = GemmaAgent(
            use_api=use_api_for_gemma, 
            local_model=local_gemma_model, 
            use_gpu=use_gpu
        )
        
        logger.info("DocuMind pipeline initialized")
    
    def process_document(self, file_path: str, question: str = None, answer_language: str = "Thai", force_reprocess: bool = False) -> dict:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            question: Optional question to answer
            answer_language: Language to answer in
            force_reprocess: Whether to force reprocessing even if cached results exist
            
        Returns:
            Results dictionary with all intermediate outputs and final answer if a question was provided
        """
        results = {}
        
        # Generate cache file paths
        doc_id = os.path.basename(file_path).replace(".", "_")
        doctags_cache = os.path.join(self.cache_dir, f"{doc_id}_doctags.txt")
        json_cache = os.path.join(self.cache_dir, f"{doc_id}_structured.json")
        index_cache = os.path.join(self.cache_dir, f"{doc_id}_index")
        
        # Step 1: OCR and document parsing with SmolDocling
        if force_reprocess or not os.path.exists(doctags_cache):
            logger.info(f"Step 1: Processing document with SmolDocling: {file_path}")
            start_time = time.time()
            
            doctags = self.smol_agent.process_document(file_path)
            
            # Cache the results
            with open(doctags_cache, 'w', encoding='utf-8') as f:
                f.write(doctags)
                
            logger.info(f"SmolDocling processing completed in {time.time() - start_time:.2f} seconds")
        else:
            logger.info(f"Using cached SmolDocling results: {doctags_cache}")
            with open(doctags_cache, 'r', encoding='utf-8') as f:
                doctags = f.read()
                
        results['doctags'] = doctags
        
        # Step 2: Data structuring with Mistral
        if force_reprocess or not os.path.exists(json_cache):
            logger.info("Step 2: Structuring data with Mistral")
            start_time = time.time()
            
            structured_data = self.mistral_agent.process_document(doctags)
            
            # Cache the results
            save_json(structured_data, json_cache)
            
            logger.info(f"Mistral processing completed in {time.time() - start_time:.2f} seconds")
        else:
            logger.info(f"Using cached Mistral results: {json_cache}")
            structured_data = load_json(json_cache)
            
        results['structured_data'] = structured_data
        
        # Step 3: Vector storage with RAG
        if force_reprocess or not os.path.exists(index_cache):
            logger.info("Step 3: Indexing document with RAG")
            start_time = time.time()
            
            self.rag_agent.index_document(structured_data)
            
            # Cache the index
            self.rag_agent.save_index(index_cache)
            
            logger.info(f"RAG indexing completed in {time.time() - start_time:.2f} seconds")
        else:
            logger.info(f"Using cached RAG index: {index_cache}")
            self.rag_agent.load_index(index_cache)
            
        # Step 4: If a question is provided, retrieve relevant information and answer
        if question:
            logger.info(f"Step 4: Retrieving relevant information for: {question}")
            start_time = time.time()
            
            # Retrieve relevant documents
            documents = self.rag_agent.retrieve(question)
            
            # Format retrieved content
            context = self.rag_agent.format_retrieved_content(documents)
            
            logger.info(f"RAG retrieval completed in {time.time() - start_time:.2f} seconds")
            
            # Answer the question with Gemma
            logger.info("Step 5: Answering question with Gemma")
            start_time = time.time()
            
            answer = self.gemma_agent.answer_question(context, question, answer_language)
            
            logger.info(f"Gemma processing completed in {time.time() - start_time:.2f} seconds")
            
            results['context'] = context
            results['answer'] = answer
            
        return results
    
    def answer_question(self, file_path: str, question: str, answer_language: str = "Thai", force_reprocess: bool = False) -> str:
        """
        Process a document and answer a question.
        
        Args:
            file_path: Path to the document file
            question: Question to answer
            answer_language: Language to answer in
            force_reprocess: Whether to force reprocessing even if cached results exist
            
        Returns:
            Answer to the question
        """
        results = self.process_document(file_path, question, answer_language, force_reprocess)
        return results.get('answer', "No answer was generated.")

def main():
    """
    Main entry point for command-line usage.
    """
    parser = argparse.ArgumentParser(description="DocuMind: Agent-based Document Processing System")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to document file")
    parser.add_argument("--question", "-q", type=str, help="Question to answer")
    parser.add_argument("--language", "-l", type=str, default="Thai", help="Language to answer in")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if cached results exist")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference if available")
    parser.add_argument("--local", action="store_true", help="Use local model for Gemma instead of API")
    parser.add_argument("--smoldocling-model", type=str, help="Path to local SmolDocling model")
    parser.add_argument("--mistral-api-key", type=str, help="Mistral API key")
    parser.add_argument("--mistral-model", type=str, default="mistral-large-latest", help="Mistral model to use")
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DocuMindPipeline(
        use_gpu=args.gpu,
        use_api_for_gemma=not args.local,
        smoldocling_model_path=args.smoldocling_model,
        mistral_api_key=args.mistral_api_key,
        mistral_model=args.mistral_model
    )
    
    # Process document
    if args.question:
        answer = pipeline.answer_question(
            args.file, 
            args.question, 
            args.language, 
            args.force
        )
        print(f"\nQuestion: {args.question}")
        print(f"Answer: {answer}")
    else:
        results = pipeline.process_document(args.file, force_reprocess=args.force)
        print(f"\nDocument processed successfully: {args.file}")
        print(f"Results saved to cache directory")

if __name__ == "__main__":
    main()