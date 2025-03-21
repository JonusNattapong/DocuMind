"""
Gemma Agent for analysis and question answering.
This agent is responsible for understanding context from the RAG agent and
answering user questions using the Gemma 3 model.
"""
import os
import logging
from typing import Dict, Any, Union, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GemmaAgent:
    """
    Agent for analysis and question answering using Gemma 3 model.
    """
    
    def __init__(self, use_api: bool = True, local_model: str = "google/gemma-2b-it", use_gpu: bool = False, quantize: bool = True):
        """
        Initialize the Gemma agent.
        
        Args:
            use_api: Whether to use the Google API (True) or a local model (False)
            local_model: Local model name if not using API
            use_gpu: Whether to use GPU for local model inference if available
            quantize: Whether to use quantization for local model to reduce memory usage
        """
        self.use_api = use_api
        self.local_model_name = local_model
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.quantize = quantize
        self.model = None
        self.tokenizer = None
        
        # Configure Google API if using it
        if self.use_api:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("No Google API key found in environment variables. Set GOOGLE_API_KEY in .env file.")
            else:
                genai.configure(api_key=api_key)
                
        logger.info(f"Gemma Agent initialized, using API: {use_api}")
        if not use_api:
            logger.info(f"Local model: {local_model} on device: {self.device}")
    
    def load_local_model(self) -> None:
        """
        Load the local Gemma model and tokenizer.
        Only used if use_api is False.
        """
        if self.use_api:
            logger.info("Using API, no need to load local model")
            return
            
        try:
            logger.info(f"Loading local Gemma model: {self.local_model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_name)
            
            # Load model with quantization if enabled
            if self.quantize and self.device == "cuda":
                logger.info("Using 8-bit quantization to save memory")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.local_model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    load_in_8bit=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.local_model_name
                ).to(self.device)
                
            logger.info("Local Gemma model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading local Gemma model: {str(e)}")
            raise
    
    def _prepare_prompt(self, context: str, question: str, answer_language: str = "Thai") -> str:
        """
        Prepare the prompt for the Gemma model.
        
        Args:
            context: Context information from RAG
            question: User's question
            answer_language: Language to answer in
            
        Returns:
            Formatted prompt
        """
        prompt = f"""
You are a helpful, accurate, and concise assistant. You will be given context about a document and a question to answer.
Only answer based on the information in the context. If the context doesn't contain the relevant information, say 'I don't have enough information to answer this question'.

Context:
{context}

Question:
{question}

Answer in {answer_language} language:
"""
        return prompt
    
    def _query_api(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Query the Gemma model through Google's API.
        
        Args:
            prompt: Formatted prompt
            temperature: Model temperature
            
        Returns:
            Model response
        """
        try:
            # Check if API key is configured
            if not os.getenv("GOOGLE_API_KEY"):
                return "Error: Google API key not configured. Please set GOOGLE_API_KEY in .env file."
                
            logger.info("Querying Gemma 3 model via API")
            
            # Initialize the model
            model = genai.GenerativeModel('gemma-3-27b')
            
            # Generate response
            response = model.generate_content(
                prompt, 
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=1024,
                    top_p=0.95,
                    top_k=40
                )
            )
            
            # Extract and return text
            return response.text
            
        except Exception as e:
            logger.error(f"Error querying Gemma API: {str(e)}")
            return f"Error querying Gemma API: {str(e)}"
    
    def _query_local_model(self, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.3) -> str:
        """
        Query the local Gemma model.
        
        Args:
            prompt: Formatted prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Model temperature
            
        Returns:
            Model response
        """
        try:
            # Load model if not loaded yet
            if self.model is None or self.tokenizer is None:
                self.load_local_model()
                
            logger.info("Querying local Gemma model")
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            
            # Generate output
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    temperature=temperature,
                    do_sample=temperature > 0
                )
                
            # Decode output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the response
            if response.startswith(prompt):
                response = response[len(prompt):]
                
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error querying local Gemma model: {str(e)}")
            return f"Error querying local Gemma model: {str(e)}"
    
    def answer_question(self, context: str, question: str, answer_language: str = "Thai") -> str:
        """
        Answer a question based on the provided context.
        
        Args:
            context: Context information from RAG
            question: User's question
            answer_language: Language to answer in
            
        Returns:
            Answer to the question
        """
        try:
            # Prepare prompt
            prompt = self._prepare_prompt(context, question, answer_language)
            
            # Query model (API or local)
            if self.use_api:
                answer = self._query_api(prompt)
            else:
                answer = self._query_local_model(prompt)
                
            logger.info("Question answered successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"I encountered an error while trying to answer your question: {str(e)}"