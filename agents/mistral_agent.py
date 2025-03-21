"""
Mistral Agent for structuring document data.
This agent is responsible for converting raw document tags into structured JSON
using the Mistral API.
"""
import os
import json
import logging
from typing import Dict, Any, Union, List, Optional
import requests
import time
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MistralAgent:
    """
    Agent for structuring document data using Mistral API.
    """
    
    def __init__(self, api_key: str = None, model: str = "mistral-large-latest"):
        """
        Initialize the Mistral agent.
        
        Args:
            api_key: Mistral API key (defaults to MISTRAL_API_KEY environment variable)
            model: Mistral model to use
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model = model
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        
        if not self.api_key:
            logger.warning("No Mistral API key provided. Please set MISTRAL_API_KEY environment variable.")
        
        logger.info(f"Mistral Agent initialized with model: {self.model}")
    
    def _validate_api_key(self) -> None:
        """
        Check if API key is set.
        """
        if not self.api_key:
            raise ValueError("Mistral API key is not set. Set MISTRAL_API_KEY environment variable.")
    
    def _prepare_prompt(self, doctags: str) -> List[Dict[str, str]]:
        """
        Prepare the prompt for the Mistral API.
        
        Args:
            doctags: Raw document tags from SmolDocling
            
        Returns:
            Formatted messages for Mistral API
        """
        # Example of few-shot learning with explicit JSON structure examples
        system_prompt = """You are an expert at converting document tags and OCR results into structured JSON format.
Follow these rules strictly:
1. Always output valid JSON with proper syntax
2. Organize content by pages as top-level keys (numbered from 1)
3. Under each page, categorize elements into:
   - 'text': Text blocks with 'content' and 'position' keys
   - 'tables': Tables with 'headers', 'rows', and 'position' keys
   - 'lists': Lists with 'items' and 'position' keys
   - 'metadata': Document metadata like title, author, date
4. If a particular element type is not present on a page, don't include its key
5. Format the JSON with proper indentation and syntax"""

        example_input = """Title: Invoice #12345
Customer: ABC Corp
Date: 2023-01-15
Item | Price | Qty | Total
Product A | $100 | 2 | $200
Product B | $50 | 3 | $150
Subtotal: $350
Tax: $35
Total: $385"""

        example_output = """{
  "page_1": {
    "metadata": {
      "title": "Invoice #12345",
      "customer": "ABC Corp",
      "date": "2023-01-15"
    },
    "tables": [
      {
        "headers": ["Item", "Price", "Qty", "Total"],
        "rows": [
          ["Product A", "$100", "2", "$200"],
          ["Product B", "$50", "3", "$150"]
        ],
        "position": "middle"
      }
    ],
    "text": [
      {
        "content": "Subtotal: $350",
        "position": "bottom"
      },
      {
        "content": "Tax: $35",
        "position": "bottom"
      },
      {
        "content": "Total: $385",
        "position": "bottom"
      }
    ]
  }
}"""

        user_prompt = f"""Please convert the following document content into a structured JSON format:

```
{doctags}
```

Remember to ensure the JSON output is properly formatted and valid."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Convert this document text to JSON:\n{example_input}"},
            {"role": "assistant", "content": example_output},
            {"role": "user", "content": user_prompt}
        ]
        
        return messages
    
    def process_doctags(self, doctags: str, max_retries: int = 3, retry_delay: int = 2) -> dict:
        """
        Process document tags and convert into structured JSON using Mistral API.
        
        Args:
            doctags: Document tags from SmolDocling
            max_retries: Maximum number of API call retries
            retry_delay: Delay between retries in seconds
            
        Returns:
            Structured JSON data
        """
        self._validate_api_key()
        
        messages = self._prepare_prompt(doctags)
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4000
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Calling Mistral API (attempt {attempt+1}/{max_retries})")
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                # Extract the response content
                content = response_data["choices"][0]["message"]["content"]
                
                # Extract JSON from the response
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|```([\s\S]*?)```|({[\s\S]*})', content)
                
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2) or json_match.group(3)
                    
                    # Parse the JSON
                    try:
                        structured_data = json.loads(json_str)
                        logger.info("Successfully processed document tags into structured JSON")
                        return structured_data
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in API response: {str(e)}")
                        logger.debug(f"JSON string attempted to parse: {json_str}")
                else:
                    # If no JSON pattern was found, try to parse the entire content
                    try:
                        structured_data = json.loads(content)
                        logger.info("Successfully processed document tags into structured JSON")
                        return structured_data
                    except json.JSONDecodeError:
                        logger.error("No valid JSON found in API response")
                
                # If we get here, JSON parsing failed
                if attempt < max_retries - 1:
                    logger.warning(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API request error: {str(e)}")
                if attempt < max_retries - 1:
                    logger.warning(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
        # If all retries failed, return an empty dict
        logger.error(f"Failed to process document tags after {max_retries} attempts")
        return {}
    
    def process_document(self, doctags: str) -> dict:
        """
        Process a document's doctags and convert to structured JSON.
        
        Args:
            doctags: Document tags from SmolDocling
            
        Returns:
            Structured JSON data
        """
        logger.info("Processing document with Mistral API")
        return self.process_doctags(doctags)