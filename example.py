"""
DocuMind Example Usage
This script demonstrates how to use DocuMind with a real document.
"""
import os
import sys
from dotenv import load_dotenv

# Ensure proper environment variables are loaded
load_dotenv()

# Import the DocuMind pipeline
from main import DocuMindPipeline

def main():
    """
    Example usage of DocuMind
    """
    # Check if Google API key is set (required for Gemma 3 API)
    if not os.getenv("GOOGLE_API_KEY") and len(sys.argv) < 2:
        print("Warning: GOOGLE_API_KEY environment variable not set.")
        print("Either set this in a .env file, or use --local flag to use local models.")
        
    # Path to your document
    document_path = "sample_document.pdf"
    
    # Check if the document exists
    if not os.path.exists(document_path):
        print(f"Error: Document not found at {document_path}")
        print("Please place a PDF document named 'sample_document.pdf' in the same directory as this script.")
        return
    
    # Questions to ask
    questions = [
        "สรุปเนื้อหาหลักของเอกสาร",
        "ยอดรวมภาษีมูลค่าเพิ่มคือเท่าไหร่?",
        "มีรายการสินค้าอะไรบ้างในเอกสาร?"
    ]
    
    # Initialize the pipeline
    # Use --gpu flag if you have a compatible GPU
    # Use --local flag if you don't have Google API key
    use_gpu = "--gpu" in sys.argv
    use_local = "--local" in sys.argv
    
    print("Initializing DocuMind pipeline...")
    pipeline = DocuMindPipeline(
        use_gpu=use_gpu,
        use_api_for_gemma=not use_local,
        cache_dir="example_cache"
    )
    
    # Process the document once (This will create cached results for faster question answering)
    print(f"Processing document: {document_path}")
    pipeline.process_document(document_path)
    
    # Answer each question
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        answer = pipeline.answer_question(document_path, question)
        print(f"Answer: {answer}")
        print("-" * 80)

if __name__ == "__main__":
    main()
