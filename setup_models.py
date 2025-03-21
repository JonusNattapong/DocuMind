"""
DocuMind Model Setup Utility
This script downloads and sets up the models required for DocuMind.
"""
import os
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_smoldocling(model_dir="./models/smoldocling", force=False):
    """
    Download and setup the SmolDocling model locally.
    
    Args:
        model_dir: Directory to save the model
        force: Whether to force re-download if model exists
    """
    try:
        # Check if model directory already exists
        if os.path.exists(os.path.join(model_dir, "config.json")) and not force:
            logger.info(f"SmolDocling model already exists at {model_dir}")
            return True
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        
        logger.info(f"Downloading SmolDocling model to {model_dir}...")
        
        # Import required libraries
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            logger.error("Required libraries not found. Please install with: pip install transformers")
            return False
            
        # Download and save the model
        logger.info("Downloading model weights (this may take a while)...")
        model = AutoModelForVision2Seq.from_pretrained("google/smoldocling-256M-preview")
        
        logger.info("Downloading model processor...")
        processor = AutoProcessor.from_pretrained("google/smoldocling-256M-preview")
        
        # Save locally
        logger.info(f"Saving model to {model_dir}...")
        model.save_pretrained(model_dir)
        processor.save_pretrained(model_dir)
        
        logger.info("SmolDocling model successfully downloaded and saved!")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up SmolDocling model: {str(e)}")
        return False

def main():
    """
    Main function to parse arguments and set up models.
    """
    parser = argparse.ArgumentParser(description="DocuMind Model Setup Utility")
    parser.add_argument("--smoldocling-dir", type=str, default="./models/smoldocling", 
                        help="Directory to save SmolDocling model")
    parser.add_argument("--force", action="store_true", 
                        help="Force re-download even if models exist")
    args = parser.parse_args()
    
    logger.info("Starting DocuMind model setup...")
    
    # Setup SmolDocling
    result = setup_smoldocling(args.smoldocling_dir, args.force)
    
    if result:
        logger.info("\nAll models have been successfully set up!")
        logger.info("You can now use DocuMind with local models.")
    else:
        logger.error("\nModel setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
