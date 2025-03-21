"""
RAG Agent for vector storage and retrieval.
This agent is responsible for storing structured document data in a vector database
and retrieving relevant information based on user queries.
"""
import os
import json
import logging
from typing import Dict, Any, Union, List, Optional
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGAgent:
    """
    Agent for vector storage and retrieval using LangChain and FAISS.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the RAG agent.
        
        Args:
            embedding_model: Name of the HuggingFace embedding model to use
        """
        self.embedding_model_name = embedding_model
        self.embeddings = None
        self.vector_store = None
        
        logger.info(f"RAG Agent initialized with embedding model: {embedding_model}")
    
    def load_embeddings(self) -> None:
        """
        Load the embedding model.
        """
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def _prepare_documents(self, structured_data: dict) -> List[Document]:
        """
        Convert structured document data into LangChain Document objects.
        
        Args:
            structured_data: Structured document data from Mistral agent
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        try:
            # Convert JSON data to string for embedding purposes
            structured_json_str = json.dumps(structured_data, ensure_ascii=False)
            
            # Create a document with the full JSON for metadata queries
            full_doc = Document(
                page_content=structured_json_str,
                metadata={"type": "full_document", "source": "document"}
            )
            documents.append(full_doc)
            
            # Create individual documents for each page and its components
            for page_key, page_data in structured_data.items():
                page_num = page_key.split('_')[-1]  # Extract page number
                
                # Process metadata if available
                if "metadata" in page_data:
                    metadata_str = json.dumps(page_data["metadata"], ensure_ascii=False)
                    metadata_doc = Document(
                        page_content=metadata_str,
                        metadata={
                            "type": "metadata", 
                            "page": page_num,
                            "source": "document"
                        }
                    )
                    documents.append(metadata_doc)
                
                # Process text blocks if available
                if "text" in page_data and isinstance(page_data["text"], list):
                    for i, text_block in enumerate(page_data["text"]):
                        if "content" in text_block:
                            text_doc = Document(
                                page_content=text_block["content"],
                                metadata={
                                    "type": "text",
                                    "page": page_num,
                                    "position": text_block.get("position", "unknown"),
                                    "index": i,
                                    "source": "document"
                                }
                            )
                            documents.append(text_doc)
                
                # Process tables if available
                if "tables" in page_data and isinstance(page_data["tables"], list):
                    for i, table in enumerate(page_data["tables"]):
                        if "headers" in table and "rows" in table:
                            # Convert table to text representation
                            table_text = "Table Headers: " + ", ".join(table["headers"]) + "\n"
                            for row in table["rows"]:
                                table_text += "Row: " + ", ".join(row) + "\n"
                                
                            table_doc = Document(
                                page_content=table_text,
                                metadata={
                                    "type": "table",
                                    "page": page_num,
                                    "position": table.get("position", "unknown"),
                                    "index": i,
                                    "source": "document"
                                }
                            )
                            documents.append(table_doc)
                
                # Process lists if available
                if "lists" in page_data and isinstance(page_data["lists"], list):
                    for i, list_item in enumerate(page_data["lists"]):
                        if "items" in list_item:
                            list_text = "List Items: " + ", ".join(list_item["items"])
                            list_doc = Document(
                                page_content=list_text,
                                metadata={
                                    "type": "list",
                                    "page": page_num,
                                    "position": list_item.get("position", "unknown"),
                                    "index": i,
                                    "source": "document"
                                }
                            )
                            documents.append(list_doc)
            
            logger.info(f"Created {len(documents)} document chunks from structured data")
            return documents
            
        except Exception as e:
            logger.error(f"Error preparing documents for RAG: {str(e)}")
            # Return at least one document with error information
            error_doc = Document(
                page_content=f"Error processing document: {str(e)}",
                metadata={"type": "error", "source": "error"}
            )
            return [error_doc]
    
    def index_document(self, structured_data: dict) -> None:
        """
        Index a structured document in the vector store.
        
        Args:
            structured_data: Structured document data from Mistral agent
        """
        try:
            # Ensure embeddings are loaded
            if self.embeddings is None:
                self.load_embeddings()
            
            # Prepare document chunks
            documents = self._prepare_documents(structured_data)
            
            # Create or update vector store
            if self.vector_store is None:
                logger.info("Creating new vector store")
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                logger.info("Adding to existing vector store")
                self.vector_store.add_documents(documents)
                
            logger.info("Document indexed successfully")
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            raise
    
    def save_index(self, index_path: str) -> None:
        """
        Save the vector store index to disk.
        
        Args:
            index_path: Path to save the index
        """
        try:
            if self.vector_store is None:
                raise ValueError("No vector store to save")
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            logger.info(f"Saving vector store to: {index_path}")
            self.vector_store.save_local(index_path)
            logger.info("Vector store saved successfully")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_index(self, index_path: str) -> None:
        """
        Load a vector store index from disk.
        
        Args:
            index_path: Path to the index
        """
        try:
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Index not found at: {index_path}")
                
            # Ensure embeddings are loaded
            if self.embeddings is None:
                self.load_embeddings()
                
            logger.info(f"Loading vector store from: {index_path}")
            self.vector_store = FAISS.load_local(index_path, self.embeddings)
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve relevant documents based on a query.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        try:
            if self.vector_store is None:
                raise ValueError("No vector store available for retrieval")
                
            logger.info(f"Retrieving documents for query: {query}")
            documents = self.vector_store.similarity_search(query, k=k)
            
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            # Return empty list in case of error
            return []
    
    def format_retrieved_content(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a single context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        try:
            if not documents:
                return "No relevant information found."
                
            context_parts = []
            
            for i, doc in enumerate(documents):
                # Format based on document type
                doc_type = doc.metadata.get("type", "unknown")
                page = doc.metadata.get("page", "unknown")
                
                if doc_type == "text":
                    context_parts.append(f"Text (Page {page}): {doc.page_content}")
                elif doc_type == "table":
                    context_parts.append(f"Table (Page {page}):\n{doc.page_content}")
                elif doc_type == "list":
                    context_parts.append(f"List (Page {page}): {doc.page_content}")
                elif doc_type == "metadata":
                    context_parts.append(f"Document Metadata: {doc.page_content}")
                else:
                    # For full document or unknown types
                    context_parts.append(f"Content: {doc.page_content}")
            
            context = "\n\n".join(context_parts)
            return context
        except Exception as e:
            logger.error(f"Error formatting context: {str(e)}")
            return f"Error formatting context: {str(e)}"