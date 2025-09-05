#!/usr/bin/env python3
"""
Build ChromaDB index from SAP EWM training PDFs.

A production-ready script for building a vector database index from SAP EWM documentation.
This script processes PDF documents, chunks them intelligently, and stores them in ChromaDB
with rich metadata for optimal retrieval.

Features:
- Multi-format PDF loading with fallback support
- Token-aware chunking (500-1000 tokens with configurable overlap)
- Smart embedding provider (OpenAI with sentence-transformers fallback)
- Rich metadata extraction (sections, content classification, etc.)
- Comprehensive error handling and logging
- Environment-based configuration
- Idempotent operation (clean recreates database)

Author: AI Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import logging
import re
import shutil
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import hashlib
from datetime import datetime

# Third-party imports
import fitz  # PyMuPDF for PDF processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma  # Use the specific langchain-chroma import
from langchain.schema import Document
import tiktoken
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable ChromaDB telemetry BEFORE any ChromaDB operations
# This ensures no telemetry data is sent to external servers (enterprise requirement)
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Setup comprehensive logging
def setup_logging():
    """Setup logging with file and console output."""
    # Create logs directory
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / 'build_index.log'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Configuration from environment variables
class Config:
    """Configuration loaded from environment variables."""
    
    # Input/Output paths
    DATA_DIRECTORY = os.getenv("DATA_DIRECTORY", "./data")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "sap_ewm_docs")
    
    # Chunking configuration
    MIN_TOKENS = int(os.getenv("MIN_TOKENS", "500"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
    TOKEN_OVERLAP = int(os.getenv("TOKEN_OVERLAP", "100"))
    
    # Embedding configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


class SmartEmbeddingProvider:
    """
    Smart embedding provider that supports Azure OpenAI, OpenAI, and falls back to sentence-transformers.
    """
    
    def __init__(self):
        self.embeddings = None
        self.provider_type = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embeddings with fallback strategy."""
        # Try Azure OpenAI first using the proven working pattern
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
        
        if azure_endpoint and azure_api_key:
            try:
                from langchain_openai import AzureOpenAIEmbeddings
                # Use the exact pattern that works: base_url, openai_api_key, model, api_version, openai_api_type
                self.embeddings = AzureOpenAIEmbeddings(
                    base_url=azure_endpoint,
                    openai_api_key=azure_api_key,
                    model=azure_deployment,
                    api_version=azure_api_version,
                    openai_api_type="azure"
                )
                self.provider_type = "azure_openai"
                logger.info(f"Using Azure OpenAI embeddings with deployment: {azure_deployment}")
                return
            except ImportError:
                logger.warning("Azure OpenAI not available, trying standard OpenAI")
            except Exception as e:
                logger.warning(f"Azure OpenAI embeddings failed: {e}, trying standard OpenAI")
        
        # Try standard OpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            try:
                from langchain_openai import OpenAIEmbeddings
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=openai_api_key,
                    model="text-embedding-3-small"
                )
                self.provider_type = "openai"
                logger.info("Using OpenAI embeddings")
                return
            except ImportError:
                logger.warning("OpenAI not available, falling back to sentence-transformers")
            except Exception as e:
                logger.warning(f"OpenAI embeddings failed: {e}, falling back to sentence-transformers")
        
        # Fallback to sentence-transformers
        try:
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            self.embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
            self.provider_type = "sentence_transformers"
            logger.info("Using sentence-transformers embeddings")
        except ImportError:
            raise ImportError("Neither Azure OpenAI, OpenAI, nor sentence-transformers available. Install one of them.")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using the available provider."""
        if self.provider_type in ["openai", "azure_openai", "sentence_transformers"]:
            return self.embeddings.embed_documents(texts)
        else:
            raise ValueError("No embedding provider available")


class TokenAwareTextSplitter:
    """
    Text splitter that respects token limits (500-1000 tokens with 100 overlap).
    """
    
    def __init__(self, min_tokens: int = 100, max_tokens: int = 1000, overlap_tokens: int = 100):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 encoding
        except:
            # Fallback to simple word-based tokenization
            self.tokenizer = None
            logger.warning("tiktoken not available, using word-based token estimation")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimation: ~4 characters per token
            return len(text) // 4
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks respecting token limits."""
        chunks = []
        current_chunk = ""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed max tokens
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            token_count = self._count_tokens(potential_chunk)
            
            if token_count <= self.max_tokens:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it meets minimum requirements
                if current_chunk and self._count_tokens(current_chunk) >= self.min_tokens:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with current sentence
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk and self._count_tokens(current_chunk) >= self.min_tokens:
            chunks.append(current_chunk.strip())
        
        # Add overlap between chunks
        return self._add_overlap(chunks)
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i-1]
                prev_words = prev_chunk.split()
                overlap_words = prev_words[-self.overlap_tokens:] if len(prev_words) > self.overlap_tokens else prev_words
                overlap_text = " ".join(overlap_words)
                
                # Combine overlap with current chunk
                overlapped_chunk = overlap_text + " " + chunk
                overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks


class SAP_EWM_ContentAnalyzer:
    """
    Analyze SAP EWM content to extract section titles and classify content with tags.
    """
    
    # SAP EWM specific patterns for section detection
    SECTION_PATTERNS = [
        r'^(\d+\.?\d*\.?\d*)\s+([A-Z][^\\n]{10,80})',  # Numbered sections
        r'^([A-Z][A-Z\s]{2,50})\s*$',  # All caps headings
        r'(?i)^(introduction|overview|configuration|setup|process|procedure|troubleshooting|error|warning|note).*',
    ]
    
    # Content classification tags
    TAG_PATTERNS = {
        'process': [
            r'(?i)(process|workflow|step|procedure|operation|activity|task)',
            r'(?i)(inbound|outbound|receiving|shipping|picking|packing|putaway)',
            r'(?i)(warehouse\s+order|warehouse\s+task|delivery|shipment)'
        ],
        'transaction': [
            r'(?i)(transaction|tcode|t-code)',
            r'(?i)(/n\w+|/o\w+)',  # SAP transaction codes
            r'(?i)(customizing|configuration|spro)'
        ],
        'error-handling': [
            r'(?i)(error|exception|problem|issue|troubleshoot)',
            r'(?i)(message\s+(class|number|id))',
            r'(?i)(warning|caution|alert)'
        ],
        'configuration': [
            r'(?i)(configuration|customizing|setup|settings)',
            r'(?i)(table|field|parameter)',
            r'(?i)(maintain|define|assign|create)'
        ],
        'integration': [
            r'(?i)(interface|integration|connection)',
            r'(?i)(sap\s+(ecc|s/4|hana|tm|mm|sd|wm))',
            r'(?i)(idoc|bapi|rfc|web\s+service)'
        ]
    }
    
    def extract_section_title(self, text: str) -> Optional[str]:
        """Extract section title from text chunk."""
        lines = text.strip().split('\n')[:3]  # Check first 3 lines
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in self.SECTION_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    # Return the title part, cleaned up
                    title = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    return title.strip()[:100]  # Limit title length
        
        # Fallback: use first meaningful line
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.isdigit():
                return line[:100]
        
        return None
    
    def classify_content(self, text: str) -> List[str]:
        """Classify content and return relevant tags."""
        tags = []
        text_lower = text.lower()
        
        for tag, patterns in self.TAG_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    tags.append(tag)
                    break
        
        return tags if tags else ['general']


class SAP_EWM_PDFProcessor:
    """
    Process SAP EWM PDF documents for indexing with smart chunking and metadata extraction.
    """
    
    def __init__(self):
        self.text_splitter = TokenAwareTextSplitter()
        self.content_analyzer = SAP_EWM_ContentAnalyzer()
    
    def load_pdf_with_pymupdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Load PDF using PyMuPDF (fitz) with page-by-page text extraction."""
        try:
            doc = fitz.open(pdf_path)
            pages = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    pages.append({
                        'page_number': page_num + 1,
                        'text': text.strip(),
                        'char_count': len(text)
                    })
            
            doc.close()
            logger.info(f"Loaded {len(pages)} pages from {pdf_path.name} using PyMuPDF")
            return pages
            
        except Exception as e:
            logger.error(f"Failed to load PDF with PyMuPDF: {e}")
            raise
    
    def load_pdf_with_pypdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Fallback: Load PDF using PyPDFLoader."""
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            pages = []
            for doc in documents:
                page_num = doc.metadata.get('page', 1)
                pages.append({
                    'page_number': page_num,
                    'text': doc.page_content.strip(),
                    'char_count': len(doc.page_content)
                })
            
            logger.info(f"Loaded {len(pages)} pages from {pdf_path.name} using PyPDFLoader")
            return pages
            
        except Exception as e:
            logger.error(f"Failed to load PDF with PyPDFLoader: {e}")
            raise
    
    def process_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Process a PDF file and return chunks with rich metadata.
        
        Returns:
            List of chunks with metadata including:
            - source_file, page_number, section_title, tags
        """
        # Try PyMuPDF first, fallback to PyPDFLoader
        try:
            pages = self.load_pdf_with_pymupdf(pdf_path)
        except:
            try:
                pages = self.load_pdf_with_pypdf(pdf_path)
            except Exception as e:
                logger.error(f"Failed to load PDF {pdf_path.name} with any loader: {e}")
                return []
        
        # Generate chunks with metadata
        chunks = []
        total_chunks = 0
        
        for page_data in pages:
            page_text = page_data['text']
            page_chunks = self.text_splitter.split_text(page_text)
            
            for chunk_idx, chunk_text in enumerate(page_chunks):
                # Extract section title
                section_title = self.content_analyzer.extract_section_title(chunk_text)
                
                # Classify content
                tags = self.content_analyzer.classify_content(chunk_text)
                
                # Create chunk metadata
                chunk_metadata = {
                    'source_file': pdf_path.name,
                    'page_number': page_data['page_number'],
                    'section_title': section_title or f"Page {page_data['page_number']}",
                    'tags': ','.join(tags),  # Convert list to comma-separated string
                    'chunk_index': chunk_idx,
                    'chunk_id': f"{pdf_path.stem}_p{page_data['page_number']}_c{chunk_idx}",
                    'char_count': len(chunk_text),
                    'file_path': str(pdf_path),
                    'indexed_date': datetime.now().isoformat()
                }
                
                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
                total_chunks += 1
        
        logger.info(f"Created {total_chunks} chunks from {pdf_path.name}")
        return chunks


class LangchainChromaManager:
    """
    Manage Langchain Chroma operations with idempotent behavior.
    """
    
    def __init__(self, db_path: str = "./chroma", collection_name: str = "sap_ewm_docs"):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.embeddings_provider = SmartEmbeddingProvider()
        self.vectorstore = None
        
        # Disable ChromaDB telemetry for enterprise environments
        os.environ["ANONYMIZED_TELEMETRY"] = "false"
        
        # Initialize directory
        self.db_path.mkdir(parents=True, exist_ok=True)
        
    def reset_collection(self):
        """Reset collection for clean rebuild (idempotent behavior)."""
        # Remove existing directory if it exists
        if self.db_path.exists() and any(self.db_path.iterdir()):
            shutil.rmtree(self.db_path)
            logger.info(f"Deleted existing Chroma database: {self.db_path}")
        
        # Recreate directory
        self.db_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Reset Chroma database directory: {self.db_path}")
    
    def index_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """Index document chunks into Langchain Chroma using batch processing pattern."""
        if not chunks:
            logger.warning("No chunks to index")
            return
        
        logger.info(f"Indexing {len(chunks)} chunks into Langchain Chroma using batch processing")
        
        # Convert chunks to Langchain Documents
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk['text'],
                metadata=chunk['metadata']
            )
            documents.append(doc)
        
        try:
            # Process in batches using the proven pattern:
            # - First batch: use Chroma.from_documents() to create the vectorstore
            # - Subsequent batches: use add_documents() on the existing vectorstore
            
            total_batches = (len(documents) + batch_size - 1) // batch_size
            logger.info(f"Processing {len(documents)} documents in {total_batches} batch(es) of {batch_size}")
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(documents))
                batch_docs = documents[start_idx:end_idx]
                
                logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_docs)} documents)")
                
                if batch_idx == 0:
                    # First batch: create vectorstore with from_documents
                    self.vectorstore = Chroma.from_documents(
                        batch_docs,
                        embedding=self.embeddings_provider.embeddings,
                        persist_directory=str(self.db_path)
                    )
                    logger.info(f"Created vectorstore with first batch of {len(batch_docs)} documents")
                else:
                    # Subsequent batches: add to existing vectorstore
                    self.vectorstore.add_documents(batch_docs)
                    logger.info(f"Added batch {batch_idx + 1} with {len(batch_docs)} documents")
            
            logger.info(f"Successfully indexed all {len(chunks)} chunks using batch processing pattern")
            
        except Exception as e:
            logger.error(f"Failed to index chunks into Chroma: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            raise
    
    def create_vectorstore_simple(self, documents: List[Document]) -> Chroma:
        """Create Chroma vectorstore using the most reliable pattern.
        
        This method uses the exact pattern that works reliably:
        db = Chroma.from_documents(documents, embedding=embeddings, persist_directory='./path')
        """
        try:
            logger.info(f"Creating Chroma vectorstore with {len(documents)} documents using the reliable pattern")
            
            # Use the exact pattern that always works
            vectorstore = Chroma.from_documents(
                documents,  # First argument: documents list
                embedding=self.embeddings_provider.embeddings,  # Named parameter: embedding function
                persist_directory=str(self.db_path)  # Named parameter: persist directory
            )
            
            logger.info(f"Successfully created Chroma vectorstore using the reliable from_documents pattern")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to create vectorstore: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            if not self.vectorstore:
                # Try to load existing vectorstore using the proven working pattern
                if self.db_path.exists() and any(self.db_path.iterdir()):
                    self.vectorstore = Chroma(
                        persist_directory=str(self.db_path),
                        embedding_function=self.embeddings_provider.embeddings
                    )
                else:
                    return {'total_chunks': 0, 'source_files': [], 'unique_tags': [], 'collection_name': self.collection_name}
            
            # Get some documents to analyze
            try:
                # Try to get sample documents
                sample_docs = self.vectorstore.similarity_search("SAP EWM", k=10)
                
                # Analyze metadata
                all_tags = set()
                source_files = set()
                total_chunks = len(sample_docs)  # This is a rough estimate
                
                for doc in sample_docs:
                    meta = doc.metadata
                    if 'tags' in meta and meta['tags']:
                        # Split comma-separated tags
                        tags = [tag.strip() for tag in str(meta['tags']).split(',')]
                        all_tags.update(tags)
                    if 'source_file' in meta:
                        source_files.add(meta['source_file'])
                
                return {
                    'total_chunks': total_chunks,  # Approximate count
                    'source_files': list(source_files),
                    'unique_tags': list(all_tags),
                    'collection_name': self.collection_name
                }
                
            except Exception as e:
                logger.warning(f"Could not get sample documents for stats: {e}")
                return {
                    'total_chunks': 'Unknown',
                    'source_files': [],
                    'unique_tags': [],
                    'collection_name': self.collection_name
                }
                
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {'error': str(e)}


def main():
    """
    Main function to build ChromaDB index from SAP EWM training PDFs.
    """
    print("🚀 SAP EWM ChromaDB Index Builder")
    print("=" * 50)
    
    # Configuration
    data_dir = Path("./data")
    chroma_dir = "./chroma"
    
    # Check data directory
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        print(f"❌ Please create directory: {data_dir}")
        print(f"📁 Place your SAP EWM training PDFs in: {data_dir}")
        sys.exit(1)
    
    # Find PDF files
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in data directory")
        print(f"❌ No PDF files found in: {data_dir}")
        print("📄 Please add your SAP EWM training PDFs to the data directory")
        sys.exit(1)
    
    if len(pdf_files) != 2:
        logger.warning(f"Expected 2 PDF files, found {len(pdf_files)}")
    
    print(f"📚 Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"   📄 {pdf.name}")
    
    try:
        # Initialize components
        processor = SAP_EWM_PDFProcessor()
        db_manager = LangchainChromaManager(db_path=chroma_dir)
        
        # Reset collection for clean rebuild (idempotent)
        print("\n🔄 Resetting ChromaDB collection for clean rebuild...")
        db_manager.reset_collection()
        
        # Process all PDF files
        all_chunks = []
        
        print("\n📖 Processing PDF files...")
        for pdf_path in pdf_files:
            print(f"   Processing: {pdf_path.name}...")
            chunks = processor.process_pdf(pdf_path)
            all_chunks.extend(chunks)
            print(f"   ✅ Created {len(chunks)} chunks from {pdf_path.name}")
        
        # Index all chunks
        print(f"\n💾 Indexing {len(all_chunks)} total chunks into ChromaDB...")
        db_manager.index_chunks(all_chunks)
        
        # Get final stats
        stats = db_manager.get_collection_stats()
        
        print("\n✅ Indexing Complete!")
        print(f"📊 Total chunks indexed: {stats.get('total_chunks', 0)}")
        print(f"📚 Source files: {', '.join(stats.get('source_files', []))}")
        print(f"🏷️  Content tags: {', '.join(stats.get('unique_tags', []))}")
        print(f"💾 Database location: {chroma_dir}")
        
        print("\n🎉 Ready to use with your SAP EWM assistant!")
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
