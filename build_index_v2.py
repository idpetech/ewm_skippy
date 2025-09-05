#!/usr/bin/env python3
"""
SAP EWM ChromaDB Index Builder V2 - Based on Working Azure OpenAI Setup

This script combines the proven Azure OpenAI configuration that works in your environment
with the advanced PDF parsing and chunking logic from the EWM Skippy implementation.

Features:
- Uses your exact working Azure OpenAI setup
- Advanced PDF parsing with PyMuPDF and fallback to PyPDFLoader
- Smart token-aware text chunking (500-1000 tokens with overlap)
- SAP EWM content analysis and tagging
- Rich metadata extraction
- Batch processing for reliable indexing
- Enterprise-compliant (telemetry disabled)

Author: AI Development Team
Version: 2.0.0
License: MIT
"""

import os
import sys
import logging
import re
import shutil
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Disable ChromaDB telemetry BEFORE any imports
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Third-party imports - using your exact working pattern
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
import fitz  # PyMuPDF for advanced PDF processing
import tiktoken
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Setup logging
def setup_logging():
    """Setup logging with file and console output."""
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / 'build_index_v2.log'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Your exact working Azure OpenAI configuration
class AzureOpenAIConfig:
    """Azure OpenAI configuration - using your exact working setup"""
    
    # Embedding configuration - your exact working values
    EMBEDDING_MODEL = "text-embedding-ada-002"
    EMBEDDING_ENDPOINT = "https://genaiapimna.jnj.com/openai-embeddings/openai"
    EMBEDDING_API_KEY = "f89d10a91b9d4cc989085a495d695eb3"
    EMBEDDING_DEPLOYMENT = "text-embedding-ada-002" 
    EMBEDDING_API_VERSION = "2022-12-01"
    
    # Processing configuration
    BATCH_SIZE = 5
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 200

class TokenAwareTextSplitter:
    """
    Advanced text splitter that respects token limits and creates meaningful chunks
    for SAP EWM content.
    """
    
    def __init__(self, min_tokens: int = 500, max_tokens: int = 1000, overlap_tokens: int = 100):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 encoding
            logger.info("Using tiktoken for accurate token counting")
        except:
            self.tokenizer = None
            logger.warning("tiktoken not available, using character-based estimation")
    
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
        
        # Split by sentences first for better semantic boundaries
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
        
        # Add overlap between chunks for better context continuity
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
        r'^(\d+\.?\d*\.?\d*)\s+([A-Z][^\n]{10,80})',  # Numbered sections
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

class AdvancedPDFProcessor:
    """
    Advanced PDF processor combining PyMuPDF and PyPDFLoader with smart chunking.
    """
    
    def __init__(self):
        self.text_splitter = TokenAwareTextSplitter()
        self.content_analyzer = SAP_EWM_ContentAnalyzer()
    
    def load_pdf_with_pymupdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Load PDF using PyMuPDF (fitz) with advanced text extraction."""
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
    
    def process_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Process a PDF file and return Langchain Documents with rich metadata.
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
        documents = []
        
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
                    'source': str(pdf_path),  # Use 'source' key for compatibility
                    'page': page_data['page_number'],
                    'section_title': section_title or f"Page {page_data['page_number']}",
                    'tags': ','.join(tags),
                    'chunk_index': chunk_idx,
                    'chunk_id': f"{pdf_path.stem}_p{page_data['page_number']}_c{chunk_idx}",
                    'char_count': len(chunk_text),
                    'indexed_date': datetime.now().isoformat()
                }
                
                # Create Langchain Document
                doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
                documents.append(doc)
        
        logger.info(f"Created {len(documents)} chunks from {pdf_path.name}")
        return documents

def main():
    """
    Main function to build ChromaDB index using your proven Azure OpenAI setup
    with advanced PDF processing.
    """
    print("🚀 SAP EWM ChromaDB Index Builder V2 (Azure OpenAI)")
    print("=" * 60)
    
    # Configuration
    data_dir = Path("./data")
    db_dir = "./data/eWMDB"
    
    # Check data directory
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        print(f"❌ Please create directory: {data_dir}")
        sys.exit(1)
    
    # Find PDF files
    pdf_files = list(data_dir.glob("**/*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in data directory")
        print(f"❌ No PDF files found in: {data_dir}")
        sys.exit(1)
    
    print(f"📚 Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"   📄 {pdf.name}")
    
    try:
        # Initialize Azure OpenAI embeddings - your exact working configuration
        print("\n🔧 Initializing Azure OpenAI embeddings...")
        embeddings = AzureOpenAIEmbeddings(
            base_url=AzureOpenAIConfig.EMBEDDING_ENDPOINT,
            openai_api_key=AzureOpenAIConfig.EMBEDDING_API_KEY,
            api_version=AzureOpenAIConfig.EMBEDDING_API_VERSION,
            model=AzureOpenAIConfig.EMBEDDING_DEPLOYMENT,
            openai_api_type="azure"
        )
        print(f"✅ Azure OpenAI embeddings initialized: {embeddings}")
        
        # Initialize PDF processor
        pdf_processor = AdvancedPDFProcessor()
        
        # Clean existing database
        if Path(db_dir).exists():
            shutil.rmtree(db_dir)
            print(f"🔄 Cleaned existing database: {db_dir}")
        
        # Process all PDF files with advanced parsing
        print(f"\n📖 Processing PDF files with advanced parsing...")
        all_documents = []
        
        for pdf_path in pdf_files:
            print(f"   Processing: {pdf_path.name}...")
            docs = pdf_processor.process_pdf(pdf_path)
            all_documents.extend(docs)
            print(f"   ✅ Created {len(docs)} chunks from {pdf_path.name}")
        
        print(f"\n💾 Total documents to index: {len(all_documents)}")
        
        # Batch processing using your exact working pattern
        batch_size = AzureOpenAIConfig.BATCH_SIZE
        batches = [all_documents[i:i+batch_size] for i in range(0, len(all_documents), batch_size)]
        
        print(f"🔄 Processing {len(all_documents)} documents in {len(batches)} batch(es) of {batch_size}")
        
        if batches:
            start_time = time.time()
            
            # First batch: create the database
            print(f"   Creating database with first batch ({len(batches[0])} documents)...")
            db = Chroma.from_documents(
                batches[0], 
                embedding=embeddings,
                persist_directory=db_dir
            )
            print(f"   ✅ Database created with first batch")
            
            # Subsequent batches: add documents
            for i, batch in enumerate(batches[1:], 1):
                print(f"   Adding batch {i + 1}/{len(batches)} ({len(batch)} documents)...")
                db.add_documents(batch)
                print(f"   ✅ Added batch {i + 1}")
                
                # Small delay to avoid rate limits
                time.sleep(0.5)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            print(f"\n✅ Indexing Complete!")
            print(f"📊 Total documents indexed: {len(all_documents)}")
            print(f"⏱️  Processing time: {elapsed_time:.2f} seconds")
            print(f"💾 Database location: {db_dir}")
            
            # Test the database
            print(f"\n🔍 Testing database with sample query...")
            test_db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
            test_query = "What are some best practices for eWM implementations"
            test_docs = test_db.similarity_search(test_query, k=3)
            
            print(f"   Found {len(test_docs)} relevant documents for test query")
            for i, doc in enumerate(test_docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                section = doc.metadata.get('section_title', 'Unknown')
                print(f"   {i}. Source: {Path(source).name}, Page: {page}, Section: {section}")
            
            print(f"\n🎉 SAP EWM ChromaDB index is ready!")
            print(f"📖 You can now use this database for your SAP EWM assistant")
            
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
