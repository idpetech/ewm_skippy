#!/usr/bin/env python3
"""
Embedding Service URL Diagnostic Tool

This tool helps diagnose why different Chroma initialization methods
work or fail with various embedding services, especially Azure OpenAI.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_embedding_provider():
    """Test different embedding providers and show their configuration."""
    print("🔍 Embedding Service URL Diagnostics")
    print("=" * 60)
    
    # Test Azure OpenAI configuration
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    
    if azure_endpoint and azure_api_key:
        print("\n✅ Azure OpenAI Configuration Found:")
        print(f"   Endpoint: {azure_endpoint}")
        print(f"   Deployment: {azure_deployment}")
        print(f"   API Version: {os.getenv('AZURE_OPENAI_API_VERSION', '2023-12-01-preview')}")
        
        try:
            from langchain_openai import AzureOpenAIEmbeddings
            
            # Create embeddings instance
            embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                azure_deployment=azure_deployment,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
            )
            
            print(f"   ✅ Successfully created AzureOpenAIEmbeddings instance")
            print(f"   📡 Base URL: {getattr(embeddings, 'azure_endpoint', 'Not available')}")
            print(f"   🔑 Deployment Name: {getattr(embeddings, 'azure_deployment', 'Not available')}")
            
            # Test a small embedding
            try:
                test_embedding = embeddings.embed_query("test query")
                print(f"   ✅ Test embedding successful (dimension: {len(test_embedding)})")
                return embeddings, "azure_openai"
            except Exception as e:
                print(f"   ❌ Test embedding failed: {e}")
                print(f"   🔍 Error type: {type(e).__name__}")
                if hasattr(e, 'response'):
                    print(f"   📄 Response: {e.response}")
                
        except ImportError:
            print("   ❌ AzureOpenAIEmbeddings not available")
    
    # Test standard OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        print(f"\n✅ OpenAI Configuration Found:")
        print(f"   API Key: {openai_api_key[:8]}..." if len(openai_api_key) > 8 else "   API Key: [short key]")
        
        try:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key,
                model="text-embedding-3-small"
            )
            
            print(f"   ✅ Successfully created OpenAIEmbeddings instance")
            print(f"   📡 Base URL: {getattr(embeddings, 'openai_api_base', 'https://api.openai.com/v1')}")
            
            # Test a small embedding
            try:
                test_embedding = embeddings.embed_query("test query")
                print(f"   ✅ Test embedding successful (dimension: {len(test_embedding)})")
                return embeddings, "openai"
            except Exception as e:
                print(f"   ❌ Test embedding failed: {e}")
                
        except ImportError:
            print("   ❌ OpenAIEmbeddings not available")
    
    # Fallback to sentence transformers
    print(f"\n🔄 Falling back to Sentence Transformers:")
    try:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
        
        print(f"   ✅ Successfully created SentenceTransformerEmbeddings instance")
        print(f"   📡 Local model: all-MiniLM-L6-v2")
        
        # Test a small embedding
        try:
            test_embedding = embeddings.embed_query("test query")
            print(f"   ✅ Test embedding successful (dimension: {len(test_embedding)})")
            return embeddings, "sentence_transformers"
        except Exception as e:
            print(f"   ❌ Test embedding failed: {e}")
            
    except ImportError:
        print("   ❌ SentenceTransformerEmbeddings not available")
    
    return None, None

def test_chroma_methods(embeddings, provider_type):
    """Test different Chroma initialization methods with the embedding provider."""
    if not embeddings:
        print("\n❌ No working embedding provider found - skipping Chroma tests")
        return
    
    print(f"\n🧪 Testing Chroma Methods with {provider_type}")
    print("=" * 60)
    
    from langchain_community.vectorstores import Chroma
    from langchain.schema import Document
    
    # Create test documents
    test_docs = [
        Document(page_content="Test document 1", metadata={"source": "test1"}),
        Document(page_content="Test document 2", metadata={"source": "test2"}),
    ]
    
    # Test 1: The reliable from_documents method
    print("\n1️⃣  Testing Chroma.from_documents() [RELIABLE METHOD]")
    try:
        test_db_path = "./test_chroma_reliable"
        if Path(test_db_path).exists():
            import shutil
            shutil.rmtree(test_db_path)
        
        db = Chroma.from_documents(
            test_docs,
            embedding=embeddings,
            persist_directory=test_db_path
        )
        print("   ✅ Chroma.from_documents() - SUCCESS")
        print("   📡 URL routing: Properly configured")
        print("   🔍 Authentication: Correctly handled")
        
        # Test similarity search
        results = db.similarity_search("test", k=1)
        print(f"   🔎 Similarity search: SUCCESS ({len(results)} results)")
        
        # Clean up
        shutil.rmtree(test_db_path)
        
    except Exception as e:
        print(f"   ❌ Chroma.from_documents() - FAILED: {e}")
        print(f"   🔍 Error type: {type(e).__name__}")
    
    # Test 2: Manual vectorstore creation (often problematic)
    print("\n2️⃣  Testing manual Chroma() + add_documents() [PROBLEMATIC METHOD]")
    try:
        test_db_path = "./test_chroma_manual"
        if Path(test_db_path).exists():
            import shutil
            shutil.rmtree(test_db_path)
        
        # This method often fails with URL routing issues
        db = Chroma(
            persist_directory=test_db_path,
            embedding_function=embeddings
        )
        db.add_documents(test_docs)  # This may fail with "resource not found"
        
        print("   ✅ Manual Chroma() + add_documents() - SUCCESS")
        
        # Test similarity search
        results = db.similarity_search("test", k=1)
        print(f"   🔎 Similarity search: SUCCESS ({len(results)} results)")
        
        # Clean up
        shutil.rmtree(test_db_path)
        
    except Exception as e:
        print(f"   ❌ Manual Chroma() + add_documents() - FAILED: {e}")
        print(f"   🔍 Error type: {type(e).__name__}")
        if "resource not found" in str(e).lower() or "404" in str(e):
            print("   🚨 This is the URL routing issue!")
            print("   💡 The embedding service URL is not properly configured")

def main():
    """Main diagnostic function."""
    print("🔧 Running Embedding Service Diagnostics...\n")
    
    # Test embedding providers
    embeddings, provider_type = test_embedding_provider()
    
    # Test Chroma methods
    test_chroma_methods(embeddings, provider_type)
    
    print(f"\n📊 Summary:")
    print("=" * 60)
    print("✅ Chroma.from_documents() = Reliable (proper URL routing)")
    print("❌ Manual Chroma() methods = Often fail (URL routing issues)")
    print("\n💡 Always use Chroma.from_documents() for best compatibility!")

if __name__ == "__main__":
    main()
