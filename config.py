"""
Configuration management for Skippy SAP EWM Assistant.
Supports multiple LLM providers with environment-based configuration.
"""

import os
from typing import Optional, Dict, Any
from enum import Enum
from pathlib import Path
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class SkippyConfig(BaseSettings):
    """Configuration settings for Skippy application."""
    
    # LLM Provider Configuration
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        env="LLM_PROVIDER",
        description="LLM provider to use"
    )
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(
        default=None,
        env="OPENAI_API_KEY",
        description="OpenAI API key"
    )
    openai_model: str = Field(
        default="gpt-4-turbo-preview",
        env="OPENAI_MODEL",
        description="OpenAI model name"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        env="OPENAI_EMBEDDING_MODEL",
        description="OpenAI embedding model"
    )
    
    # Azure OpenAI Configuration
    azure_openai_api_key: Optional[str] = Field(
        default=None,
        env="AZURE_OPENAI_API_KEY",
        description="Azure OpenAI API key"
    )
    azure_openai_endpoint: Optional[str] = Field(
        default=None,
        env="AZURE_OPENAI_ENDPOINT",
        description="Azure OpenAI endpoint URL"
    )
    azure_openai_api_version: str = Field(
        default="2024-02-15-preview",
        env="AZURE_OPENAI_API_VERSION",
        description="Azure OpenAI API version"
    )
    azure_openai_deployment_name: Optional[str] = Field(
        default=None,
        env="AZURE_OPENAI_DEPLOYMENT_NAME",
        description="Azure OpenAI deployment name"
    )
    azure_openai_embedding_deployment: Optional[str] = Field(
        default=None,
        env="AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        description="Azure OpenAI embedding deployment name"
    )
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(
        default=None,
        env="ANTHROPIC_API_KEY",
        description="Anthropic API key"
    )
    anthropic_model: str = Field(
        default="claude-3-sonnet-20240229",
        env="ANTHROPIC_MODEL",
        description="Anthropic model name"
    )
    
    # Local LLM Configuration
    local_llm_base_url: str = Field(
        default="http://localhost:11434",
        env="LOCAL_LLM_BASE_URL",
        description="Local LLM base URL"
    )
    local_llm_model: str = Field(
        default="llama2:7b",
        env="LOCAL_LLM_MODEL",
        description="Local LLM model name"
    )
    local_embedding_model: str = Field(
        default="nomic-embed-text",
        env="LOCAL_EMBEDDING_MODEL",
        description="Local embedding model name"
    )
    
    # Application Configuration
    app_title: str = Field(
        default="Skippy - SAP EWM Assistant",
        env="APP_TITLE",
        description="Application title"
    )
    app_description: str = Field(
        default="Your friendly SAP Extended Warehouse Management coach",
        env="APP_DESCRIPTION",
        description="Application description"
    )
    
    # ChromaDB Configuration
    chroma_db_path: str = Field(
        default="./data/chroma_db",
        env="CHROMA_DB_PATH",
        description="ChromaDB storage path"
    )
    collection_name: str = Field(
        default="sap_ewm_docs",
        env="COLLECTION_NAME",
        description="ChromaDB collection name"
    )
    
    # PDF Processing Configuration
    pdf_data_path: str = Field(
        default="./data/pdfs",
        env="PDF_DATA_PATH",
        description="Path to PDF documents"
    )
    chunk_size: int = Field(
        default=1000,
        env="CHUNK_SIZE",
        description="Text chunk size for processing"
    )
    chunk_overlap: int = Field(
        default=200,
        env="CHUNK_OVERLAP",
        description="Text chunk overlap"
    )
    
    # LLM Settings
    max_tokens: int = Field(
        default=2000,
        env="MAX_TOKENS",
        description="Maximum tokens for LLM response"
    )
    temperature: float = Field(
        default=0.7,
        env="TEMPERATURE",
        description="LLM temperature setting"
    )
    
    # Retrieval Configuration
    top_k_results: int = Field(
        default=5,
        env="TOP_K_RESULTS",
        description="Number of top results to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.7,
        env="SIMILARITY_THRESHOLD",
        description="Similarity threshold for retrieval"
    )
    
    # UI Configuration
    skippy_avatar_path: str = Field(
        default="./assets/avatars/skippy.png",
        env="SKIPPY_AVATAR_PATH",
        description="Path to Skippy's avatar image"
    )
    chat_theme: str = Field(
        default="light",
        env="CHAT_THEME",
        description="Chat interface theme"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level"
    )
    log_file: str = Field(
        default="./logs/skippy.log",
        env="LOG_FILE",
        description="Log file path"
    )
    
    # Security Configuration
    allowed_hosts: str = Field(
        default="localhost,127.0.0.1",
        env="ALLOWED_HOSTS",
        description="Comma-separated allowed hosts"
    )
    cors_origins: str = Field(
        default="http://localhost:8501",
        env="CORS_ORIGINS",
        description="Comma-separated CORS origins"
    )

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration based on selected provider."""
        if self.llm_provider == LLMProvider.OPENAI:
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "embedding_model": self.openai_embedding_model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        elif self.llm_provider == LLMProvider.AZURE_OPENAI:
            return {
                "provider": "azure_openai",
                "api_key": self.azure_openai_api_key,
                "endpoint": self.azure_openai_endpoint,
                "api_version": self.azure_openai_api_version,
                "deployment_name": self.azure_openai_deployment_name,
                "embedding_deployment": self.azure_openai_embedding_deployment,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        elif self.llm_provider == LLMProvider.ANTHROPIC:
            return {
                "provider": "anthropic",
                "api_key": self.anthropic_api_key,
                "model": self.anthropic_model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        elif self.llm_provider == LLMProvider.LOCAL:
            return {
                "provider": "local",
                "base_url": self.local_llm_base_url,
                "model": self.local_llm_model,
                "embedding_model": self.local_embedding_model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def validate_configuration(self) -> None:
        """Validate configuration based on selected provider."""
        if self.llm_provider == LLMProvider.OPENAI and not self.openai_api_key:
            raise ValueError("OpenAI API key is required when using OpenAI provider")
        
        if self.llm_provider == LLMProvider.AZURE_OPENAI:
            required_azure_fields = [
                self.azure_openai_api_key,
                self.azure_openai_endpoint,
                self.azure_openai_deployment_name,
                self.azure_openai_embedding_deployment
            ]
            if not all(required_azure_fields):
                raise ValueError("All Azure OpenAI configuration fields are required")
        
        if self.llm_provider == LLMProvider.ANTHROPIC and not self.anthropic_api_key:
            raise ValueError("Anthropic API key is required when using Anthropic provider")

    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        directories = [
            Path(self.chroma_db_path).parent,
            Path(self.pdf_data_path),
            Path(self.skippy_avatar_path).parent,
            Path(self.log_file).parent,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global configuration instance
config = SkippyConfig()


def get_config() -> SkippyConfig:
    """Get the global configuration instance."""
    return config


def validate_and_setup_config() -> SkippyConfig:
    """Validate configuration and setup required directories."""
    config.validate_configuration()
    config.ensure_directories()
    return config
