#!/usr/bin/env python3
"""
Skippy Multi-Coach System Setup

This script helps set up the multi-coach system by:
1. Creating necessary directories
2. Building indexes for different coach types
3. Setting up sample data
4. Configuring the system
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.index_builders import IndexBuilderFactory

# =============================================================================
# SETUP CONFIGURATION
# =============================================================================

class SetupConfig:
    """Configuration for setup process"""
    
    def __init__(self):
        # Azure OpenAI Configuration (same as main config)
        self.embedding_endpoint = os.getenv("AZURE_EMBEDDING_ENDPOINT", "https://genaiapimna.jnj.com/openai-embeddings/openai")
        self.embedding_api_key = os.getenv("AZURE_EMBEDDING_API_KEY", "f89d10a91b9d4cc989085a495d695eb3")
        self.embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        self.embedding_api_version = os.getenv("AZURE_EMBEDDING_API_VERSION", "2022-12-01")
        
        # Data paths
        self.data_paths = {
            'document': './data/pdfs',  # For EWM, Business Analyst, Support coaches
            'source_code': './data/source_code',  # For Dev Guru coach
            'database_schema': './data/schemas'  # For Dev Guru coach
        }
        
        # Output paths
        self.output_paths = {
            'ewm': './data/ewm_db',
            'business_analyst': './data/ba_db',
            'support': './data/support_db',
            'dev_guru': './data/dev_db'
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for index builders"""
        return {
            'embedding_endpoint': self.embedding_endpoint,
            'embedding_api_key': self.embedding_api_key,
            'embedding_deployment': self.embedding_deployment,
            'embedding_api_version': self.embedding_api_version
        }

# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_logging():
    """Setup logging for setup process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('setup.log'),
            logging.StreamHandler()
        ]
    )

def create_directories():
    """Create necessary directories"""
    directories = [
        './data/ewm_db',
        './data/ba_db', 
        './data/support_db',
        './data/dev_db',
        './data/pdfs',
        './data/source_code',
        './data/schemas',
        './logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {directory}")

def copy_existing_data():
    """Copy existing data to appropriate locations"""
    # Copy existing ChromaDB if it exists
    existing_chroma = Path('./chroma')
    if existing_chroma.exists():
        # Copy to EWM database
        import shutil
        ewm_db = Path('./data/ewm_db')
        if ewm_db.exists():
            shutil.rmtree(ewm_db)
        shutil.copytree(existing_chroma, ewm_db)
        logging.info("Copied existing ChromaDB to EWM database")
    
    # Copy existing PDFs if they exist
    existing_pdfs = Path('./data')
    if existing_pdfs.exists():
        pdfs_dir = Path('./data/pdfs')
        for pdf_file in existing_pdfs.glob('*.pdf'):
            shutil.copy2(pdf_file, pdfs_dir)
            logging.info(f"Copied PDF: {pdf_file.name}")

def build_indexes(config: SetupConfig):
    """Build indexes for all coach types"""
    logging.info("Starting index building process...")
    
    # Define which data sources each coach should use
    coach_data_mapping = {
        'ewm': 'document',
        'business_analyst': 'document', 
        'support': 'document',
        'dev_guru': 'source_code'  # Can also use database_schema
    }
    
    results = {}
    
    for coach_type, data_type in coach_data_mapping.items():
        data_path = config.data_paths[data_type]
        output_path = config.output_paths[coach_type]
        
        # Check if data exists
        if not Path(data_path).exists() or not any(Path(data_path).iterdir()):
            logging.warning(f"No data found for {coach_type} at {data_path}")
            results[coach_type] = False
            continue
        
        try:
            # Create appropriate builder
            if data_type == 'document':
                builder = IndexBuilderFactory.create_builder('document', config.to_dict())
            elif data_type == 'source_code':
                builder = IndexBuilderFactory.create_builder('source_code', config.to_dict())
            elif data_type == 'database_schema':
                builder = IndexBuilderFactory.create_builder('database_schema', config.to_dict())
            else:
                logging.error(f"Unknown data type: {data_type}")
                results[coach_type] = False
                continue
            
            # Build index
            logging.info(f"Building index for {coach_type} from {data_path}")
            result = builder.build_index(data_path, output_path)
            results[coach_type] = result
            
            if result:
                logging.info(f"✅ Successfully built index for {coach_type}")
            else:
                logging.error(f"❌ Failed to build index for {coach_type}")
                
        except Exception as e:
            logging.error(f"Error building index for {coach_type}: {e}")
            results[coach_type] = False
    
    return results

def create_sample_data():
    """Create sample data for testing"""
    # Create sample source code structure
    sample_code_dir = Path('./data/source_code')
    sample_code_dir.mkdir(exist_ok=True)
    
    # Create a sample Python file
    sample_py = sample_code_dir / 'sample_module.py'
    sample_py.write_text('''
"""
Sample Python module for testing Dev Guru Coach
"""

class SampleClass:
    """A sample class for demonstration"""
    
    def __init__(self, name: str):
        self.name = name
        self.value = 0
    
    def increment(self) -> int:
        """Increment the value"""
        self.value += 1
        return self.value
    
    def get_info(self) -> str:
        """Get information about the instance"""
        return f"SampleClass(name={self.name}, value={self.value})"

def sample_function(x: int, y: int) -> int:
    """A sample function that adds two numbers"""
    return x + y

# Sample usage
if __name__ == "__main__":
    obj = SampleClass("test")
    print(obj.get_info())
    print(f"Sum: {sample_function(5, 3)}")
''')
    
    # Create a sample SQL schema
    sample_schema_dir = Path('./data/schemas')
    sample_schema_dir.mkdir(exist_ok=True)
    
    sample_sql = sample_schema_dir / 'sample_schema.sql'
    sample_sql.write_text('''
-- Sample database schema for testing Dev Guru Coach

CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
''')
    
    logging.info("Created sample data files")

def print_setup_summary(results: Dict[str, bool]):
    """Print setup summary"""
    print("\n" + "="*60)
    print("🚀 SKIPPY MULTI-COACH SYSTEM SETUP SUMMARY")
    print("="*60)
    
    print("\n📊 Index Building Results:")
    for coach_type, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {coach_type.replace('_', ' ').title()}: {status}")
    
    print("\n🎯 Available Coaches:")
    print("  🏭 EWM Coach - SAP warehouse operations")
    print("  📋 Business Analyst Coach - Requirements & processes") 
    print("  🔧 Support Coach - Technical troubleshooting")
    print("  💻 Dev Guru Coach - Code analysis & development")
    print("  🌟 Mixed Coach - All capabilities combined")
    
    print("\n📁 Data Directories:")
    print("  ./data/ewm_db - EWM Coach database")
    print("  ./data/ba_db - Business Analyst Coach database")
    print("  ./data/support_db - Support Coach database")
    print("  ./data/dev_db - Dev Guru Coach database")
    print("  ./data/pdfs - Document files")
    print("  ./data/source_code - Source code files")
    print("  ./data/schemas - Database schema files")
    
    print("\n🚀 Next Steps:")
    print("  1. Add your data files to the appropriate directories")
    print("  2. Run this setup script again to rebuild indexes")
    print("  3. Launch the system: python launch_multi_coach.py")
    
    print("\n" + "="*60)

def main():
    """Main setup function"""
    setup_logging()
    
    print("🚀 Setting up Skippy Multi-Coach System...")
    print("="*50)
    
    try:
        # Create configuration
        config = SetupConfig()
        
        # Step 1: Create directories
        print("📁 Creating directories...")
        create_directories()
        
        # Step 2: Copy existing data
        print("📋 Copying existing data...")
        copy_existing_data()
        
        # Step 3: Create sample data
        print("📝 Creating sample data...")
        create_sample_data()
        
        # Step 4: Build indexes
        print("🔨 Building indexes...")
        results = build_indexes(config)
        
        # Step 5: Print summary
        print_setup_summary(results)
        
        print("\n✅ Setup completed successfully!")
        print("Run 'python launch_multi_coach.py' to start the system.")
        
    except Exception as e:
        logging.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
