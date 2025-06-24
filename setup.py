#!/usr/bin/env python3
"""
Setup script for Universal Personal Knowledge Context System
"""

import os
import sys
import json
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/input',
        'data/output', 
        'data/cache',
        'data/logs',
        'data/processed',
        'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def setup_environment():
    """Setup environment file if it doesn't exist"""
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            import shutil
            shutil.copy('.env.example', '.env')
            print("✓ Created .env file from .env.example")
            print("⚠️  Please edit .env file with your API key and vault path")
        else:
            print("⚠️  No .env.example found. Please create .env manually")
    else:
        print("✓ .env file already exists")

def generate_claude_config():
    """Generate Claude Desktop MCP configuration"""
    current_dir = os.path.abspath('.')
    python_path = os.path.join(current_dir, 'myenv', 'bin', 'python')
    server_path = os.path.join(current_dir, 'src', 'mcp_server.py')
    data_path = os.path.join(current_dir, 'data')
    
    # Check if virtual environment exists
    if not os.path.exists(python_path):
        python_path = sys.executable
        print(f"⚠️  Using system Python: {python_path}")
        print("   Consider using a virtual environment for better isolation")
    
    config = {
        "universal-knowledge": {
            "command": python_path,
            "args": [server_path, data_path],
            "env": {
                "GRAPHRAG_API_KEY": "${GRAPHRAG_API_KEY}"
            }
        }
    }
    
    # Save config for user to add to Claude Desktop
    with open('claude_mcp_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Generated Claude Desktop MCP configuration: claude_mcp_config.json")
    return config

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'graphrag',
        'fastmcp', 
        'pandas',
        'watchdog',
        'python-frontmatter',
        'obsidiantools'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    else:
        print("✓ All required dependencies are installed")
        return True

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        print(f"   Current version: {sys.version}")
        return False
    else:
        print(f"✓ Python version {sys.version.split()[0]} is compatible")
        return True

def main():
    print("🚀 Setting up Universal Personal Knowledge Context System...")
    print("=" * 60)
    
    # Check Python version first
    if not check_python_version():
        return 1
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Setup incomplete - please install dependencies first")
        print("   pip install -r requirements.txt")
        return 1
    
    create_directories()
    setup_environment()
    claude_config = generate_claude_config()
    
    print("\n✅ Setup complete!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your OpenAI API key and vault path")
    print("2. Add the MCP configuration to Claude Desktop:")
    print(f"   - Open: ~/Library/Application Support/Claude/claude_desktop_config.json")
    print(f"   - Add the configuration from: claude_mcp_config.json")
    print("3. Run: python main.py --interactive")
    print("\n💡 For detailed instructions, see README.md")
    
    return 0
    
if __name__ == "__main__":
    exit(main()) 