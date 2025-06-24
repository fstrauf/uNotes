#!/usr/bin/env python3
"""
Setup script for Universal Personal Knowledge Context System
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists() and env_example.exists():
        print("Creating .env file from template...")
        env_file.write_text(env_example.read_text())
        print("Please edit .env file and add your OpenAI API key")
        return False
    elif not env_file.exists():
        print("No .env file found. Please create one with GRAPHRAG_API_KEY")
        return False
    
    return True

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher required")
        return False
    return True

def setup_directories():
    """Create necessary directories"""
    dirs = ['data', 'data/input', 'data/output', 'config']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("Created necessary directories")

def main():
    print("Setting up Universal Personal Knowledge Context System...")
    
    if not check_python_version():
        return 1
    
    setup_directories()
    
    if not create_env_file():
        print("\nNext steps:")
        print("1. Edit the .env file and add your OpenAI API key")
        print("2. Run: python main.py --vault-path /path/to/your/obsidian/vault")
        return 1
    
    print("\nSetup complete!")
    print("\nUsage examples:")
    print("1. Process vault and run indexing:")
    print("   python main.py --vault-path /path/to/vault")
    print("\n2. Interactive query mode:")
    print("   python main.py --vault-path /path/to/vault --interactive")
    print("\n3. Run validation tests:")
    print("   python main.py --vault-path /path/to/vault --run-validation")
    
    return 0

if __name__ == "__main__":
    exit(main()) 