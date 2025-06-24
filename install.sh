#!/bin/bash

echo "🚀 Universal Personal Knowledge Context System - Installation Script"
echo "=================================================================="

# Check if Python 3.8+ is available
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            echo "✓ Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
            return 0
        fi
    fi
    
    if command -v python &> /dev/null; then
        PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            echo "✓ Python $PYTHON_VERSION found"
            PYTHON_CMD="python"
            return 0
        fi
    fi
    
    echo "❌ Python 3.8+ is required but not found"
    echo "   Please install Python 3.8 or higher and try again"
    return 1
}

# Create virtual environment
create_venv() {
    echo "📦 Creating virtual environment..."
    if [ ! -d "myenv" ]; then
        $PYTHON_CMD -m venv myenv
        echo "✓ Virtual environment created"
    else
        echo "✓ Virtual environment already exists"
    fi
}

# Activate virtual environment and install dependencies
install_deps() {
    echo "📥 Installing dependencies..."
    
    # Activate virtual environment
    source myenv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        echo "✓ Dependencies installed"
    else
        echo "❌ requirements.txt not found"
        return 1
    fi
}

# Run setup script
run_setup() {
    echo "⚙️  Running setup script..."
    source myenv/bin/activate
    $PYTHON_CMD setup.py
}

# Main installation process
main() {
    # Check if we're in the right directory
    if [ ! -f "setup.py" ] || [ ! -f "requirements.txt" ]; then
        echo "❌ Please run this script from the project root directory"
        echo "   (where setup.py and requirements.txt are located)"
        exit 1
    fi
    
    # Check Python version
    if ! check_python; then
        exit 1
    fi
    
    # Create virtual environment
    if ! create_venv; then
        exit 1
    fi
    
    # Install dependencies
    if ! install_deps; then
        exit 1
    fi
    
    # Run setup
    run_setup
    
    echo ""
    echo "🎉 Installation complete!"
    echo ""
    echo "📋 Quick start:"
    echo "1. Edit .env file with your OpenAI API key and vault path"
    echo "2. Run: source myenv/bin/activate && python main.py --interactive"
    echo ""
    echo "📖 For detailed instructions, see README.md"
}

# Run main function
main "$@" 