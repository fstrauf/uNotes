# Contributing to Universal Personal Knowledge Context System

Thank you for your interest in contributing! This guide will help you get started.

## 🚀 Quick Start for Contributors

1. **Fork the repository**
2. **Clone your fork:**
   ```bash
   git clone https://github.com/yourusername/universal-knowledge-context.git
   cd universal-knowledge-context
   ```
3. **Set up development environment:**
   ```bash
   ./install.sh
   ```
4. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🛠️ Development Setup

### Prerequisites
- Python 3.8+
- OpenAI API key (for testing)
- Git

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If it exists

# Run tests
python -m pytest tests/

# Run with development vault
python main.py --vault-path ./test_vault --interactive
```

## 📋 Contribution Areas

### High Priority
- [ ] **Cross-platform support** (Windows, Linux)
- [ ] **Alternative note formats** (Notion, Roam Research, Logseq)
- [ ] **Performance optimizations** for large vaults
- [ ] **Better error handling** and user feedback

### Medium Priority
- [ ] **Web interface** for knowledge exploration
- [ ] **Visualization tools** for knowledge graphs
- [ ] **Export capabilities** (PDF reports, etc.)
- [ ] **Advanced search features**

### Documentation
- [ ] **Video tutorials**
- [ ] **More example configurations**
- [ ] **Troubleshooting guides**
- [ ] **API documentation**

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_obsidian_processor.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Test Guidelines
- Write tests for new features
- Maintain or improve test coverage
- Test with different vault configurations
- Include edge cases

## 📝 Code Style

### Python Style
- Follow **PEP 8**
- Use **type hints** where possible
- Document functions with docstrings
- Keep functions focused and small

### Example
```python
def process_notes(vault_path: str, output_dir: str) -> List[Dict[str, Any]]:
    """
    Process notes from vault and extract content.
    
    Args:
        vault_path: Path to the notes vault
        output_dir: Directory for processed output
        
    Returns:
        List of processed note dictionaries
        
    Raises:
        ValueError: If vault_path doesn't exist
    """
    # Implementation here
```

## 🔄 Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality  
3. **Ensure tests pass**
4. **Update README** if applicable
5. **Submit pull request** with clear description

### PR Template
```
## What This Changes
Brief description of the changes

## Why This Change
Explain the motivation and context

## Testing
- [ ] Existing tests pass
- [ ] New tests added (if applicable)
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Breaking changes noted
```

## 🐛 Reporting Issues

### Bug Reports
Include:
- **Operating system** and version
- **Python version**
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Error messages** (full traceback)
- **Log files** if relevant

### Feature Requests
Include:
- **Use case** description
- **Proposed solution** (if any)
- **Alternative approaches** considered
- **Implementation complexity** estimate

## 📚 Architecture Overview

### Core Components
```
src/
├── obsidian_processor.py    # Note parsing and processing
├── graphrag_manager.py      # GraphRAG integration
├── mcp_server.py           # Claude Desktop MCP server
├── file_watcher.py         # File monitoring
└── validation_tests.py     # Quality assurance
```

### Data Flow
```
Notes → Processor → GraphRAG → Knowledge Graph → MCP → Claude Desktop
```

## 🎯 Development Priorities

### Current Focus
1. **Stability** - Fix edge cases and improve error handling
2. **Performance** - Optimize for large vaults (1000+ notes)
3. **Usability** - Better setup experience and documentation

### Future Roadmap
1. **Multi-format support** - Beyond Obsidian/Markdown
2. **Collaborative features** - Shared knowledge bases
3. **Advanced analytics** - Knowledge evolution tracking

## 💬 Getting Help

- **GitHub Issues** - For bugs and feature requests
- **Discussions** - For questions and ideas
- **Discord** - For real-time chat (link in README)

## 🙏 Recognition

Contributors will be:
- Listed in README acknowledgments
- Tagged in release notes for their contributions
- Invited to become maintainers for significant contributions

---

**Happy coding!** 🚀 