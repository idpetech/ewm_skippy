# Contributing to Skippy SAP EWM Assistant

Thank you for your interest in contributing to Skippy! 🎉 

This document provides guidelines and information for contributors to help make the development process smooth and collaborative.

## 🤝 How to Contribute

### Reporting Bugs

1. **Check existing issues** first to avoid duplicates
2. **Use the issue templates** when creating new bug reports
3. **Include detailed information**:
   - Steps to reproduce the bug
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)
   - Error messages and logs
   - Screenshots if applicable

### Suggesting Features

1. **Search existing issues** for similar feature requests
2. **Use the feature request template**
3. **Provide clear use cases** and benefits
4. **Include mockups or examples** when helpful

### Contributing Code

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** following our coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Commit with clear messages** following our format
7. **Push to your branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request** with detailed description

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- OpenAI API key (for testing)

### Local Development

1. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/ewm_skippy.git
   cd ewm_skippy
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8 mypy  # Development tools
   ```

4. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

5. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

## 📝 Coding Standards

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **String quotes**: Use double quotes for strings
- **Imports**: Group imports (standard library, third-party, local)
- **Type hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings

### Code Formatting

We use **Black** for code formatting:

```bash
# Format all Python files
black .

# Check formatting without changes
black --check --diff .
```

### Linting

We use **flake8** for linting:

```bash
# Run linting
flake8 .

# Fix common issues automatically
autopep8 --in-place --recursive .
```

### Type Checking

We use **mypy** for type checking:

```bash
# Run type checking
mypy *.py --ignore-missing-imports
```

### Documentation

- **All functions** should have docstrings
- **Complex logic** should have inline comments
- **Public APIs** should be well-documented
- **Update README.md** for user-facing changes

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_assistant.py -v
```

### Writing Tests

- **Write tests** for all new functionality
- **Use descriptive test names** that explain the scenario
- **Follow AAA pattern**: Arrange, Act, Assert
- **Mock external dependencies** (OpenAI API, file system, etc.)
- **Test edge cases** and error conditions

Example test structure:
```python
def test_assistant_generates_response_with_context():
    # Arrange
    assistant = SkippyAssistant("test-api-key")
    question = "How do I configure storage types?"
    
    # Act
    response = assistant.generate_response(question)
    
    # Assert
    assert response is not None
    assert len(response) > 0
    assert "storage types" in response.lower()
```

## 📋 Commit Guidelines

### Commit Message Format

Use the following format for commit messages:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(chat): Add conversation reset functionality

Add button to clear chat history and reset conversation state.
Includes proper session state management and user confirmation.

Closes #123

fix(retriever): Handle empty ChromaDB collections gracefully

Add validation to prevent errors when querying empty collections.
Includes fallback behavior and user-friendly error messages.

docs(readme): Update deployment instructions for Railway

Add step-by-step Railway deployment guide with environment
variable configuration examples.
```

## 🔍 Code Review Process

### Pull Request Guidelines

1. **Provide clear description** of changes
2. **Reference related issues** using keywords (fixes #123)
3. **Include screenshots** for UI changes
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Keep PRs focused** - one feature per PR
7. **Respond to feedback** constructively

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact is considered
- [ ] Security implications are reviewed
- [ ] Error handling is appropriate

## 🌟 Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub contributors** page
- **Special thanks** in documentation

## 📞 Getting Help

If you need help or have questions:

1. **Check the documentation** in README.md
2. **Search existing issues** for similar questions
3. **Create a new issue** with the question template
4. **Join discussions** in GitHub Discussions
5. **Contact maintainers** for urgent matters

## 🎯 Areas for Contribution

We especially welcome contributions in:

### 🤖 **AI and NLP**
- Improve response quality and relevance
- Add support for other LLM providers
- Enhance context retrieval algorithms
- Optimize token usage and costs

### 📚 **SAP EWM Knowledge**
- Add more SAP EWM training content
- Improve process documentation
- Enhance error handling scenarios
- Add integration examples

### 🎨 **User Interface**
- Improve UI/UX design
- Add accessibility features
- Mobile responsiveness
- Dark/light theme support

### 🔧 **Technical Infrastructure**
- Performance optimizations
- Monitoring and analytics
- Security enhancements
- Deployment automation

### 📖 **Documentation**
- User guides and tutorials
- API documentation
- Deployment guides
- Video tutorials

### 🧪 **Testing**
- Unit test coverage
- Integration tests
- Performance testing
- Security testing

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Skippy! 🚀**

*Together we're making SAP EWM more accessible and user-friendly for the warehouse management community.*
