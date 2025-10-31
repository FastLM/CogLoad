# Contributing to Cognitive Load Traces

Thank you for your interest in contributing to the CLT framework! This document provides guidelines for contributing.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/CogLoad.git`
3. Install dependencies: `pip install -r requirements.txt -e .`
4. Install dev dependencies: `pip install -e ".[dev]"`

## Development Workflow

### Making Changes

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes following the existing code style
3. Add tests for new functionality
4. Run tests: `pytest tests/`
5. Run linting: `black . && flake8 . && mypy .`
6. Commit with clear messages: `git commit -m "Add feature X"`

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Write docstrings for all public functions and classes
- Keep functions focused and modular
- Add comments for complex logic

### Testing

- Write unit tests for all new functionality
- Aim for >80% code coverage
- Test edge cases and error conditions
- Use descriptive test names

### Documentation

- Update README.md for user-facing changes
- Add docstrings to new classes and functions
- Update examples if adding new features
- Document any breaking changes

## Submission Process

1. Push your changes: `git push origin feature/your-feature-name`
2. Create a pull request on GitHub
3. Ensure all tests pass and CI is green
4. Request review from maintainers
5. Address review comments

## Areas for Contribution

### High Priority

- Additional intervention strategies
- Support for more model architectures
- Performance optimizations
- Extended documentation and tutorials

### Medium Priority

- New visualization types
- Multimodal reasoning support
- Production deployment guides
- Benchmark improvements

### Research Directions

- Cross-model CLT transferability
- Interpretability metrics
- Safety applications
- Real-world case studies

## Questions?

Feel free to open an issue for questions or discussions about contributions.

