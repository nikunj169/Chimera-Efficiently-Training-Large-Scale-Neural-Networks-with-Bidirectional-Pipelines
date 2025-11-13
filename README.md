# Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines

This project implements the Chimera framework for efficient training of large-scale neural networks using bidirectional pipelines.

For detailed documentation, project overview, and usage instructions, please refer to the main README located in the `docs/` directory:

[Go to Documentation](docs/README.md)

## Project Structure

*   `chimera/`: The core Python package containing the implementation of the Chimera framework.
*   `docs/`: Comprehensive documentation, including the main README, project explanations, and research papers.
*   `examples/`: Demonstration scripts showcasing various aspects of the Chimera framework.
*   `scripts/`: Utility scripts and experimental code.
*   `tests/`: Unit and integration tests for the project.
*   `requirements.txt`: Python package dependencies.
*   `pyproject.toml`: Project metadata and build configuration.

## Installation

To set up the project, first ensure you have Python 3.8+ installed. It is recommended to use a virtual environment:

```bash
python -m venv nhpc
./nhpc/Scripts/activate # On Windows
source nhpc/bin/activate # On Linux/macOS
pip install -r requirements.txt
pip install -e . # Install the chimera package in editable mode
```

## Running Examples

Navigate to the `examples/` directory and run the demonstration scripts:

```bash
cd examples
python demo_model.py
python demo_performance.py
python demo_schedule.py
```

## Running Tests

To run all tests, execute the `run_all_tests.py` script from the `tests/` directory:

```bash
cd tests
python run_all_tests.py
```
