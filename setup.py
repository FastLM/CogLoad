"""
Setup script for Cognitive Load Traces (CLT) framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Cognitive Load Traces framework for transformer interpretability"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="cogload",
    version="0.1.0",
    author="Dong Liu, Yanxuan Yu",
    author_email="dong.liu.dl2367@yale.edu",
    description="Cognitive Load Traces: A mid-level interpretability framework for deep transformer models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yale-nlp/CogLoad",
    packages=find_packages(exclude=["examples", "tests", "benchmarks"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cogload-benchmark=benchmarks.evaluate_clt:run_benchmark",
        ],
    },
)

