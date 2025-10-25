"""Setup script for Bayesian Expectation Transformer."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="bayesian-transformer",
    version="0.1.0",
    author="Michael Neuberger",
    author_email="",
    description="Bayesian Expectation Transformer with Uncertainty Quantification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scheitelpunk/Bayesian",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
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
        "api": [
            "fastapi>=0.95.0",
            "uvicorn>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bayesian-transformer-api=src.api.server:main",
        ],
    },
)
