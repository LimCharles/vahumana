from setuptools import setup, find_packages

setup(
    name="vahumana",
    version="0.1.0",
    description="Human-like lossy reconstructive emotional memory augmentation for LLMs",
    author="Charles Lim",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "scikit-learn",
        "tqdm",
    ],
)
