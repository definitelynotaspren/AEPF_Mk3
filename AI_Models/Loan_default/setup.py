"""Setup script for loan default prediction package."""
from setuptools import setup, find_namespace_packages

setup(
    name="loan_default",
    version="0.1.0",
    packages=find_namespace_packages(include=['AI_Models.*']),
    install_requires=[
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0'
    ],
    python_requires='>=3.8',
) 