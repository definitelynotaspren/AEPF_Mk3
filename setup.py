"""Setup script for AEPF project."""
from setuptools import setup, find_packages

setup(
    name="aepf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'streamlit>=1.24.0',
        'PyYAML>=6.0.1',
        'setuptools>=42.0.0',
        'wheel>=0.37.0'
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'pylint>=2.17.0'
        ]
    },
    python_requires='>=3.8',
) 