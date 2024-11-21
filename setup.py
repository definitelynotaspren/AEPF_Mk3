from setuptools import setup, find_packages

setup(
    name="AEPF_Core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0"
    ]
) 