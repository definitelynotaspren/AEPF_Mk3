"""Loan default prediction package."""
from pathlib import Path

BASE_PATH = Path(__file__).parent

# Import key functions
from .model_report import generate_model_report

__all__ = ['generate_model_report', 'BASE_PATH'] 