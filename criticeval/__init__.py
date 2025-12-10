# criticeval/__init__.py
"""Public package exports for criticeval."""

from .structure import (
    Problem, SolverInput, SolverOutput, JudgerInput, JudgerOutput
)

__all__ = ["Problem", "SolverInput", "SolverOutput", "JudgerInput", "JudgerOutput"]