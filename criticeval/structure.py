from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Problem:
    # Meta Problem Info
    source: Optional[str] = None
    topic: Optional[str] = None

    # Core Problem
    task: str = ""
    images: Optional[List[str]] = field(default_factory=list)
    target_solution: Optional[str] = None
    target_answer: Optional[str] = None

    # Eval Criteria
    global_eval_criteria: Optional[str] = None
    task_eval_criteria: Optional[str] = None


@dataclass
class SolverInput(Problem):
    # Template
    solver_template_name: Optional[str] = None
    solver_prompt: str = ""


@dataclass
class SolverOutput(SolverInput):
    # Meta Solver Info
    solver_model_name: Optional[str] = None
    solver_temperature: Optional[float] = None
    solver_top_p: Optional[float] = None
    solver_top_k: Optional[int] = None
    solver_max_tokens: Optional[int] = None

    # LLM Solution
    solver_solution: str = ""
    solver_answer: str = ""


@dataclass
class JudgerInput(SolverOutput):
    # Judger Input
    judger_template_name: Optional[str] = None
    judger_prompt: str = ""


@dataclass
class JudgerOutput(JudgerInput):
    judger_model_name: Optional[str] = None
    judger_temperature: Optional[float] = None
    judger_top_p: Optional[float] = None
    judger_top_k: Optional[int] = None
    judger_max_tokens: Optional[int] = None

    judger_solution: str = ""
    judger_answer: str = ""
