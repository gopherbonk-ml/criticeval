from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Problem:
    # Meta Problem Info
    source: Optional[str] = None
    topic: Optional[str] = None
    difficulty: Optional[int] = None

    # Core Problem
    task: str = ""
    images: Optional[List[str]] = field(default_factory=list)
    target_solution: Optional[str] = None
    target_answer: Optional[str] = None

    # Eval Criteria
    global_eval_criteria: Optional[str] = None
    task_eval_criteria: Optional[str] = None


@dataclass
class Solution:
    solution: str = ""
    answer: str = ""


@dataclass
class SolverOutput:
    # Meta Problem Info
    source: Optional[str] = None
    topic: Optional[str] = None
    difficulty: Optional[int] = None

    # Meta Solver Info
    solver_model_name: Optional[str] = None
    solver_temperute: Optional[float] = None
    solver_top_p: Optional[float] = None
    solver_top_k: Optional[int] = None
    solver_max_tokens: Optional[int] = None

    # Core Problem
    task: str = ""
    images: Optional[List[str]] = field(default_factory=list)
    target_solution: Optional[str] = None
    target_answer: Optional[str] = None

    # LLM Solution
    llm_solution: str = ""
    llm_answer: str = ""


@dataclass
class JudgerInput:
    problem: Problem
    llm_solution: Solution


'''
Results:
{
    task: str,
    target_solution: str,
    target_answer: str,

    task_eval_criteria: str,
    global_eval_criteria: str,

    welcome_prompt: str,

    llm_solution: str,
    llm_answer: str,

    judger_input: str,
    judger_output: str,
}
'''

@dataclass
class Results:
    problem: Problem
    llm_solution: Solution

    judger_input: str = ""
    judger_output: str = ""
    judger_assesment: str = ""


@dataclass
class MetaInfo:
    solver_model_name: str
    solver_model_path: str

    solver_max_tokens: int
    solver_temperature: float
    solver_top_p: float
    solver_top_k: int

    judger_model_name: str
    judger_model_path: str

    judger_max_tokens: int
    judger_temperature: float
    judger_top_p: float
    judger_top_k: int

    format_file: str
    extract_answer_func: str




