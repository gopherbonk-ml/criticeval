from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Optional, List, Dict, TypeVar, Type


T = TypeVar("T", bound="ExtraKwargsMixin")


@dataclass
class ExtraKwargsMixin:
    _extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls: Type[T], data: Mapping[str, Any], *, keep_extra: bool = True) -> T:
        allowed = {f.name for f in fields(cls)}
        known = {k: v for k, v in data.items() if k in allowed}
        obj = cls(**known)
        if keep_extra:
            obj._extra.update({k: v for k, v in data.items() if k not in allowed})
        return obj


@dataclass
class Problem(ExtraKwargsMixin):
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
    solver_extracted_fields: Optional[Dict[str, str]] = field(default_factory=dict)


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
    judger_extracted_fields: Optional[Dict[str, str]] = field(default_factory=dict)
