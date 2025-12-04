from pathlib import Path
from dataclasses import asdict
from typing import Optional, Union

from jinja2 import Environment, FileSystemLoader

from ..structure import Problem, Solution


DEFAULT_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _get_env(templates_dir: Optional[Union[str, Path]] = None) -> Environment:
    base_dir = Path(templates_dir) if templates_dir else DEFAULT_TEMPLATES_DIR
    return Environment(
        loader=FileSystemLoader(str(base_dir)),
        autoescape=False
    )


def render_solver_prompt(problem: Problem, template_file: str, templates_dir: Optional[Union[str, Path]] = None):
    template = _get_env(templates_dir).get_template(template_file)
    return template.render(**asdict(problem))


def render_judger_prompt(problem: Problem, solution: Solution, template_file: str, templates_dir: Optional[Union[str, Path]] = None):
    template = _get_env(templates_dir).get_template(template_file)
    return template.render(**asdict(problem), **asdict(solution))
