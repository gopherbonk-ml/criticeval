from pathlib import Path
from dataclasses import asdict
from typing import Optional, Union

from jinja2 import Environment, FileSystemLoader

from ..structure import Problem, SolverOutput


DEFAULT_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _get_env(templates_dir: Optional[Union[str, Path]] = None) -> Environment:
    base_dir = Path(templates_dir) if templates_dir else DEFAULT_TEMPLATES_DIR
    return Environment(
        loader=FileSystemLoader(str(base_dir)),
        autoescape=False
    )


def render_solver_prompt(problem: Problem, template_file: str):
    template = _get_env(DEFAULT_TEMPLATES_DIR).get_template(template_file + ".jinja")
    return template.render(**asdict(problem))


def render_judger_prompt(solver_output: SolverOutput, template_file: str):
    template = _get_env(DEFAULT_TEMPLATES_DIR).get_template(template_file + ".jinja")
    return template.render(**asdict(solver_output))
