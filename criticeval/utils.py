import logging
import os
import json
import csv
from pathlib import Path
import pandas as pd
from .structure import Problem
from dataclasses import asdict


logger = logging.getLogger("criticeval.utils")


def check_images():
    pass


def read_input_data(
        problem_file: str,
        images_dir: Path = None
    ) -> list[Problem]:
    """
    Read problems from CSV or JSON file

    Return list of `Problem` instances.
    """
    p = Path(problem_file)
    if not p.exists():
        raise FileNotFoundError(f"Problem file not found: {problem_file}")

    if p.suffix == ".csv":
        df = pd.read_csv(p)
    elif p.suffix == ".json":
        df = pd.read_json(p)
    else:
        raise ValueError(f"Unsupported file format: {problem_file}")

    problems: list[Problem] = []

    for _, row in df.iterrows():
        sample = row.to_dict()

        images = sample.get("images")
        sample["images"] = []
        if images:
            sample["images"] = [images_dir / image for image in images]

        problems.append(Problem(
            source=sample.get("source", None),
            topic=sample.get("topic", None),
            task=sample.get("task", None),
            images=sample.get("images", None),
            target_solution=sample.get("target_solution", None),
            target_answer=sample.get("target_answer", None),
            global_eval_criteria=sample.get("global_eval_criteria", None),
            task_eval_criteria=sample.get("task_eval_criteria", None),
        ))

    return problems


def check_or_create_output_dir(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be stored in {output_dir}")


def validate_config(cfg) -> None:
    """
    Basic config validation: check presence of required keys and basic types.

    Raises ValueError on invalid config.
    """
    required_paths = ["paths", "solver", "judger", "template", "data"]
    for key in required_paths:
        if key not in cfg:
            raise ValueError(f"Missing required config section: {key}")
