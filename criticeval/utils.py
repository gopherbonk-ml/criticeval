import logging
import os
import json
import csv
from pathlib import Path
import pandas as pd
from .structure import Problem, Results


logger = logging.getLogger("criticeval.utils")


def read_input_data(problem_file: str) -> list[Problem]:
    """Read problems from CSV or JSON file and return list of `Problem` instances."""
    p = Path(problem_file)
    if not p.exists():
        raise FileNotFoundError(f"Problem file not found: {problem_file}")

    if p.suffix == ".csv":
        problems = pd.read_csv(p)
        return [Problem(**row.to_dict()) for _, row in problems.iterrows()]
    if p.suffix == ".json":
        problems = pd.read_json(p)
        return [Problem(**row.to_dict()) for _, row in problems.iterrows()]

    raise ValueError(f"Unsupported file format: {problem_file}")


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


def save_results(output_dir: str, results: list[Results]):
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # JSONL
    with open(outp / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "problem": asdict(r.problem),
                "llm_solution": asdict(r.llm_solution),
                "judger_input": r.judger_input,
                "judger_output": r.judger_output,
            }, ensure_ascii=False) + "\n")

    # CSV summary
    with open(outp / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["problem_task", "answer", "judger_output"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "problem_task": getattr(r.problem, "task", ""),
                "answer": getattr(r.llm_solution, "answer", ""),
                "judger_output": r.judger_output,
            })