import logging
from pathlib import Path

import os
import json
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import List, Optional, Callable
import pandas as pd
from dataclasses import asdict

from .structure import (
    Problem, SolverInput, SolverOutput, JudgerInput, JudgerOutput
)
from .templates.formatter import render_solver_prompt, render_judger_prompt
from .serve_model.model_api import (
    LLMHost,
    LLMBackendConfig,
    LLMSamplingConfig,
)
from .extractors import get_field_extractor_function
from .utils import read_input_data, check_or_create_output_dir


logger = logging.getLogger("criticeval")
logging.basicConfig(format="[CriticEval] %(message)s")


# =====================
#  IO Helpers
# =====================


def save_solver_outputs(
    output_dir: str | Path,
    solver_outputs: List[SolverOutput],
    experiment_id: str = "base",
) -> Path:
    """
    Сохраняем solver-outputs в один JSON-файл:
    <output_dir>/solver_outputs/<experiment_id>/solver_outputs.json
    """
    base = Path(output_dir) / "solver_outputs" / experiment_id
    base.mkdir(parents=True, exist_ok=True)

    serialized = [asdict(so) for so in solver_outputs]
    output_file = base / "solver_outputs.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=4)

    logger.info(f"Saved solver outputs to {output_file}")
    return output_file


def load_solver_outputs(path: str | Path) -> List[SolverOutput]:
    path = Path(path)
    if path.is_dir():
        path = path / "solver_outputs.json"

    if not path.exists():
        raise FileNotFoundError(f"Solver outputs file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return [SolverOutput.from_dict(item) for item in raw]


# =====================
#  Solver Phase
# =====================


def create_solver_inputs(
    problems: List[Problem],
    templates: List[str],
) -> List[SolverInput]:
    solver_inputs: List[SolverInput] = []
    for template in templates:
        for problem in problems:
            prompt = render_solver_prompt(problem, template)
            solver_inputs.append(
                SolverInput.from_dict({
                    **asdict(problem),
                    "solver_template_name": template,
                    "solver_prompt": prompt,
                })
            )
    return solver_inputs


def eval_solver(
    solver_inputs: List[SolverInput],
    llm_host: LLMHost,
    extract_field_fn: Callable[[str], str],
    backend_cfg: LLMBackendConfig,
    sampling_cfg: LLMSamplingConfig,
) -> List[SolverOutput]:

    solver_outputs: List[SolverOutput] = []

    model_name = getattr(backend_cfg, "model", None)
    temperature = getattr(sampling_cfg, "temperature", None)
    top_p = getattr(sampling_cfg, "top_p", None)
    top_k = getattr(sampling_cfg, "top_k", None)
    max_tokens = getattr(sampling_cfg, "max_tokens", None)

    prompts = [
        getattr(s, "solver_prompt", None)
        for s in solver_prompts
    ]

    images_list = [
        getattr(s, "images", None)
        for s in solver_prompts
    ]

    solver_solutions = llm_host.generate_many(
        prompts,
        images=images_list
    )

    for s_inp, solver_solution in zip(solver_inputs, solver_solutions):
        extracted_fields = extract_field_fn(solver_solution)

        base_dict = asdict(s_inp)
        payload = {
            **base_dict,
            "solver_model_name": model_name,
            "solver_temperature": temperature,
            "solver_top_p": top_p,
            "solver_top_k": top_k,
            "solver_max_tokens": max_tokens,
            "solver_solution": solver_solution,
            "solver_extracted_fields": extracted_fields
        }
        solver_outputs.append(SolverOutput.from_dict(payload))

    return solver_outputs


# =====================
#  Judger Phase
# =====================


def create_judger_inputs(
    solver_outputs: List[SolverOutput],
    judger_templates: List[str],
) -> List[JudgerInput]:
    judger_inputs: List[JudgerInput] = []
    for judger_template in judger_templates:
        for solver_output in solver_outputs:
            judger_prompt = render_judger_prompt(solver_output, judger_template)
            judger_inputs.append(
                JudgerInput.from_dict({
                    **asdict(solver_output),
                    "judger_template_name": judger_template,
                    "judger_prompt": judger_prompt,
                })
            )
    return judger_inputs


def eval_judger(
    judger_inputs: List[SolverInput],
    llm_host: LLMHost,
    extract_field_fn: Callable[[str], str],
    backend_cfg: LLMBackendConfig,
    sampling_cfg: LLMSamplingConfig,
):

    judger_outputs: List[JudgerOutput] = []

    model_name = getattr(backend_cfg, "model", None)
    temperature = getattr(sampling_cfg, "temperature", None)
    top_p = getattr(sampling_cfg, "top_p", None)
    top_k = getattr(sampling_cfg, "top_k", None)
    max_tokens = getattr(sampling_cfg, "max_tokens", None)

    prompts = [
        getattr(j, "judger_prompt", None)
        for j in judger_inputs
    ]

    images_list = [
        getattr(j, "images", None)
        for j in judger_inputs
    ]

    judger_solutions = llm_host.generate_many(
        prompts,
        images=images_list
    )

    for j_inp, judger_solution in zip(judger_inputs, judger_solutions):
        extracted_fields = extract_field_fn(judger_solution)

        base_dict = asdict(j_inp)
        payload = {
            **base_dict,
            "judger_model_name": model_name,
            "judger_temperature": temperature,
            "judger_top_p": top_p,
            "judger_top_k": top_k,
            "judger_max_tokens": max_tokens,
            "judger_solution": judger_solution,
            "judger_extracted_fields": extracted_fields
        }
        judger_outputs.append(JudgerOutput(**payload))

    return judger_outputs


# =====================
#  main
# =====================


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger.info("CriticEval Configuration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    data_dir = Path(cfg.paths.data_dir)
    images_dir = data_dir / "images"

    save_dir = Path(cfg.paths.save_dir)
    check_or_create_output_dir(save_dir)

    def resolve(path: Optional[str]) -> Optional[Path]:
        return data_dir / path if path else None

    problems_file = resolve(cfg.data.get("problem_file", None))
    solver_outputs_file = resolve(cfg.data.get("solver_output_file", None))

    mode = cfg.evaluation.mode or "all"
    logger.info(f"Starting evaluation in mode: {mode}")


    # =====================
    # Solver Evaluation
    # =====================
    
    solver_outputs: List[SolverOutput] = []
    problems: List[Problem] = []

    if mode in ("solver_only", "all"):
        if problems_file is None:
            raise ValueError("cfg.data.problem_file must be set for solver mode")

        solver_cfg = cfg.solver

        extract_field_fn = get_field_extractor_function(
            cfg.template.extract_field_func_for_solver
        )

        solver_templates = cfg.template.get("solver_templates", [])

        problems = read_input_data(problems_file, images_dir)
        solver_inputs = create_solver_inputs(problems, solver_templates)

        backend_cfg = LLMBackendConfig(
            **OmegaConf.to_container(solver_cfg.backend, resolve=True),
        )
        sampling_cfg = LLMSamplingConfig(
            **OmegaConf.to_container(solver_cfg.sampling_params, resolve=True),
        )
        
        solver_llm_host = LLMHost(cfg=backend_cfg, sampling_params=sampling_cfg)
        solver_llm_host.start()
        try:
            solver_outputs = eval_solver(
                solver_inputs=solver_inputs,
                llm_host=solver_llm_host,
                extract_field_fn=extract_field_fn,
                backend_cfg=backend_cfg,
                sampling_cfg=sampling_cfg,
            )
        finally:
            solver_llm_host.stop()

        if cfg.outputs.get("save_solver_outputs", False):
            experiment_id = cfg.get("experiment_id", "base")
            save_solver_outputs(save_dir, solver_outputs, experiment_id=experiment_id)

    else:
        if solver_outputs_file is None:
            raise ValueError(
                "In 'judger_only' mode you must set cfg.data.solver_output_file"
            )
        solver_outputs = load_solver_outputs(solver_outputs_file)

    # =====================
    # Judger Evaluation
    # =====================
    judger_inputs: List[JudgerInput] = []

    if mode in ("judger_only", "all"):
        judger_cfg = cfg.judger

        backend_cfg = LLMBackendConfig(
            **OmegaConf.to_container(judger_cfg.backend, resolve=True),
        )
        sampling_cfg = LLMSamplingConfig(
            **OmegaConf.to_container(judger_cfg.sampling_params, resolve=True),
        )

        extract_field_fn = get_field_extractor_function(
            cfg.template.extract_field_func_for_judger
        )

        judger_templates = cfg.template.get("judger_templates", [])
        judger_inputs = create_judger_inputs(solver_outputs, judger_templates)

        judger_llm_host = LLMHost(cfg=backend_cfg, sampling_params=sampling_cfg)
        judger_llm_host.start()
        try:
            judger_outputs = eval_judger(
                judger_inputs=judger_inputs,
                llm_host=judger_llm_host,
                extract_field_fn=extract_field_fn,
                backend_cfg=backend_cfg,
                sampling_cfg=sampling_cfg,
            )
        finally:
            judger_llm_host.stop()

    # =====================
    # Make Results
    # =====================
    results = [
        {
            "meta_info": {
                "model_name": jo.judger_model_name,
                "temperature": jo.judger_temperature,
                "top_p": jo.judger_top_p,
                "top_k": jo.judger_top_k,
                "max_tokens": jo.judger_max_tokens
            },
            "problem": {
                "meta_info": {
                    "source": jo.source,
                    "topic": jo.topic
                },
                "task": jo.task,
                "images": jo.images,
                "target_solution": jo.target_solution,
                "target_answer": jo.target_answer,
                "global_eval_criteria": jo.global_eval_criteria,
                "task_eval_criteria": jo.task_eval_criteria
            },
            "solver_result": {
                "meta_info": {
                    "model_name": jo.solver_model_name,
                    "temperature": jo.solver_temperature,
                    "top_p": jo.solver_top_p,
                    "top_k": jo.solver_top_k,
                    "max_tokens": jo.solver_max_tokens
                },
                "template_name": jo.solver_template_name,
                "prompt": jo.solver_prompt,
                "solution": jo.solver_solution,
                "extracted_fields": jo.solver_extracted_fields
            },
            "judger_result": {
                "meta_info": {
                    "model_name": jo.judger_model_name,
                    "temperature": jo.judger_temperature,
                    "top_p": jo.judger_top_p,
                    "top_k": jo.judger_top_k,
                    "max_tokens": jo.judger_max_tokens
                },
                "template_name": jo.judger_template_name,
                "prompt": jo.judger_prompt,
                "solution": jo.judger_solution,
                "extracted_fields": jo.judger_extracted_fields
            },
            "extra_kwargs": {
                **jo._extra
            }
        }
        for jo in judger_outputs
    ]

    results_file = save_dir / "results.json"
    with open(str(results_file), "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=False
        )

    logger.info("Evaluation finished.")


if __name__ == "__main__":
    main()
