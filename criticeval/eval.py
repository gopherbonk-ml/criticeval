import logging
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .structure import Problem, Solution, JudgerInput, Results
from .templates.formatter import render_solver_prompt, render_judger_prompt
from .serve_model.model_api import (
    LLMHost,
    LLMBackendConfig,
    LLMSamplingConfig,
)
from .extractors import get_answer_extractor_function
from .utils import read_input_data, check_or_create_output_dir, save_results


logger = logging.getLogger("criticeval")
logging.basicConfig(format="[CriticEval] %(message)s")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    output_dir = to_absolute_path(cfg.paths.save_dir)
    check_or_create_output_dir(output_dir)

    # Load problems
    problems: list[Problem] = []
    data_dir = cfg.paths.data_dir or ""
    for problem_file in cfg.data.problems_files:
        problem_path = Path(data_dir) / problem_file
        problems.extend(read_input_data(problem_path))

    templates_dir = cfg.paths.get("templates_dir", None)

    # Initialize solver
    solver_sampling = LLMSamplingConfig(cfg.solver.sampling_params)
    solver_cfg = LLMBackendConfig(cfg.solver.backend)
    solver_llm = LLMHost(solver_cfg, solver_sampling)

    logger.info("Starting solver LLM host")
    solver_llm.start()

    # Generate solutions
    extract_answer_fn = get_answer_extractor_function(
        cfg.template.extract_answer_func if cfg.template.get("use_extract_answer", False) else "base_answer_extractor"
    )

    llm_solutions: list[Solution] = []
    for problem in tqdm(problems, desc="Generating Solutions"):
        solver_prompt = render_solver_prompt(problem, cfg.template.solver_template, templates_dir)
        solution_text = solver_llm.generate(solver_prompt, images=problem.images)
        answer = extract_answer_fn(solution_text)
        llm_solutions.append(
            Solution(solution=solution_text, answer=answer)
        )

    solver_llm.stop()
    del solver_llm, solver_cfg

    # Initialize judger
    judger_sampling = LLMSamplingConfig(cfg.judger.sampling_params)
    judger_cfg = LLMBackendConfig(cfg.judger.backend)
    judger_llm = LLMHost(judger_cfg, judger_sampling)

    logger.info("Starting judger LLM host")
    judger_llm.start()

    # Evaluate solutions
    results: list[Results] = []
    for problem, llm_solution in tqdm(list(zip(problems, llm_solutions)), desc="Evaluating Solutions"):
        judger_prompt = render_judger_prompt(problem, llm_solution, cfg.template.judger_template, templates_dir)
        judger_output = judger_llm.generate(judger_prompt, images=problem.images)
        results.append(
            Results(
                problem=problem,
                llm_solution=llm_solution,
                judger_input=judger_prompt,
                judger_output=judger_output,
            )
        )

    judger_llm.stop()

    # Save results
    logger.info(f"Completed evaluation of {len(results)} problems — saving to {output_dir}")
    save_results(output_dir, results)


if __name__ == "__main__":
    main()
