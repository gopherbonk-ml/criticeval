import logging
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import List

from .structure import Problem, Solution, SolverOutput, JudgerInput, Results
from .templates.formatter import render_solver_prompt, render_judger_prompt
from .serve_model.model_api import (
    LLMHost,
    LLMBackendConfig,
    LLMSamplingConfig,
)
from .extractors import get_answer_extractor_function
from .utils import read_input_data, check_or_create_output_dir, save_results
import multiprocessing as mp
import ray


logger = logging.getLogger("criticeval")
logging.basicConfig(format="[CriticEval] %(message)s")


def ray_init():
    if not ray.is_initialized():
        # # ray_init_kwargs = {
        # #     "num_cpus": 32,
        # #     "num_gpus": 8
        # # }
        # # runtime_env_kwargs = {}
        
        # # ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env_kwargs})
        # ray.init(**OmegaConf.to_container(ray_init_kwargs))
        ray.init()
    print(f"Availavle Ray Resources: {ray.cluster_resources()}")


def save_solver_outputs(output_dir: str, solver_outputs: List[SolverOutput], experiment_id: str = "base"):
    outp = Path(output_dir) / f"solver_outputs_{experiment_id}"
    outp.mkdir(parents=True, exist_ok=True)

    serialized = []
    for so in solver_outputs:
        serialized.append(OmegaConf.to_container(so))

    output_file = outp / "solver_outputs.json"
    with open(output_file, "w", encoding="utf-8") as f:
        import json
        json.dump(serialized, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Saved solver outputs to {output_file}")


def save_results(output_dir: str, results: List[Results], experiment_id: str = "base"):
    pass


def eval_solver(cfg: DictConfig):
    extract_answer_fn = get_answer_extractor_function(
        cfg.template.extract_answer_func if cfg.template.get("use_extract_answer", False) else "base_answer_extractor"
    )

    problems: list[Problem] = []
    data_dir = cfg.paths.data_dir or ""

    for problem_file in cfg.data.problems_files:
        problem_path = Path(data_dir) / problem_file
        problems.extend(read_input_data(problem_path))

    templates_dir = cfg.paths.get("templates_dir", None)

    solver_cfg = LLMBackendConfig(
        **OmegaConf.to_container(cfg.solver.backend, resolve=True)
    )
    solver_sampling = LLMSamplingConfig(
        **OmegaConf.to_container(cfg.solver.sampling_params, resolve=True)
    )

    num_workers = 4
    workers = [LLMHost.options(resources={"NPU": 2}).remote(solver_cfg, solver_sampling) for _ in range(num_workers)]
    ray.get([w.start.remote() for w in workers])
    
    solver_outputs: List[SolverOutput] = []
    for i, problem in tqdm(enumerate(problems), desc="Generating Solutions"):
        worker = workers[i % num_workers]

        solver_prompt = render_solver_prompt(problem, cfg.template.solver_template, templates_dir)
        solver_solution = ray.get(worker.generate.remote(solver_prompt, images=problem.images))
        solver_answer = extract_answer_fn(solver_solution)
        solver_output = OmegaConf.merge(
            problem, 
            **{
                "solver_model_name": cfg.solver.backend.vllm.model,
                "solver_temperature": cfg.solver.sampling_params.temperature,
                "solver_top_p": cfg.solver.sampling_params.top_p,
                "solver_top_k": cfg.solver.sampling_params.top_k,
                "solver_max_tokens": cfg.solver.sampling_params.max_tokens,
                "llm_solution": solver_solution,
                "llm_answer": solver_answer
            }
        )
        solver_outputs.append(SolverOutput(**OmegaConf.to_container(solver_output)))
    
    # Save Results
    if cfg.outputs.save_solver_outputs:
        save_solver_outputs(cfg.paths.save_dir, solver_outputs, experiment_id="base")

    return solver_outputs


def eval_judger(
    cfg: DictConfig,
    problems: List[Problem],
    solver_outputs: List[SolverOutput],
    workers: List[ray.actor.ActorHandle]
):
    pass


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger.info("CriticEval Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    ray_init()
    eval_solver(cfg)
    # check_or_create_output_dir(output_dir)

    # # Load problems
    # problems: list[Problem] = []
    # data_dir = cfg.paths.data_dir or ""
    # for problem_file in cfg.data.problems_files:
    #     problem_path = Path(data_dir) / problem_file
    #     problems.extend(read_input_data(problem_path))

    # templates_dir = cfg.paths.get("templates_dir", None)

    # # Initialize solver
    # solver_cfg = LLMBackendConfig(
    #     **OmegaConf.to_container(cfg.solver.backend, resolve=True)
    # )

    # solver_sampling = LLMSamplingConfig(
    #     **OmegaConf.to_container(cfg.solver.sampling_params, resolve=True)
    # )
    # solver_llm = LLMHost(solver_cfg, solver_sampling)

    # logger.info("Starting solver LLM host")
    # solver_llm.start()

    # # Generate solutions
    # extract_answer_fn = get_answer_extractor_function(
    #     cfg.template.extract_answer_func if cfg.template.get("use_extract_answer", False) else "base_answer_extractor"
    # )

    # llm_solutions: list[Solution] = []
    # for problem in tqdm(problems, desc="Generating Solutions"):
    #     solver_prompt = render_solver_prompt(problem, cfg.template.solver_template, templates_dir)
    #     solution_text = solver_llm.generate(solver_prompt, images=problem.images)
    #     answer = extract_answer_fn(solution_text)
    #     llm_solutions.append(
    #         Solution(solution=solution_text, answer=answer)
    #     )

    # solver_llm.stop()
    # del solver_llm, solver_cfg

    # # Initialize judger
    # judger_cfg = LLMBackendConfig(
    #     **OmegaConf.to_container(cfg.judger.backend, resolve=True)
    # )

    # judger_sampling = LLMSamplingConfig(
    #     **OmegaConf.to_container(cfg.judger.sampling_params, resolve=True)
    # )
    # judger_llm = LLMHost(judger_cfg, judger_sampling)

    # logger.info("Starting judger LLM host")
    # judger_llm.start()

    # # Evaluate solutions
    # results: list[Results] = []
    # for problem, llm_solution in tqdm(list(zip(problems, llm_solutions)), desc="Evaluating Solutions"):
    #     judger_prompt = render_judger_prompt(problem, llm_solution, cfg.template.judger_template, templates_dir)
    #     judger_output = judger_llm.generate(judger_prompt, images=problem.images)
    #     judger_assesment = extract_answer_fn(judger_output)
    #     results.append(
    #         Results(
    #             problem=problem,
    #             llm_solution=llm_solution,
    #             judger_input=judger_prompt,
    #             judger_output=judger_output,
    #             judger_assesment=judger_assesment
    #         )
    #     )

    # judger_llm.stop()

    # # Save results
    # logger.info(f"Completed evaluation of {len(results)} problems — saving to {output_dir}")
    # save_results(output_dir, results)


if __name__ == "__main__":
    main()