#!/usr/bin/env bash


python -m criticeval.eval \
  paths.data_dir="data" \
  data.problems_files='["problems.csv"]' \
  paths.save_dir="/Users/gopherbonk/criticeval/outputs" \
  paths.templates_dir="/Users/gopherbonk/criticeval/criticeval/templates/templates" \
  template.solver_template="base_solver.jinja" \
  template.judger_template="base_judger.jinja" \
  template.use_extract_answer=True \
  template.extract_answer_func="boxed_answer_extractor" \
  solver.backend.backend_module="vllm"