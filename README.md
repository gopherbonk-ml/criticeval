**Essence**:

1. Template Set:

criticeval/templates/templates contains a set of templates.

When creating your template, the structures used to render it will be useful.

The **SolverInput** structure is used to render the prompt for the solver.
The **JudgerInput** structure is used to render the prompt for the judger.


The choice of template is specified in the config parameters: 

- template.solver_templates=["base_solver"]
- template.judger_templates=["base_judger"]

It is possible to set several templates for evaluation at once.

2. Extractors Registry

Available functions for response extraction are available in criticeval/extractors/extractors.py

You can register your own function for extract using decorator:

```
@register(name="base_answer_extractor")
def base_answer_extractor(response):
    return response
```

The selection and use of the function for the extract is determined by the parameters:

- template.use_extract_answer_for_solver=True 
- template.extract_answer_func_for_solver="boxed_answer_extractor"
- template.use_extract_answer_for_judger=True 
- template.extract_answer_func_for_judger="boxed_answer_extractor"

3. Input Data:






EXAMPLES:

Basic Example 1:

Solver (vllm-npu) and Judger (vllm-npu)

Basic Example 2:

Solver (vllm-npu) and Judger (API)

Basic Example 3:

ONLY Judger (API) with Solver Output (data.data_dir / data.solver_output_file).
Result will be save in

Basic Example 4:





