## Essence

### 1 Templates

CriticEval uses Jinja2 templates to render prompts for both the solver and the judger.

#### 1.1 Where templates live

All built-in templates are stored in:

- `criticeval/templates/templates/`

When you add a new template, place it in the same directory (and follow the conventions used by the existing ones).

#### 1.2 Rendering inputs

Templates are rendered using typed input structures:

- **`SolverInput`** — the data structure used to render the solver prompt.
- **`JudgerInput`** — the data structure used to render the judger prompt.

In practice, this means that your template variables should correspond to fields available on these structures (and any additional context the framework injects at render time).

#### 1.3 Selecting templates via config

Template selection is controlled via the config:

The choice of template is specified in the config parameters: 

```yaml
template:
  solver_templates: ["base_solver"]
  judger_templates: ["base_judger"]
```

### 2 Extractors Registry

CriticEval supports extracting structured fields from LLM responses (both solver and judger). These extracted fields are then persisted in output objects and can be reused downstream (e.g., solver-extracted fields can be injected into the judger prompt).

#### 2.1 Where extractors live

Built-in extractor functions are defined in:

- `criticeval/extractors/extractors.py`

#### 2.2 Registering a custom extractor

To add your own extraction logic, define a function that accepts the raw model response (`str`) and returns a `dict` of extracted fields. Then register it with the extractor registry:

```python
from criticeval.extractors import register

@register(name="nothing")
def nothing_extractor(response: str) -> dict:
    return {}
```

#### 2.3 Selecting extractors via config



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





