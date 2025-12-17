# Examples of Usage

### Data Prepare (evaluation.mode="all")

### Data Prepare (evaluation.mode="judger_only")

### Example №1

- `start_examples/start_example1.sh`

### Example №2

- `start_examples/start_example2.sh`

### Example №3

- `start_examples/start_example3.sh`

# Essence

## 1 Templates

CriticEval uses Jinja2 templates to render prompts for both the solver and the judger.

### Where templates live

All built-in templates are stored in:

- `criticeval/templates/templates/`

When you add a new template, place it in the same directory (and follow the conventions used by the existing ones).

### Rendering inputs

Templates are rendered using typed input structures:

- **`SolverInput`** — the data structure used to render the solver prompt.
- **`JudgerInput`** — the data structure used to render the judger prompt.

In practice, this means that your template variables should correspond to fields available on these structures (and any additional context the framework injects at render time).

### Selecting templates via config

Template selection is controlled via the config:

```yaml
template:
  solver_templates: ["base_solver"]
  judger_templates: ["base_judger"]
```

## 2 Extractors Registry

CriticEval supports extracting structured fields from LLM responses (both solver and judger). These extracted fields are then persisted in output objects and can be reused downstream (e.g., solver-extracted fields can be injected into the judger prompt).

#### Where extractors live

Built-in extractor functions are defined in:

- `criticeval/extractors/extractors.py`

#### Registering a custom extractor

To add your own extraction logic, define a function that accepts the raw model response (`str`) and returns a `dict` of extracted fields. Then register it with the extractor registry:

```python
from criticeval.extractors import register

@register(name="nothing")
def nothing_extractor(response: str) -> dict:
    return {}
```

#### Selecting extractors via config

Extractor selection is controlled via the config:

```yaml
template:
  extract_field_func_for_solver: "nothing"
  extract_field_func_for_judger: "nothing"
```


3. Input&Output Data:







