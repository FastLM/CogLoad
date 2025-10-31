# Quick Start Guide

Get started with Cognitive Load Traces in 5 minutes.

## Installation

```bash
# Clone repository
git clone https://github.com/yale-nlp/CogLoad.git
cd CogLoad

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Basic Usage

### 1. Load a Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
```

### 2. Initialize CLT Framework

```python
from cogload import CognitiveLoadTraces

clt_computer = CognitiveLoadTraces(
    w_cli=(0.4, 0.4, 0.2)  # IL, EL, GL weights
)
```

### 3. Compute CLT Trace

```python
# Prepare input
text = "Solve: 2x + 5 = 15. Show your work step by step."
inputs = tokenizer(text, return_tensors="pt")

# Compute CLT
clt_trace, cli_trace, metadata = clt_computer.compute_clt_trace(
    model,
    inputs.input_ids,
    attention_mask=inputs.attention_mask
)

print(f"CLT shape: {clt_trace.shape}")
print(f"CLI shape: {cli_trace.shape}")
```

### 4. Visualize Results

```python
from cogload import CLTVisualizer

visualizer = CLTVisualizer()

# Temporal traces
visualizer.plot_temporal_traces(
    clt_trace, cli_trace,
    save_path="clt_trace.png"
)

# Simplex diagram
visualizer.plot_simplex(
    clt_trace,
    save_path="clt_simplex.png"
)
```

### 5. Apply Load-Guided Interventions

```python
from cogload import LoadGuidedDecoding

# Initialize LGD
lgd_system = LoadGuidedDecoding(
    tau_warn=0.6,
    tau_act=0.8
)

# Apply interventions
intervention_history, flags = lgd_system.apply_interventions(
    clt_trace, cli_trace, model
)

# Print results
print(f"Applied {len(intervention_history)} interventions")
for interv in intervention_history[:5]:
    print(f"  Step {interv['step']}: {interv['intervention']}")
```

## Example: GSM8K Math Problem

```python
from cogload import CognitiveLoadTraces, CLTVisualizer, LoadGuidedDecoding

# Initialize framework
clt_computer = CognitiveLoadTraces()
visualizer = CLTVisualizer()
lgd_system = LoadGuidedDecoding()

# Math problem
problem = """
Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning.
She bakes 4 for her friends every day. She sells the remainder at the farmers' market
for $2 per duck egg. How much in dollars does she make every day?
"""

prompt = f"Question: {problem}\nAnswer: Let's think step by step."
inputs = tokenizer(prompt, return_tensors="pt")

# Compute CLT
clt_trace, cli_trace, _ = clt_computer.compute_clt_trace(
    model, inputs.input_ids, attention_mask=inputs.attention_mask
)

# Identify high-load steps
error_steps = [i for i, cli in enumerate(cli_trace) if cli > 0.8]
print(f"High-load steps: {error_steps}")

# Visualize
visualizer.plot_temporal_traces(
    clt_trace, cli_trace,
    error_events=error_steps,
    save_path="math_problem_clt.png"
)
```

## Example: Summarization

```python
# Article to summarize
article = """
The UK's education secretary announced plans to increase funding for primary schools
by £1.3bn next year. The money will be used to reduce class sizes and improve
infrastructure. Critics say the funding is insufficient given inflation rates.
"""

prompt = f"Summarize the following article in one sentence: {article}\n\nSummary:"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

# Compute CLT
clt_trace, cli_trace, _ = clt_computer.compute_clt_trace(
    model, inputs.input_ids, attention_mask=inputs.attention_mask
)

# Identify cognitive phases
phases = {
    'planning': [],
    'search': [],
    'consolidation': []
}

for i, (il, el, gl) in enumerate(clt_trace):
    if gl > 0.7 and el < 0.4:
        phases['planning'].append(i)
    elif el > 0.7 and gl < 0.4:
        phases['search'].append(i)
    elif abs(il - 0.33) < 0.2:
        phases['consolidation'].append(i)

print(f"Planning: {len(phases['planning'])} steps")
print(f"Search: {len(phases['search'])} steps")
print(f"Consolidation: {len(phases['consolidation'])} steps")
```

## Running Examples

```bash
# GSM8K example
python examples/gsm8k_example.py

# XSum example
python examples/xsum_example.py

# Benchmark evaluation
python benchmarks/evaluate_clt.py --model mistralai/Mistral-7B-Instruct-v0.2 --n_samples 50

# Run tests
pytest tests/
```

## Customizing Interventions

Create custom interventions:

```python
from cogload.interventions import Intervention
import numpy as np

class CustomIntervention(Intervention):
    def apply(self, model, state_dict):
        # Your intervention logic
        state_dict['my_intervention'] = True
        return state_dict
    
    def score(self, clt, history):
        # Return relevance score
        return clt[1]  # Use EL component
    
# Use in LGD
lgd_system = LoadGuidedDecoding(
    interventions=[
        CustomIntervention(),
        # ... other interventions
    ]
)
```

## Understanding Results

### Load Components

- **IL (Intrinsic Load)**: High during complex reasoning steps
- **EL (Extraneous Load)**: High when model is inefficient
- **GL (Germane Load)**: High during schema building

### CLI (Composite Load Index)

Weighted combination: `CLI = 0.4*IL + 0.4*EL + 0.2*GL`

### Visualizations

- **Temporal Traces**: Shows load evolution over time
- **Simplex**: Geometric view of cognitive strategy
- **Heatmaps**: Layer-wise load distribution
- **Correlation**: Relationship between load and errors

## Next Steps

- Read [full documentation](README.md)
- Explore [theory](docs/theory.md)
- Check [API reference](docs/api_reference.md)
- Run [examples](examples/)
- Run [benchmarks](benchmarks/)

## Troubleshooting

### Model Loading Issues

Use smaller models for testing:
```python
model_name = "microsoft/phi-2"  # Small demo model
```

### Memory Issues

Use smaller sequences:
```python
inputs = tokenizer(text, max_length=256, truncation=True, return_tensors="pt")
```

### Visualization Not Showing

Save to file instead:
```python
visualizer.plot_temporal_traces(clt_trace, cli_trace, save_path="output.png")
```

## Citation

```bibtex
@article{liu2025cognitive,
  title={Cognitive Load Traces as Symbolic and Visual Accounts of Deep Model Cognition},
  author={Liu, Dong and Yu, Yanxuan},
  journal={CogInterp @ NeurIPS 2025},
  year={2025}
}
```

## Support

- Issues: https://github.com/yale-nlp/CogLoad/issues
- Discussions: https://github.com/yale-nlp/CogLoad/discussions

