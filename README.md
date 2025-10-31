# Cognitive Load Traces (CLT)

A mid-level interpretability framework for deep transformer models, inspired by Cognitive Load Theory in human cognition.

## Overview

Cognitive Load Traces (CLTs) provide symbolic and visual accounts of how deep models allocate internal resources during reasoning tasks. The framework formalizes model cognition through three components:

- **Intrinsic Load (IL)**: Task-inherent difficulty measured by attention entropy and representation dispersion
- **Extraneous Load (EL)**: Process inefficiency measured by KV-cache miss ratio and decoding stability
- **Germane Load (GL)**: Schema-building effort measured by consolidation patterns and concept reuse

## Key Features

1. **Symbolic Framework**: Formal three-component stochastic process `CLT_t = (IL_t, EL_t, GL_t)`
2. **Visualization Tools**: Temporal curves, simplex diagrams, heatmaps, and radar profiles
3. **Load-Guided Interventions**: Adaptive decoding strategies that improve reasoning efficiency by 15-30%
4. **Error Prediction**: CLI spikes predict 73% of reasoning errors before they occur

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from cogload import CognitiveLoadTraces, CLTVisualizer, LoadGuidedDecoding

# Load model
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Initialize CLT framework
clt_computer = CognitiveLoadTraces(
    w_cli=(0.4, 0.4, 0.2)  # weights for IL, EL, GL
)

# Compute CLT for input
inputs = tokenizer("Solve: 2x + 5 = 15", return_tensors="pt")
clt_trace, cli_trace, metadata = clt_computer.compute_clt_trace(
    model, inputs.input_ids, attention_mask=inputs.attention_mask
)

# Visualize
visualizer = CLTVisualizer()
visualizer.plot_temporal_traces(clt_trace, cli_trace, save_path="clt_trace.png")
visualizer.plot_simplex(clt_trace, save_path="simplex.png")
```

### Load-Guided Decoding

```python
# Initialize LGD system
lgd_system = LoadGuidedDecoding(
    tau_warn=0.6,  # Warning threshold
    tau_act=0.8    # Action threshold
)

# Apply interventions
intervention_history, flags = lgd_system.apply_interventions(
    clt_trace, cli_trace, model
)

# Generate with interventions
output_ids, intervention_log = lgd_system.decode_with_interventions(
    model, input_ids, clt_computer, max_new_tokens=100
)
```

## Examples

### GSM8K Math Reasoning

Run the GSM8K example to analyze mathematical reasoning:

```bash
python examples/gsm8k_example.py
```

This demonstrates:
- CLT computation for multi-step math problems
- Error step identification through CLI spikes
- Load-guided interventions for improved accuracy

### XSum Summarization

Run the XSum example to analyze summarization:

```bash
python examples/xsum_example.py
```

This demonstrates:
- Phase identification (planning/search/consolidation)
- Simplex visualization of cognitive strategies
- Load-guided improvements in summary quality

## Architecture

```
cogload/
├── __init__.py          # Package initialization
├── clt.py              # Core CLT computation
├── visualization.py    # Visualization tools
└── interventions.py    # Load-guided decoding
```

## Cognitive Load Components

### Intrinsic Load (IL)

Captures inherent task difficulty:

```python
H_t = (1/L) Σ_l (-Σ_i a_{t,i}^l log a_{t,i}^l)  # Attention entropy
Disp_t = (1/L) Σ_l ||h_t^l - h̄_t|| / ||h̄_t||   # Representation dispersion
IL_t = α₁ Ĥ_t + α₂ Disp̂_t
```

### Extraneous Load (EL)

Measures process inefficiency:

```python
Miss_t = 1 - hits_t / queries_t                 # KV-cache miss ratio
Stab_t = KL(p_t || p̃_t)                        # Decoding stability
EL_t = β₁ Misŝ_t + β₂ Stab̂_t
```

### Germane Load (GL)

Tracks schema construction:

```python
Consol_t = (1/(L-1)) Σ_l cos(Δh_t^{l+1}, Δh_t^l)  # Consolidation
Reuse_t = Σ_i 1[a_{t,i}^max > θ] · 1[concept(i)=active] / (Σ_i 1[a_{t,i}^max > θ] + ε)
GL_t = γ₁ (1 - Consol̂_t) + γ₂ (1 - Reusê_t)
```

## Visualization Types

1. **Temporal Traces**: IL/EL/GL/CLI curves over decoding steps
2. **Simplex Diagrams**: Geometric representation in IL-EL-GL space
3. **Layer-Time Heatmaps**: Load distribution across model depth
4. **Radar Profiles**: Overall cognitive characteristics
5. **Correlation Analysis**: Relationship between CLT and errors

## Load-Guided Interventions

The framework implements adaptive interventions:

- **Cache Stabilization**: Reduces EL spikes through proactive cache management
- **Decoding Control**: Improves stability via temperature scaling
- **Planning Aid**: Decomposes tasks to lower IL
- **Consolidation Aid**: Strengthens connections to reduce GL
- **Token Merging**: Reduces computational overhead

## Experimental Results

On GSM8K and XSum benchmarks:

| Method | GSM8K (Acc) | XSum (ROUGE-L) | CLI Correlation |
|--------|-------------|----------------|-----------------|
| Baseline | 65.1 | 29.3 | - |
| CLT + LGD | **70.2** | **33.9** | **0.87** |

Key findings:
- 73% of reasoning errors coincide with EL spikes > 0.8
- LGD improves efficiency by 15-30% while maintaining accuracy
- Larger models show stronger CLI correlations across scales

## Citation

If you use this framework, please cite:

```bibtex
@article{liu2025cognitive,
  title={Cognitive Load Traces as Symbolic and Visual Accounts of Deep Model Cognition},
  author={Liu, Dong and Yu, Yanxuan},
  journal={CogInterp @ NeurIPS 2025},
  year={2025}
}
```

## License

This project is licensed under CC BY 4.0.

## Acknowledgments

Inspired by Cognitive Load Theory (Sweller, 1988; Paas & van Merriënboer, 1993) and built on modern transformer architectures.

## Future Directions

- [ ] Extend to multimodal reasoning tasks
- [ ] Real-time intervention systems for production
- [ ] Architecture-aware CLTs for mixture-of-experts models
- [ ] Data-centric safety frameworks using load signals

