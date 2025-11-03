# Cognitive Load Traces (CLT)

A mid-level interpretability framework for deep transformer models, inspired by Cognitive Load Theory.

## Overview

CLTs analyze how models allocate internal resources during reasoning through three components:

- **IL (Intrinsic Load)**: Task difficulty via attention entropy & representation dispersion
- **EL (Extraneous Load)**: Process inefficiency via KV-cache misses & decoding stability  
- **GL (Germane Load)**: Schema-building via consolidation & concept reuse

## Features

- **Symbolic Framework**: `CLT_t = (IL_t, EL_t, GL_t)`
<!-- - **Visualization**: Temporal curves, simplex diagrams, heatmaps -->
- **Load-Guided Interventions**: 15-30% efficiency improvement
- **Error Prediction**: 73% error correlation with CLI spikes

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from cogload import CognitiveLoadTraces, CLTVisualizer, LoadGuidedDecoding

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Compute CLT
clt = CognitiveLoadTraces()
inputs = tokenizer("Solve: 2x + 5 = 15", return_tensors="pt")
clt_trace, cli_trace, _ = clt.compute_clt_trace(model, inputs.input_ids, attention_mask=inputs.attention_mask)

# Visualize
viz = CLTVisualizer()
viz.plot_temporal_traces(clt_trace, cli_trace, save_path="trace.png")
viz.plot_simplex(clt_trace, save_path="simplex.png")

# Load-guided decoding
lgd = LoadGuidedDecoding(tau_warn=0.6, tau_act=0.8)
intervention_history, _ = lgd.apply_interventions(clt_trace, cli_trace, model)
```

## Examples

```bash
python examples/gsm8k_example.py   # Math reasoning
python examples/xsum_example.py    # Summarization
```

## Components

**IL**: `H_t = (1/L) Σ entropy` + `Disp_t = ||h - h̄||`  
**EL**: `Miss_t = cache_misses/queries` + `Stab_t = KL(p_t || p_{t-1})`  
**GL**: `Consol_t = cos(Δh^l, Δh^{l+1})` + `Reuse_t = active_concepts/total`

## Interventions

- Cache Stabilization (high EL)
- Decoding Control (high EL)
- Planning Aid (high IL)
- Consolidation Aid (high GL)
- Token Merging (general)

## Results

| Method | GSM8K | XSum | CLI Corr |
|--------|-------|------|----------|
| Baseline | 65.1 | 29.3 | - |
| CLT + LGD | **70.2** | **33.9** | **0.87** |

- 73% errors align with EL spikes > 0.8
- 15-30% efficiency improvement
- Stronger correlations in larger models

## Citation

```bibtex
@article{liu2025cognitive,
  title={Cognitive Load Traces as Symbolic and Visual Accounts of Deep Model Cognition},
  author={Liu, Dong and Yu, Yanxuan},
  journal={CogInterp @ NeurIPS 2025},
  year={2025}
}
```
