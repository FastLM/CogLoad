# API Reference

Complete API documentation for the CLT framework.

## Core Classes

### CognitiveLoadTraces

Main class for computing Cognitive Load Traces.

```python
from cogload import CognitiveLoadTraces

clt = CognitiveLoadTraces(
    alpha_il=(0.5, 0.5),
    beta_el=(0.5, 0.5),
    gamma_gl=(0.5, 0.5),
    w_cli=(0.4, 0.4, 0.2),
    theta_attention=0.1,
    normalize=True
)
```

#### Parameters

- `alpha_il` (tuple, default: (0.5, 0.5)): Weights for attention entropy and dispersion in IL
- `beta_el` (tuple, default: (0.5, 0.5)): Weights for miss ratio and stability in EL
- `gamma_gl` (tuple, default: (0.5, 0.5)): Weights for consolidation and reuse in GL
- `w_cli` (tuple, default: (0.4, 0.4, 0.2)): Weights for IL, EL, GL in composite load index
- `theta_attention` (float, default: 0.1): Threshold for attention-based concept detection
- `normalize` (bool, default: True): Apply robust normalization

#### Methods

##### `compute_clt_trace(model, input_ids, attention_mask, kv_cache, concept_map)`

Compute complete CLT trace over sequence.

**Parameters:**
- `model`: Transformer model
- `input_ids` (torch.Tensor): Input token IDs [batch_size, seq_len]
- `attention_mask` (torch.Tensor): Attention mask [batch_size, seq_len]
- `kv_cache` (dict, optional): KV cache for tracking misses
- `concept_map` (dict, optional): Position-to-concept mapping

**Returns:**
- `clt_trace` (np.ndarray): [T, 3] array of (IL_t, EL_t, GL_t)
- `cli_trace` (np.ndarray): [T] array of composite load indices
- `metadata` (dict): Raw measurements

##### `compute_intrinsic_load(attention_weights, hidden_states, step, sequence_length)`

Compute Intrinsic Load (IL).

**Parameters:**
- `attention_weights` (torch.Tensor): [num_layers, num_heads, seq_len]
- `hidden_states` (torch.Tensor): [num_layers, seq_len, hidden_dim]
- `step` (int): Current decoding step
- `sequence_length` (int): Total sequence length

**Returns:**
- `float`: IL value

##### `compute_extraneous_load(kv_cache_misses, total_queries, current_probs, previous_probs)`

Compute Extraneous Load (EL).

**Parameters:**
- `kv_cache_misses` (int): Number of cache misses
- `total_queries` (int): Total queries
- `current_probs` (torch.Tensor): Current decoding probabilities
- `previous_probs` (torch.Tensor, optional): Previous probabilities

**Returns:**
- `float`: EL value

##### `compute_germane_load(hidden_states, attention_weights, step, concept_map)`

Compute Germane Load (GL).

**Parameters:**
- `hidden_states` (torch.Tensor): [num_layers, seq_len, hidden_dim]
- `attention_weights` (torch.Tensor): [num_layers, num_heads, seq_len]
- `step` (int): Current step
- `concept_map` (dict, optional): Concept mapping

**Returns:**
- `float`: GL value

##### `identify_load_type(il, el, gl)`

Identify dominant load type.

**Parameters:**
- `il`, `el`, `gl` (float): Load component values

**Returns:**
- `str`: Dominant load type ('IL', 'EL', or 'GL')

### CLTVisualizer

Visualization toolkit for CLT analysis.

```python
from cogload import CLTVisualizer

visualizer = CLTVisualizer(style='seaborn-v0_8')
```

#### Methods

##### `plot_temporal_traces(clt_trace, cli_trace, error_events, save_path, figsize)`

Plot IL/EL/GL/CLI curves over time.

**Parameters:**
- `clt_trace` (np.ndarray): [T, 3] CLT values
- `cli_trace` (np.ndarray): [T] CLI values
- `error_events` (list, optional): Error step indices
- `save_path` (str, optional): Output path
- `figsize` (tuple, default: (12, 6)): Figure size

**Returns:**
- `fig`, `ax`: Matplotlib figure and axis

##### `plot_simplex(clt_trace, labels, save_path, figsize)`

Plot IL-EL-GL simplex diagram.

**Parameters:**
- `clt_trace` (np.ndarray): [T, 3] CLT values
- `labels` (list, optional): Point labels
- `save_path` (str, optional): Output path
- `figsize` (tuple, default: (10, 10)): Figure size

**Returns:**
- `fig`, `ax`: Matplotlib figure and axis

##### `plot_layer_time_heatmap(layer_loads, save_path, figsize)`

Plot heatmap of load across layers and time.

**Parameters:**
- `layer_loads` (np.ndarray): [num_layers, T] load values
- `save_path` (str, optional): Output path
- `figsize` (tuple, default: (12, 6)): Figure size

**Returns:**
- `fig`, `ax`: Matplotlib figure and axis

##### `plot_radar_profile(load_profile, save_path, figsize)`

Plot radar/spider chart of cognitive characteristics.

**Parameters:**
- `load_profile` (dict): Metric → value mapping
- `save_path` (str, optional): Output path
- `figsize` (tuple, default: (8, 8)): Figure size

**Returns:**
- `fig`, `ax`: Matplotlib figure and axis

##### `plot_correlation_analysis(clt_trace, cli_trace, error_events, save_path, figsize)`

Plot correlation analysis between CLT and errors.

**Parameters:**
- `clt_trace` (np.ndarray): [T, 3] CLT values
- `cli_trace` (np.ndarray): [T] CLI values
- `error_events` (list): Error indices
- `save_path` (str, optional): Output path
- `figsize` (tuple, default: (14, 6)): Figure size

**Returns:**
- `fig`, `axes`: Matplotlib figure and axes

##### `create_interactive_trace(clt_trace, cli_trace, save_path)`

Create interactive Plotly visualization.

**Parameters:**
- `clt_trace` (np.ndarray): [T, 3] CLT values
- `cli_trace` (np.ndarray): [T] CLI values
- `save_path` (str, optional): Output HTML path

**Returns:**
- `fig`: Plotly figure

### LoadGuidedDecoding

Load-Guided Decoding system with interventions.

```python
from cogload import LoadGuidedDecoding

lgd_system = LoadGuidedDecoding(
    tau_warn=0.6,
    tau_act=0.8,
    interventions=None  # Optional list of interventions
)
```

#### Parameters

- `tau_warn` (float, default: 0.6): Warning threshold for light interventions
- `tau_act` (float, default: 0.8): Action threshold for active interventions
- `interventions` (list, optional): Available interventions

#### Methods

##### `select_intervention(clt, history, active_only)`

Select most relevant intervention.

**Parameters:**
- `clt` (np.ndarray): [IL, EL, GL] values
- `history` (dict): Previous state
- `active_only` (bool): Only active interventions

**Returns:**
- `Intervention` or `None`: Selected intervention

##### `apply_interventions(clt_trace, cli_trace, model, initial_state)`

Apply interventions over full trace.

**Parameters:**
- `clt_trace` (np.ndarray): [T, 3] CLT values
- `cli_trace` (np.ndarray): [T] CLI values
- `model`: Model to apply interventions to
- `initial_state` (dict, optional): Initial state

**Returns:**
- `tuple`: (intervention_history, flags)

##### `decode_with_interventions(model, input_ids, clt_computer, ...)`

Decode with load-guided interventions.

**Parameters:**
- `model`: Language model
- `input_ids` (torch.Tensor): Input tokens
- `clt_computer`: CLT computation object
- `attention_mask` (torch.Tensor, optional): Attention mask
- `max_new_tokens` (int, default: 100): Max tokens
- `**kwargs`: Additional generation args

**Returns:**
- `tuple`: (decoded_ids, intervention_log)

## Interventions

### CacheStabilization

Stabilize KV-cache to reduce extraneous load.

```python
from cogload.interventions import CacheStabilization

intervention = CacheStabilization(warmup_steps=5, retention_ratio=0.8)
```

### DecodingControl

Control decoding via temperature and top-k.

```python
from cogload.interventions import DecodingControl

intervention = DecodingControl(temperature_scale=0.8, top_k=50)
```

### PlanningAid

Aid planning through decomposition.

```python
from cogload.interventions import PlanningAid

intervention = PlanningAid(decomposition_depth=2)
```

### ConsolidationAid

Aid consolidation via refinement.

```python
from cogload.interventions import ConsolidationAid

intervention = ConsolidationAid(refinement_steps=2)
```

### TokenMerging

Merge redundant tokens.

```python
from cogload.interventions import TokenMerging

intervention = TokenMerging(merge_threshold=0.9)
```

## Utility Functions

### Custom Intervention

Create custom interventions:

```python
from cogload.interventions import Intervention

class MyIntervention(Intervention):
    def apply(self, model, state_dict):
        # Apply intervention logic
        return state_dict
    
    def score(self, clt, history):
        # Return relevance score
        return 0.5
```

## Data Structures

### CLT Metadata

Returned by `compute_clt_trace()`:

```python
{
    'attention_entropies': List[float],
    'dispersions': List[float],
    'miss_ratios': List[float],
    'stabilities': List[float],
    'consolidations': List[float],
    'reuses': List[float]
}
```

### Intervention History

Returned by `apply_interventions()`:

```python
[
    {
        'step': int,
        'intervention': str,
        'cli': float,
        'clt': List[float],
        'type': 'active' | 'warn'
    }
]
```

