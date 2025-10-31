# CLT Theory and Formulation

This document provides a detailed explanation of the theoretical foundations of Cognitive Load Traces.

## Cognitive Load Theory

Our framework is inspired by Cognitive Load Theory (CLT) from human cognition research (Sweller, 1988). CLT distinguishes three types of cognitive load:

1. **Intrinsic Load (IL)**: The inherent difficulty of a task
2. **Extraneous Load (EL)**: Inefficiency introduced by the process
3. **Germane Load (GL)**: Effort spent building schemas

## Mathematical Formulation

### Three-Component Process

For a transformer model $\mathcal{M}$ with $L$ layers processing input sequence $x_{1:T}$, we define Cognitive Load Traces as:

$$\mathbf{CLT}_t = (IL_t, EL_t, GL_t) \in [0,1]^3$$

The **Composite Load Index** is:

$$CLI_t = \mathbf{w}^\top \mathbf{CLT}_t$$

where $\mathbf{w} = (w_I, w_E, w_G)$ are learned weights.

### Intrinsic Load (IL)

Captures task difficulty through attention dispersion and representational spread:

**Attention Entropy**:
$$H_t = \frac{1}{L}\sum_{l=1}^{L}\left(-\sum_i a_{t,i}^l \log a_{t,i}^l\right)$$

**Representation Dispersion**:
$$\text{Disp}_t = \frac{1}{L}\sum_{l=1}^{L}\frac{\|h_t^l - \bar{h}_t\|_2}{\|\bar{h}_t\|_2 + \epsilon}$$

**IL Composite**:
$$IL_t = \alpha_1 \hat{H}_t + \alpha_2 \hat{\text{Disp}}_t$$

where $\hat{x}$ denotes robust normalization.

### Extraneous Load (EL)

Reflects process inefficiency through cache misses and decoding instability:

**KV-Cache Miss Ratio**:
$$\text{Miss}_t = 1 - \frac{\text{hits}_t}{\text{queries}_t + \epsilon}$$

**Decoding Stability**:
$$\text{Stab}_t = \text{KL}(p_t \| \tilde{p}_t)$$

**EL Composite**:
$$EL_t = \beta_1 \hat{\text{Miss}}_t + \beta_2 \hat{\text{Stab}}_t$$

### Germane Load (GL)

Encodes schema-building effort through consolidation and reuse:

**Consolidation**:
$$\text{Consol}_t = \frac{1}{L-1}\sum_{l=1}^{L-1}\cos(\Delta h_t^{l+1}, \Delta h_t^l)$$

**Concept Reuse**:
$$\text{Reuse}_t = \frac{\sum_i \mathbf{1}[a_{t,i}^{\max} > \theta] \cdot \mathbf{1}[\text{concept}(i)=\text{active}]}{\sum_i \mathbf{1}[a_{t,i}^{\max} > \theta] + \epsilon}$$

**GL Composite**:
$$GL_t = \gamma_1(1-\hat{\text{Consol}}_t) + \gamma_2(1-\hat{\text{Reuse}}_t)$$

### Robust Normalization

All proxy values are normalized using median and interquartile range for comparability:

$$\hat{x}_t = \frac{x_t - \text{median}(x_{1:T})}{\text{IQR}(x_{1:T}) + \epsilon}$$

## Load-Guided Decoding (LGD)

### Intervention Selection

Given $\mathbf{CLT}_t = (IL_t, EL_t, GL_t)$, interventions are selected based on dominant load:

$$\mathcal{I}_t = \arg\max_{i \in \mathcal{I}} \text{score}_i(\mathbf{CLT}_t, \mathcal{H}_{t-1})$$

### Threshold Scheme

We maintain two thresholds:
- **Warning threshold** ($\tau_{\text{warn}}$): Light interventions
- **Action threshold** ($\tau_{\text{act}}$): Active interventions

### Algorithm

**Algorithm 1: Load-Guided Decoding (LGD)**

```
Input: Model M, input x, weights w, thresholds τ_warn, τ_act, interventions I
for t = 1 … T do
    CLT_t ← ComputeCLT(M, x, t)
    CLI_t ← w^T CLT_t
    if CLI_t > τ_act then
        Apply(I_act)
    else if CLI_t > τ_warn then
        Apply(I_warn)
    end if
end for
```

## Visualizations

### Temporal Traces

Plot $IL_t$, $EL_t$, $GL_t$, and $CLI_t$ over time to reveal cognitive dynamics.

### Simplex Diagrams

Project CLT onto barycentric coordinates for geometric interpretation:
- Vertices: Pure load types (IL/EL/GL)
- Centers: Balanced strategies

### Layer-Time Heatmaps

Show load distribution across model depth and time.

## Interpretation Guidelines

### Identifying Load Types

- **High IL**: Complex, multi-faceted tasks requiring careful planning
- **High EL**: Computational inefficiency, cache thrashing, unstable decoding
- **High GL**: Active schema construction, concept formation

### Phase Identification (Summarization)

- **Planning**: High GL, low EL (outline generation)
- **Search**: High EL, low GL (content retrieval)
- **Consolidation**: Balanced loads (final synthesis)

### Error Prediction

Empirically, 73% of reasoning errors coincide with EL spikes exceeding 0.8.

## References

- Sweller, J. (1988). Cognitive load during problem solving: Effects on learning. *Cognitive Science*, 12(2), 257-285.
- Paas, F., & van Merriënboer, J. J. (1993). The efficiency of instructional conditions: An approach to combine mental effort and performance measures. *Human Factors*, 35(4), 737-743.
- Chandler, P., & Sweller, J. (1991). Cognitive load theory and the format of instruction. *Cognition and Instruction*, 8(4), 293-332.

