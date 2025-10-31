"""
Core CLT computation module implementing the Cognitive Load Traces framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import math


class CognitiveLoadTraces:
    """
    Cognitive Load Traces (CLT) for transformer model interpretability.
    
    Implements the three-component stochastic process:
    CLT_t = (IL_t, EL_t, GL_t) representing Intrinsic, Extraneous, and Germane load.
    """
    
    def __init__(
        self,
        alpha_il: Tuple[float, float] = (0.5, 0.5),
        beta_el: Tuple[float, float] = (0.5, 0.5),
        gamma_gl: Tuple[float, float] = (0.5, 0.5),
        w_cli: Tuple[float, float, float] = (0.4, 0.4, 0.2),
        theta_attention: float = 0.1,
        normalize: bool = True
    ):
        """
        Initialize CLT framework.
        
        Args:
            alpha_il: Weights for attention entropy and dispersion in IL
            beta_el: Weights for miss ratio and stability in EL
            gamma_gl: Weights for consolidation and reuse in GL
            w_cli: Weights for IL, EL, GL in composite load index
            theta_attention: Threshold for attention-based concept detection
            normalize: Whether to apply robust normalization
        """
        self.alpha_il = alpha_il
        self.beta_el = beta_el
        self.gamma_gl = gamma_gl
        self.w_cli = w_cli
        self.theta_attention = theta_attention
        self.normalize = normalize
        
    def compute_intrinsic_load(
        self,
        attention_weights: torch.Tensor,
        hidden_states: torch.Tensor,
        step: int,
        sequence_length: int
    ) -> float:
        """
        Compute Intrinsic Load (IL) at step t.
        
        Captures task difficulty via attention dispersion and representational spread.
        
        Args:
            attention_weights: [num_layers, num_heads, seq_len] attention weights
            hidden_states: [num_layers, seq_len, hidden_dim] hidden states
            step: Current decoding step
            sequence_length: Total sequence length for normalization
            
        Returns:
            IL_t value
        """
        num_layers, num_heads, seq_len = attention_weights.shape
        
        # Attention entropy across layers
        entropies = []
        for l in range(num_layers):
            layer_entropy = 0.0
            for h in range(num_heads):
                # Get attention distribution for this head
                attn_dist = attention_weights[l, h, :step+1]
                attn_dist = attn_dist + 1e-10  # Avoid log(0)
                attn_dist = attn_dist / attn_dist.sum()
                
                # Compute entropy
                entropy = -(attn_dist * torch.log(attn_dist)).sum().item()
                layer_entropy += entropy
            entropies.append(layer_entropy / num_heads)
        
        H_t = np.mean(entropies)
        
        # Representation dispersion across layers
        dispersions = []
        for l in range(num_layers):
            # Get hidden states for this layer
            h_l = hidden_states[l, :step+1]  # [seq_len, hidden_dim]
            
            # Compute mean representation
            h_mean = h_l.mean(dim=0)  # [hidden_dim]
            
            # Compute dispersion
            norm_mean = torch.norm(h_mean) + 1e-6
            dispersion = torch.norm(h_l - h_mean) / (norm_mean + 1e-6)
            dispersions.append(dispersion.item())
        
        Disp_t = np.mean(dispersions)
        
        # Combine with weights
        IL_t = self.alpha_il[0] * H_t + self.alpha_il[1] * Disp_t
        
        return IL_t
    
    def compute_extraneous_load(
        self,
        kv_cache_misses: int,
        total_queries: int,
        current_probs: torch.Tensor,
        previous_probs: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute Extraneous Load (EL) at step t.
        
        Reflects process inefficiency via KV-cache miss ratio and decoding stability.
        
        Args:
            kv_cache_misses: Number of cache misses at this step
            total_queries: Total number of queries
            current_probs: [vocab_size] current decoding probabilities
            previous_probs: [vocab_size] previous decoding probabilities (optional)
            
        Returns:
            EL_t value
        """
        # KV-cache miss ratio
        Miss_t = 1.0 - (total_queries - kv_cache_misses) / (total_queries + 1e-10)
        
        # Decoding stability (KL divergence)
        if previous_probs is not None:
            # Compute KL divergence between consecutive distributions
            curr_probs = current_probs + 1e-10
            prev_probs = previous_probs + 1e-10
            curr_probs = curr_probs / curr_probs.sum()
            prev_probs = prev_probs / prev_probs.sum()
            
            Stab_t = F.kl_div(
                F.log_softmax(prev_probs, dim=-1),
                curr_probs,
                reduction='sum'
            ).item()
        else:
            Stab_t = 0.0
        
        # Combine with weights
        EL_t = self.beta_el[0] * Miss_t + self.beta_el[1] * Stab_t
        
        return EL_t
    
    def compute_germane_load(
        self,
        hidden_states: torch.Tensor,
        attention_weights: torch.Tensor,
        step: int,
        concept_map: Optional[Dict[int, str]] = None
    ) -> float:
        """
        Compute Germane Load (GL) at step t.
        
        Encodes schema-building effort via consolidation and concept reuse.
        
        Args:
            hidden_states: [num_layers, seq_len, hidden_dim] hidden states
            attention_weights: [num_layers, num_heads, seq_len] attention weights
            step: Current decoding step
            concept_map: Optional mapping from position to concept identity
            
        Returns:
            GL_t value
        """
        num_layers, seq_len, hidden_dim = hidden_states.shape
        
        # Consolidation: cosine similarity of consecutive layer gradients
        consolidations = []
        for l in range(num_layers - 1):
            # Compute representation changes
            h_curr = hidden_states[l, :step+1]
            h_next = hidden_states[l+1, :step+1]
            
            if step > 0:
                delta_h_l = h_next[-1] - h_curr[-1]
                delta_h_l_prev = h_next[-2] - h_curr[-2] if step > 0 else delta_h_l
                
                # Cosine similarity
                cos_sim = F.cosine_similarity(
                    delta_h_l.unsqueeze(0),
                    delta_h_l_prev.unsqueeze(0)
                ).item()
                consolidations.append(abs(cos_sim))
        
        Consol_t = np.mean(consolidations) if consolidations else 0.5
        
        # Concept reuse: fraction of high-attention tokens that are active concepts
        if concept_map is not None:
            reuses = []
            for l in range(num_layers):
                # Get attention weights for this layer
                attn_weights = attention_weights[l, :, :step+1]  # [num_heads, seq_len]
                attn_max = attn_weights.max(dim=0)[0]  # [seq_len]
                
                # Count high-attention positions
                high_attn_mask = attn_max > self.theta_attention
                num_high_attn = high_attn_mask.sum().item()
                
                if num_high_attn > 0:
                    # Count active concepts in high-attention positions
                    active_concepts = 0
                    for i in range(step + 1):
                        if high_attn_mask[i] and i in concept_map:
                            if concept_map[i] != 'inactive':
                                active_concepts += 1
                    
                    reuse_ratio = active_concepts / (num_high_attn + 1e-10)
                    reuses.append(reuse_ratio)
            
            Reuse_t = np.mean(reuses) if reuses else 0.0
        else:
            Reuse_t = 0.0
        
        # Combine with weights (inverse consolidation and reuse indicate higher load)
        GL_t = self.gamma_gl[0] * (1.0 - Consol_t) + self.gamma_gl[1] * (1.0 - Reuse_t)
        
        return GL_t
    
    def compute_clt_trace(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[Dict] = None,
        concept_map: Optional[Dict[int, str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Compute complete CLT trace over sequence.
        
        Args:
            model: Transformer model
            input_ids: [batch_size, seq_len] input token ids
            attention_mask: [batch_size, seq_len] attention mask
            kv_cache: Optional KV cache for tracking misses
            concept_map: Optional position-to-concept mapping
            
        Returns:
            clt_trace: [T, 3] array of (IL_t, EL_t, GL_t) for each step
            cli_trace: [T] array of composite load indices
            metadata: Dictionary with raw measurements
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Store traces
        il_trace = []
        el_trace = []
        gl_trace = []
        cli_trace = []
        metadata = {
            'attention_entropies': [],
            'dispersions': [],
            'miss_ratios': [],
            'stabilities': [],
            'consolidations': [],
            'reuses': []
        }
        
        # Initialize decoder probabilities
        prev_probs = None
        
        # Forward pass to get hidden states and attention
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True
            )
            
            hidden_states = torch.stack(outputs.hidden_states[1:])  # Skip embedding layer
            attentions = outputs.attentions
            
            num_layers = hidden_states.shape[0]
            
            # Reshape attention from list of tensors to [num_layers, num_heads, seq_len, seq_len]
            if attentions:
                all_attentions = []
                for layer in attentions:
                    # layer: [batch_size, num_heads, seq_len, seq_len]
                    all_attentions.append(layer[0])  # Take first batch item
                attention_weights = torch.stack(all_attentions)  # [num_layers, num_heads, seq_len, seq_len]
            else:
                # Fallback: uniform attention if not available
                num_heads = 8
                attention_weights = torch.ones(num_layers, num_heads, seq_len, seq_len) / seq_len
            
            # Compute CLT for each decoding step
            for t in range(seq_len):
                # Get attention for step t
                attn_t = attention_weights[:, :, t, :t+1].squeeze(dim=-1)  # [num_layers, num_heads, t+1]
                if attn_t.dim() == 2:
                    attn_t = attn_t.unsqueeze(-1)
                
                # Compute IL
                il_t = self.compute_intrinsic_load(
                    attn_t,
                    hidden_states[:, :, :],
                    t,
                    seq_len
                )
                il_trace.append(il_t)
                
                # Compute EL (simplified version without actual cache tracking)
                kv_misses = 0  # Would track actual misses in production
                total_queries = (t + 1) * num_layers
                
                if t < seq_len - 1 and hasattr(outputs, 'logits'):
                    current_probs = F.softmax(outputs.logits[0, t], dim=-1)
                    el_t = self.compute_extraneous_load(
                        kv_misses, total_queries,
                        current_probs,
                        prev_probs
                    )
                    prev_probs = current_probs
                else:
                    el_t = 0.0
                
                el_trace.append(el_t)
                
                # Compute GL - use full attention sequence for this layer
                # Get average attention across sequence dimension
                attn_for_gl = attention_weights[:, :, :t+1].mean(dim=-2)  # [num_layers, num_heads, t+1]
                # Reduce to [num_layers, num_heads] by taking mean
                attn_for_gl = attn_for_gl.mean(dim=-1)  # [num_layers, num_heads]
                gl_t = self.compute_germane_load(
                    hidden_states[:, :, :],
                    attn_for_gl,
                    t,
                    concept_map
                )
                gl_trace.append(gl_t)
                
                # Store metadata
                metadata['attention_entropies'].append(il_t)
                metadata['miss_ratios'].append(el_t)
                
        # Normalize traces if requested
        if self.normalize:
            il_trace = self._normalize_trace(il_trace)
            el_trace = self._normalize_trace(el_trace)
            gl_trace = self._normalize_trace(gl_trace)
        
        # Stack into CLT trace
        clt_trace = np.stack([
            np.array(il_trace),
            np.array(el_trace),
            np.array(gl_trace)
        ], axis=1)
        
        # Compute composite load index
        cli_trace = np.array([
            self.w_cli[0] * il_t + self.w_cli[1] * el_t + self.w_cli[2] * gl_t
            for il_t, el_t, gl_t in zip(il_trace, el_trace, gl_trace)
        ])
        
        return clt_trace, cli_trace, metadata
    
    def _normalize_trace(self, trace: List[float]) -> np.ndarray:
        """
        Apply robust normalization using median and IQR.
        
        x_hat = (x - median) / (IQR + epsilon)
        
        Args:
            trace: List of values to normalize
            
        Returns:
            Normalized array
        """
        trace_array = np.array(trace)
        median = np.median(trace_array)
        iqr = stats.iqr(trace_array)
        epsilon = 1e-10
        
        normalized = (trace_array - median) / (iqr + epsilon)
        
        return normalized
    
    def identify_load_type(self, il: float, el: float, gl: float) -> str:
        """
        Identify dominant load type from CLT components.
        
        Args:
            il: Intrinsic load value
            el: Extraneous load value
            gl: Germane load value
            
        Returns:
            String describing dominant load type
        """
        loads = {'IL': il, 'EL': el, 'GL': gl}
        dominant = max(loads.items(), key=lambda x: x[1])
        
        return dominant[0]

