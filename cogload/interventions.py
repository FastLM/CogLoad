"""
Load-Guided Decoding (LGD) interventions based on CLT signals.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from enum import Enum


class InterventionType(Enum):
    """Types of cognitive load interventions."""
    PLANNING_AID = "planning_aid"      # High IL
    EFFICIENCY_AID = "efficiency_aid"  # High EL
    CONSOLIDATION_AID = "consolidation_aid"  # High GL
    NONE = "none"


class Intervention(ABC):
    """Base class for load-guided interventions."""
    
    @abstractmethod
    def apply(self, model, state_dict):
        """
        Apply intervention to model or decoding process.
        
        Args:
            model: The model to apply intervention to
            state_dict: Current state including CLT values, cache, etc.
            
        Returns:
            Updated state dictionary
        """
        pass
    
    @abstractmethod
    def score(self, clt: np.ndarray, history: Dict) -> float:
        """
        Score how relevant this intervention is for current load state.
        
        Args:
            clt: [IL, EL, GL] current load values
            history: Previous state history
            
        Returns:
            Relevance score (higher = more relevant)
        """
        pass


class CacheStabilization(Intervention):
    """
    Stabilize KV-cache to reduce extraneous load spikes.
    Implements proactive cache warming and retention strategies.
    """
    
    def __init__(self, warmup_steps: int = 5, retention_ratio: float = 0.8):
        self.warmup_steps = warmup_steps
        self.retention_ratio = retention_ratio
    
    def apply(self, model, state_dict):
        """Apply cache stabilization."""
        if 'kv_cache' in state_dict:
            kv_cache = state_dict['kv_cache']
            # Implement cache warming logic
            # Increase retention rate for recent keys
            # This is a simplified version
            
            # Mark cache entries as "protected" based on recent access
            if 'access_history' not in state_dict:
                state_dict['access_history'] = {}
            
            state_dict['kv_cache'] = kv_cache
            state_dict['cache_stabilized'] = True
        
        return state_dict
    
    def score(self, clt: np.ndarray, history: Dict) -> float:
        """Score based on extraneous load."""
        return clt[1]  # EL component


class DecodingControl(Intervention):
    """
    Control decoding process to reduce instability.
    Implements temperature scaling and top-k filtering.
    """
    
    def __init__(self, temperature_scale: float = 0.8, top_k: int = 50):
        self.temperature_scale = temperature_scale
        self.top_k = top_k
    
    def apply(self, model, state_dict):
        """Apply decoding control."""
        state_dict['temperature'] = state_dict.get('temperature', 1.0) * self.temperature_scale
        state_dict['top_k'] = self.top_k
        state_dict['decoding_controlled'] = True
        return state_dict
    
    def score(self, clt: np.ndarray, history: Dict) -> float:
        """Score based on extraneous load."""
        return clt[1]  # EL component


class PlanningAid(Intervention):
    """
    Aid planning by providing intermediate subgoals.
    Reduces intrinsic load through decomposition.
    """
    
    def __init__(self, decomposition_depth: int = 2):
        self.decomposition_depth = decomposition_depth
    
    def apply(self, model, state_dict):
        """Apply planning aid."""
        state_dict['planning_aid'] = True
        state_dict['decomposition_depth'] = self.decomposition_depth
        
        # Inject subgoal prompts if available
        if 'subgoals' in state_dict:
            state_dict['use_subgoals'] = True
        
        return state_dict
    
    def score(self, clt: np.ndarray, history: Dict) -> float:
        """Score based on intrinsic load."""
        return clt[0]  # IL component


class ConsolidationAid(Intervention):
    """
    Aid consolidation through representation refinement.
    Reduces germane load by strengthening connections.
    """
    
    def __init__(self, refinement_steps: int = 2):
        self.refinement_steps = refinement_steps
    
    def apply(self, model, state_dict):
        """Apply consolidation aid."""
        state_dict['consolidation_aid'] = True
        state_dict['refinement_steps'] = self.refinement_steps
        return state_dict
    
    def score(self, clt: np.ndarray, history: Dict) -> float:
        """Score based on germane load."""
        return clt[2]  # GL component


class TokenMerging(Intervention):
    """
    Merge redundant tokens to reduce sequence length.
    Lowers computational overhead.
    """
    
    def __init__(self, merge_threshold: float = 0.9):
        self.merge_threshold = merge_threshold
    
    def apply(self, model, state_dict):
        """Apply token merging."""
        state_dict['token_merging'] = True
        state_dict['merge_threshold'] = self.merge_threshold
        return state_dict
    
    def score(self, clt: np.ndarray, history: Dict) -> float:
        """Score based on extraneous load."""
        return clt[1] * 0.7  # EL component with weight


class LoadGuidedDecoding:
    """
    Main Load-Guided Decoding (LGD) system.
    
    Implements Algorithm 1 from the paper:
    - Computes CLT at each step
    - Monitors CLI against thresholds
    - Applies interventions based on load type
    """
    
    def __init__(
        self,
        tau_warn: float = 0.6,
        tau_act: float = 0.8,
        interventions: Optional[List[Intervention]] = None
    ):
        """
        Initialize LGD system.
        
        Args:
            tau_warn: Warning threshold for CLI
            tau_act: Action threshold for CLI
            interventions: List of available interventions
        """
        self.tau_warn = tau_warn
        self.tau_act = tau_act
        
        if interventions is None:
            # Default interventions
            self.interventions = [
                CacheStabilization(),
                DecodingControl(),
                PlanningAid(),
                ConsolidationAid(),
                TokenMerging()
            ]
        else:
            self.interventions = interventions
        
        # Track intervention history
        self.intervention_history = []
    
    def select_intervention(
        self,
        clt: np.ndarray,
        history: Dict,
        active_only: bool = False
    ) -> Optional[Intervention]:
        """
        Select most relevant intervention based on current CLT.
        
        Args:
            clt: [IL, EL, GL] current load values
            history: Previous state history
            active_only: If True, only apply when CLI > tau_act
            
        Returns:
            Selected intervention or None
        """
        # Compute composite load index
        cli = np.dot([0.4, 0.4, 0.2], clt)
        
        if active_only and cli < self.tau_act:
            return None
        elif not active_only and cli < self.tau_warn:
            return None
        
        # Score all interventions
        scores = [interv.score(clt, history) for interv in self.interventions]
        
        # Select intervention with highest score
        max_idx = np.argmax(scores)
        selected_intervention = self.interventions[max_idx]
        
        return selected_intervention
    
    def apply_interventions(
        self,
        clt_trace: np.ndarray,
        cli_trace: np.ndarray,
        model: nn.Module,
        initial_state: Optional[Dict] = None
    ) -> Tuple[List[Dict], List[bool]]:
        """
        Apply load-guided interventions over full trace.
        
        Args:
            clt_trace: [T, 3] array of CLT values
            cli_trace: [T] array of CLI values
            model: The model to apply interventions to
            initial_state: Initial state dictionary
            
        Returns:
            Tuple of (intervention_history, intervention_flags)
        """
        if initial_state is None:
            initial_state = {}
        
        intervention_history = []
        intervention_flags = []
        
        for t in range(len(clt_trace)):
            current_clt = clt_trace[t]
            cli_t = cli_trace[t]
            
            # Determine intervention level
            active_intervention = None
            if cli_t > self.tau_act:
                # Active intervention
                active_intervention = self.select_intervention(
                    current_clt,
                    initial_state,
                    active_only=True
                )
            elif cli_t > self.tau_warn:
                # Warning intervention (lighter)
                active_intervention = self.select_intervention(
                    current_clt,
                    initial_state,
                    active_only=False
                )
            
            # Apply intervention
            applied = False
            if active_intervention is not None:
                initial_state = active_intervention.apply(model, initial_state)
                applied = True
                
                intervention_history.append({
                    'step': t,
                    'intervention': active_intervention.__class__.__name__,
                    'cli': float(cli_t),
                    'clt': current_clt.tolist(),
                    'type': 'active' if cli_t > self.tau_act else 'warn'
                })
            
            intervention_flags.append(applied)
        
        return intervention_history, intervention_flags
    
    def decode_with_interventions(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        clt_computer,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **kwargs
    ):
        """
        Decode sequence with load-guided interventions.
        
        Args:
            model: The language model
            input_ids: Initial input token ids
            clt_computer: CLT computation object
            attention_mask: Optional attention mask
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for generation
            
        Returns:
            Decoded sequence and intervention log
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Initialize state
        state_dict = {
            'kv_cache': None,
            'temperature': kwargs.get('temperature', 1.0),
            'top_k': kwargs.get('top_k', None)
        }
        
        # Generate with interventions
        generated_ids = input_ids.clone()
        all_clt_trace = []
        all_cli_trace = []
        intervention_log = []
        
        for step in range(max_new_tokens):
            # Compute CLT for current state
            # This is simplified - in practice you'd update at each generation step
            if step == 0:
                clt_trace, cli_trace, _ = clt_computer.compute_clt_trace(
                    model, generated_ids, 
                    attention_mask=attention_mask if attention_mask is not None 
                    else torch.ones_like(generated_ids)
                )
                all_clt_trace = clt_trace
                all_cli_trace = cli_trace
            
            # Select and apply intervention
            current_clt = all_clt_trace[min(step, len(all_clt_trace)-1)]
            intervention = self.select_intervention(
                current_clt,
                state_dict,
                active_only=(all_cli_trace[min(step, len(all_cli_trace)-1)] > self.tau_act)
            )
            
            if intervention is not None:
                state_dict = intervention.apply(model, state_dict)
                intervention_log.append({
                    'step': step,
                    'intervention': intervention.__class__.__name__,
                    'clt': current_clt.tolist()
                })
            
            # Generate next token (simplified)
            # In practice, use model.generate with interventions applied
            with torch.no_grad():
                outputs = model(
                    generated_ids,
                    attention_mask=attention_mask,
                    **state_dict,
                    **kwargs
                )
                logits = outputs.logits[:, -1, :]
                
                # Apply decoding control if set
                if state_dict.get('decoding_controlled', False):
                    temperature = state_dict.get('temperature', 1.0)
                    top_k = state_dict.get('top_k', None)
                    
                    logits = logits / temperature
                    
                    if top_k is not None:
                        top_k_logits, top_k_indices = torch.topk(logits, top_k)
                        logits = torch.full_like(logits, float('-inf'))
                        logits.scatter_(1, top_k_indices, top_k_logits)
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones(batch_size, 1, device=device)
                    ], dim=1)
        
        return generated_ids, intervention_log
    
    def evaluate_interventions(
        self,
        baseline_results: Dict,
        intervention_results: Dict,
        metric_key: str = 'accuracy'
    ) -> Dict:
        """
        Evaluate effectiveness of interventions.
        
        Args:
            baseline_results: Baseline model results
            intervention_results: Results with interventions
            metric_key: Key for performance metric
            
        Returns:
            Dictionary with improvement statistics
        """
        baseline_score = baseline_results.get(metric_key, 0.0)
        intervention_score = intervention_results.get(metric_key, 0.0)
        
        improvement = intervention_score - baseline_score
        improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0
        
        return {
            'baseline': baseline_score,
            'intervention': intervention_score,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'num_interventions': len(self.intervention_history)
        }

