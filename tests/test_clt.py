"""
Unit tests for CLT framework.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from cogload import CognitiveLoadTraces, CLTVisualizer, LoadGuidedDecoding
from cogload.interventions import CacheStabilization, DecodingControl, PlanningAid


class MockTransformer(nn.Module):
    """Mock transformer for testing."""
    
    def __init__(self, vocab_size=1000, num_layers=6, hidden_dim=128, num_heads=8):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        
        # Create encoder layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Embedding and output layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass with mock attention."""
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        x = self.embedding(input_ids)
        
        # Store attention and hidden states
        hidden_states = [x]
        attentions = []
        
        # Pass through layers
        for layer in self.layers:
            # Create mock attention weights
            attn_weights = self._create_mock_attention(batch_size, seq_len)
            attentions.append(attn_weights)
            
            x = layer(x, src_key_padding_mask=attention_mask)
            hidden_states.append(x)
        
        # Stack hidden states (skip embedding layer)
        all_hidden_states = torch.stack(hidden_states[1:])
        
        # Stack attention weights
        all_attentions = torch.stack(attentions)
        
        # Output logits
        logits = self.output_proj(x)
        
        # Create output structure
        class Output:
            def __init__(self):
                self.logits = logits
                self.hidden_states = (x,) + tuple(hidden_states[1:])
                self.attentions = tuple(all_attentions)
        
        return Output()
    
    def _create_mock_attention(self, batch_size, seq_len):
        """Create mock attention weights."""
        # Normalized random attention
        attn = torch.rand(batch_size, self.num_heads, seq_len, seq_len)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        return attn


class TestCognitiveLoadTraces(unittest.TestCase):
    """Test CLT computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.clt_computer = CognitiveLoadTraces(
            alpha_il=(0.5, 0.5),
            beta_el=(0.5, 0.5),
            gamma_gl=(0.5, 0.5),
            w_cli=(0.4, 0.4, 0.2)
        )
        self.model = MockTransformer()
        self.input_ids = torch.randint(0, 1000, (1, 20))
        self.attention_mask = torch.ones((1, 20), dtype=torch.bool)
    
    def test_clt_initialization(self):
        """Test CLT initialization."""
        self.assertIsNotNone(self.clt_computer)
        self.assertEqual(self.clt_computer.w_cli, (0.4, 0.4, 0.2))
    
    def test_clt_trace_computation(self):
        """Test CLT trace computation."""
        clt_trace, cli_trace, metadata = self.clt_computer.compute_clt_trace(
            self.model,
            self.input_ids,
            attention_mask=self.attention_mask
        )
        
        # Check shapes
        self.assertEqual(clt_trace.shape[1], 3)  # IL, EL, GL
        self.assertEqual(clt_trace.shape[0], self.input_ids.shape[1])
        self.assertEqual(len(cli_trace), self.input_ids.shape[1])
        
        # Check values are reasonable
        self.assertTrue(np.all(clt_trace >= -10))  # Allow some negative after normalization
        self.assertTrue(np.all(clt_trace <= 10))
    
    def test_normalization(self):
        """Test robust normalization."""
        trace = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = self.clt_computer._normalize_trace(trace)
        
        self.assertEqual(len(normalized), len(trace))
        self.assertIsInstance(normalized, np.ndarray)
    
    def test_intrinsic_load_computation(self):
        """Test intrinsic load computation."""
        num_layers = 4
        num_heads = 8
        seq_len = 10
        
        # Create mock attention and hidden states
        attention_weights = torch.ones(num_layers, num_heads, seq_len) / seq_len
        hidden_states = torch.randn(num_layers, seq_len, 64)
        
        il = self.clt_computer.compute_intrinsic_load(
            attention_weights,
            hidden_states,
            step=5,
            sequence_length=seq_len
        )
        
        self.assertIsInstance(il, float)
        self.assertGreater(il, 0)
    
    def test_extraneous_load_computation(self):
        """Test extraneous load computation."""
        kv_misses = 10
        total_queries = 100
        current_probs = torch.softmax(torch.randn(1000), dim=0)
        previous_probs = torch.softmax(torch.randn(1000), dim=0)
        
        el = self.clt_computer.compute_extraneous_load(
            kv_misses, total_queries,
            current_probs, previous_probs
        )
        
        self.assertIsInstance(el, float)
        self.assertGreater(el, 0)
    
    def test_germane_load_computation(self):
        """Test germane load computation."""
        num_layers = 4
        seq_len = 10
        hidden_dim = 64
        
        hidden_states = torch.randn(num_layers, seq_len, hidden_dim)
        attention_weights = torch.ones(num_layers, 8, seq_len) / seq_len
        
        gl = self.clt_computer.compute_germane_load(
            hidden_states,
            attention_weights,
            step=5,
            concept_map=None
        )
        
        self.assertIsInstance(gl, float)
        self.assertGreater(gl, 0)
    
    def test_load_type_identification(self):
        """Test load type identification."""
        # High IL
        load_type = self.clt_computer.identify_load_type(0.9, 0.2, 0.3)
        self.assertEqual(load_type, 'IL')
        
        # High EL
        load_type = self.clt_computer.identify_load_type(0.2, 0.9, 0.3)
        self.assertEqual(load_type, 'EL')
        
        # High GL
        load_type = self.clt_computer.identify_load_type(0.2, 0.3, 0.9)
        self.assertEqual(load_type, 'GL')


class TestInterventions(unittest.TestCase):
    """Test load-guided interventions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.clt_computer = CognitiveLoadTraces()
        self.lgd_system = LoadGuidedDecoding(
            tau_warn=0.6,
            tau_act=0.8
        )
        self.model = MockTransformer()
    
    def test_intervention_initialization(self):
        """Test intervention system initialization."""
        self.assertIsNotNone(self.lgd_system)
        self.assertEqual(len(self.lgd_system.interventions), 5)  # Default interventions
    
    def test_cache_stabilization(self):
        """Test cache stabilization intervention."""
        intervention = CacheStabilization()
        
        # High EL should score high
        score = intervention.score(np.array([0.2, 0.9, 0.3]), {})
        self.assertGreater(score, 0.8)
        
        # Low EL should score low
        score = intervention.score(np.array([0.2, 0.3, 0.3]), {})
        self.assertLess(score, 0.4)
    
    def test_decoding_control(self):
        """Test decoding control intervention."""
        intervention = DecodingControl()
        
        # High EL should score high
        score = intervention.score(np.array([0.2, 0.9, 0.3]), {})
        self.assertGreater(score, 0.8)
    
    def test_planning_aid(self):
        """Test planning aid intervention."""
        intervention = PlanningAid()
        
        # High IL should score high
        score = intervention.score(np.array([0.9, 0.2, 0.3]), {})
        self.assertGreater(score, 0.8)
    
    def test_intervention_selection(self):
        """Test intervention selection logic."""
        # High EL scenario
        clt = np.array([0.2, 0.9, 0.3])
        intervention = self.lgd_system.select_intervention(clt, {}, active_only=True)
        self.assertIsNotNone(intervention)
        
        # Low CLI scenario
        clt = np.array([0.1, 0.1, 0.1])
        intervention = self.lgd_system.select_intervention(clt, {}, active_only=True)
        self.assertIsNone(intervention)


class TestVisualization(unittest.TestCase):
    """Test visualization tools."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = CLTVisualizer()
        self.clt_trace = np.random.rand(50, 3)
        self.cli_trace = np.random.rand(50)
        self.error_events = [10, 25, 40]
    
    def test_temporal_traces_plot(self):
        """Test temporal traces plotting."""
        fig, ax = self.visualizer.plot_temporal_traces(
            self.clt_trace,
            self.cli_trace,
            error_events=self.error_events
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
    
    def test_simplex_plot(self):
        """Test simplex diagram plotting."""
        fig, ax = self.visualizer.plot_simplex(self.clt_trace)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
    
    def test_correlation_analysis(self):
        """Test correlation analysis plotting."""
        fig, axes = self.visualizer.plot_correlation_analysis(
            self.clt_trace,
            self.cli_trace,
            self.error_events
        )
        self.assertIsNotNone(fig)
        self.assertEqual(len(axes), 3)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test full CLT pipeline."""
        clt_computer = CognitiveLoadTraces()
        lgd_system = LoadGuidedDecoding()
        model = MockTransformer()
        visualizer = CLTVisualizer()
        
        # Create input
        input_ids = torch.randint(0, 1000, (1, 20))
        attention_mask = torch.ones((1, 20), dtype=torch.bool)
        
        # Compute CLT
        clt_trace, cli_trace, _ = clt_computer.compute_clt_trace(
            model, input_ids, attention_mask=attention_mask
        )
        
        # Apply interventions
        intervention_history, _ = lgd_system.apply_interventions(
            clt_trace, cli_trace, model
        )
        
        # Visualize
        visualizer.plot_temporal_traces(clt_trace, cli_trace)
        
        # Check results
        self.assertGreater(len(clt_trace), 0)
        self.assertGreater(len(intervention_history), 0)


if __name__ == '__main__':
    unittest.main()

