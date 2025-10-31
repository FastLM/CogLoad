"""
Example usage of CLT framework on XSum summarization tasks.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cogload import CognitiveLoadTraces, CLTVisualizer, LoadGuidedDecoding
import numpy as np
from typing import List, Dict


def setup_model(model_name: str = "facebook/opt-1.3b"):
    """Load model and tokenizer for summarization."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded successfully")
    return model, tokenizer


def generate_example_article() -> str:
    """Generate a sample XSum article."""
    article = """The UK's education secretary has announced plans to increase funding
    for primary schools by £1.3bn next year. The money will be used to reduce class sizes
    and improve infrastructure. Critics say the funding is insufficient given inflation rates.
    Teachers' unions have welcomed the announcement but called for more investment in
    teacher training and support. The government has also pledged to review the national
    curriculum and assessment frameworks. The changes come amid growing concerns about
    educational inequality and the impact of the pandemic on children's learning."""
    return article


def compute_clt_for_summarization(model, tokenizer, article: str, clt_computer):
    """Compute CLT trace for summarization task."""
    # Prepare input with summarization prompt
    prompt = f"Summarize the following article in one sentence: {article}\n\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    print("Computing CLT trace for summarization...")
    clt_trace, cli_trace, metadata = clt_computer.compute_clt_trace(
        model,
        inputs.input_ids,
        attention_mask=inputs.attention_mask
    )
    
    return clt_trace, cli_trace, metadata


def identify_planning_search_consolidation_phases(clt_trace: np.ndarray) -> Dict[str, List[int]]:
    """
    Identify distinct cognitive phases in summarization:
    - Planning (high GL, low EL)
    - Search (high EL, low GL)
    - Consolidation (balanced loads)
    """
    phases = {
        'planning': [],
        'search': [],
        'consolidation': []
    }
    
    for i in range(len(clt_trace)):
        il, el, gl = clt_trace[i]
        
        # Planning: high GL, low EL
        if gl > 0.7 and el < 0.4:
            phases['planning'].append(i)
        # Search: high EL, low GL
        elif el > 0.7 and gl < 0.4:
            phases['search'].append(i)
        # Consolidation: balanced
        elif abs(il - 0.33) < 0.2 and abs(el - 0.33) < 0.2 and abs(gl - 0.33) < 0.2:
            phases['consolidation'].append(i)
    
    return phases


def run_baseline_summarization(model, tokenizer, article: str, max_tokens: int = 100):
    """Run baseline summarization without interventions."""
    prompt = f"Summarize the following article in one sentence: {article}\n\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the summary part
    summary = response.split("Summary:")[-1].strip()
    return summary


def run_load_guided_summarization(
    model, 
    tokenizer, 
    article: str, 
    clt_computer,
    lgd_system,
    max_tokens: int = 100
):
    """Run load-guided summarization with interventions."""
    prompt = f"Summarize the following article in one sentence: {article}\n\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    # Compute CLT first
    clt_trace, cli_trace, _ = clt_computer.compute_clt_trace(
        model,
        inputs.input_ids,
        attention_mask=inputs.attention_mask
    )
    
    # Apply interventions
    intervention_history, _ = lgd_system.apply_interventions(
        clt_trace,
        cli_trace,
        model
    )
    
    # Generate with interventions
    temperature = 0.7
    top_k = 50
    
    # Apply intervention parameters
    for intervention in intervention_history:
        if 'Control' in intervention['intervention']:
            temperature = temperature * 0.8
            top_k = 50
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = response.split("Summary:")[-1].strip()
    
    return summary, intervention_history


def analyze_phase_transitions(phases: Dict[str, List[int]], clt_trace: np.ndarray):
    """Analyze transitions between cognitive phases."""
    print("\n" + "=" * 80)
    print("Phase Analysis:")
    print("=" * 80)
    
    print(f"Planning phases: {len(phases['planning'])} steps")
    print(f"Search phases: {len(phases['search'])} steps")
    print(f"Consolidation phases: {len(phases['consolidation'])} steps")
    
    # Compute average loads for each phase
    for phase_name, phase_steps in phases.items():
        if phase_steps:
            phase_clts = clt_trace[phase_steps]
            avg_il = phase_clts[:, 0].mean()
            avg_el = phase_clts[:, 1].mean()
            avg_gl = phase_clts[:, 2].mean()
            
            print(f"\n{phase_name.capitalize()} phase averages:")
            print(f"  IL: {avg_il:.3f}, EL: {avg_el:.3f}, GL: {avg_gl:.3f}")


def main():
    """Main example workflow for XSum."""
    print("=" * 80)
    print("Cognitive Load Traces (CLT) - XSum Summarization Example")
    print("=" * 80)
    
    # Initialize CLT framework
    clt_computer = CognitiveLoadTraces(
        alpha_il=(0.5, 0.5),
        beta_el=(0.5, 0.5),
        gamma_gl=(0.5, 0.5),
        w_cli=(0.4, 0.4, 0.2)
    )
    
    # Initialize LGD system
    lgd_system = LoadGuidedDecoding(
        tau_warn=0.6,
        tau_act=0.8
    )
    
    # Load model
    print("\nNote: Using a small model for demonstration.")
    
    try:
        model, tokenizer = setup_model("microsoft/phi-2")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please install a compatible model or adjust model_name")
        return
    
    # Generate example article
    article = generate_example_article()
    print(f"\nArticle: {article[:100]}...")
    
    # Compute CLT trace
    print("\nComputing CLT trace...")
    try:
        clt_trace, cli_trace, metadata = compute_clt_for_summarization(
            model, tokenizer, article, clt_computer
        )
        print(f"CLT trace shape: {clt_trace.shape}")
        print(f"CLI trace shape: {cli_trace.shape}")
        
        # Identify cognitive phases
        phases = identify_planning_search_consolidation_phases(clt_trace)
        analyze_phase_transitions(phases, clt_trace)
        
        # Visualize
        print("\nGenerating visualizations...")
        visualizer = CLTVisualizer()
        
        # Temporal traces
        visualizer.plot_temporal_traces(
            clt_trace,
            cli_trace,
            save_path="xsum_temporal_traces.png"
        )
        print("Saved: xsum_temporal_traces.png")
        
        # Simplex - key visualization for phase identification
        visualizer.plot_simplex(
            clt_trace,
            save_path="xsum_simplex.png"
        )
        print("Saved: xsum_simplex.png")
        
        # Print statistics
        print("\n" + "=" * 80)
        print("CLT Statistics:")
        print("=" * 80)
        print(f"Average IL: {clt_trace[:, 0].mean():.3f}")
        print(f"Average EL: {clt_trace[:, 1].mean():.3f}")
        print(f"Average GL: {clt_trace[:, 2].mean():.3f}")
        print(f"Average CLI: {cli_trace.mean():.3f}")
        
        # Compare baseline vs load-guided
        print("\n" + "=" * 80)
        print("Comparing Baseline vs Load-Guided Summarization:")
        print("=" * 80)
        
        baseline_summary = run_baseline_summarization(model, tokenizer, article)
        print("\nBaseline summary:")
        print(baseline_summary)
        
        lgd_summary, intervention_log = run_load_guided_summarization(
            model, tokenizer, article, clt_computer, lgd_system
        )
        print("\nLoad-guided summary:")
        print(lgd_summary)
        
        print(f"\nApplied {len(intervention_log)} interventions")
        for interv in intervention_log[:5]:
            print(f"Step {interv['step']}: {interv['intervention']}")
        
    except Exception as e:
        print(f"Error during computation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

