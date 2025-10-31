"""
Example usage of CLT framework on GSM8K math reasoning tasks.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cogload import CognitiveLoadTraces, CLTVisualizer, LoadGuidedDecoding
import numpy as np
from typing import List, Dict


def setup_model(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded successfully")
    return model, tokenizer


def generate_example_problem() -> str:
    """Generate a sample GSM8K math problem."""
    problem = """Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning.
She bakes 4 for her friends every day. She sells the remainder at the farmers' market
for $2 per duck egg. How much in dollars does she make every day at the farmers' market?"""
    return problem


def compute_clt_for_problem(model, tokenizer, problem: str, clt_computer):
    """Compute CLT trace for a reasoning problem."""
    # Prepare input
    prompt = f"Question: {problem}\nAnswer: Let's think step by step."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Computing CLT trace...")
    clt_trace, cli_trace, metadata = clt_computer.compute_clt_trace(
        model,
        inputs.input_ids,
        attention_mask=inputs.attention_mask
    )
    
    return clt_trace, cli_trace, metadata


def identify_error_steps(cli_trace: np.ndarray, threshold: float = 0.8) -> List[int]:
    """
    Identify potential error steps based on CLI spikes.
    This is a simplified approach; in practice you'd compare against ground truth.
    """
    error_steps = []
    for i, cli_val in enumerate(cli_trace):
        if cli_val > threshold:
            error_steps.append(i)
    return error_steps


def run_baseline_decoding(model, tokenizer, problem: str, max_tokens: int = 200):
    """Run baseline decoding without interventions."""
    prompt = f"Question: {problem}\nAnswer: Let's think step by step."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
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
    return response


def run_load_guided_decoding(
    model, 
    tokenizer, 
    problem: str, 
    clt_computer,
    lgd_system,
    max_tokens: int = 200
):
    """Run load-guided decoding with interventions."""
    prompt = f"Question: {problem}\nAnswer: Let's think step by step."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
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
    
    # Generate with interventions (simplified - use computed parameters)
    temperature = 0.7
    top_k = 50
    
    # Apply intervention parameters
    for intervention in intervention_history:
        if 'Control' in intervention['intervention']:
            temperature = temperature * 0.8  # Reduce temperature
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
    return response, intervention_history


def main():
    """Main example workflow."""
    print("=" * 80)
    print("Cognitive Load Traces (CLT) - GSM8K Example")
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
    
    # Load model (use smaller model for demo)
    print("\nNote: Using a small model for demonstration.")
    print("In practice, use models like Mistral-7B-Instruct, LLaMA-3-8B, etc.")
    
    try:
        model, tokenizer = setup_model("microsoft/phi-2")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please install a compatible model or adjust model_name")
        return
    
    # Generate example problem
    problem = generate_example_problem()
    print(f"\nProblem: {problem}")
    
    # Compute CLT trace
    print("\nComputing CLT trace...")
    try:
        clt_trace, cli_trace, metadata = compute_clt_for_problem(
            model, tokenizer, problem, clt_computer
        )
        print(f"CLT trace shape: {clt_trace.shape}")
        print(f"CLI trace shape: {cli_trace.shape}")
        
        # Identify error steps
        error_steps = identify_error_steps(cli_trace, threshold=0.8)
        print(f"Identified {len(error_steps)} high-load steps")
        
        # Visualize
        print("\nGenerating visualizations...")
        visualizer = CLTVisualizer()
        
        # Temporal traces
        visualizer.plot_temporal_traces(
            clt_trace,
            cli_trace,
            error_events=error_steps,
            save_path="gsm8k_temporal_traces.png"
        )
        print("Saved: gsm8k_temporal_traces.png")
        
        # Simplex
        visualizer.plot_simplex(
            clt_trace,
            save_path="gsm8k_simplex.png"
        )
        print("Saved: gsm8k_simplex.png")
        
        # Correlation analysis
        visualizer.plot_correlation_analysis(
            clt_trace,
            cli_trace,
            error_steps,
            save_path="gsm8k_correlation.png"
        )
        print("Saved: gsm8k_correlation.png")
        
        # Print statistics
        print("\n" + "=" * 80)
        print("CLT Statistics:")
        print("=" * 80)
        print(f"Average IL: {clt_trace[:, 0].mean():.3f}")
        print(f"Average EL: {clt_trace[:, 1].mean():.3f}")
        print(f"Average GL: {clt_trace[:, 2].mean():.3f}")
        print(f"Average CLI: {cli_trace.mean():.3f}")
        print(f"Max CLI: {cli_trace.max():.3f} at step {cli_trace.argmax()}")
        
        # Compare baseline vs load-guided
        print("\n" + "=" * 80)
        print("Comparing Baseline vs Load-Guided Decoding:")
        print("=" * 80)
        
        baseline_response = run_baseline_decoding(model, tokenizer, problem)
        print("\nBaseline response:")
        print(baseline_response)
        
        lgd_response, intervention_log = run_load_guided_decoding(
            model, tokenizer, problem, clt_computer, lgd_system
        )
        print("\nLoad-guided response:")
        print(lgd_response)
        
        print(f"\nApplied {len(intervention_log)} interventions")
        for interv in intervention_log[:5]:  # Show first 5
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

