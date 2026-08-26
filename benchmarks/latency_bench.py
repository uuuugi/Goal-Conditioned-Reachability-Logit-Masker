"""
Benchmark: Per-Token Masking Latency Benchmark (O(1) Verification)
Measures the runtime overhead of GCLM across different vocab sizes and batch sizes.
"""

import time
import torch
from tabulate import tabulate

from core.fsm_builder import ReachabilityFSM
from core.logit_processor import GoalReachabilityLogitsProcessor


def benchmark_latency(device: str = "cpu"):
    print(f"\n[BENCHMARK] Running GCLM Latency Benchmark on Device: {device.upper()}")
    
    vocab_sizes = [32000, 151643]  # Standard LLaMA vs Qwen2.5 vocab sizes
    batch_sizes = [1, 4, 16, 64]
    num_states = 100
    max_budget = 50
    num_iterations = 1000
    warmup = 100

    results = []

    for vocab_size in vocab_sizes:
        # Build a synthetic FSM
        fsm = ReachabilityFSM(num_states=num_states, vocab_size=vocab_size, device=device)
        # Add random transitions
        for s in range(num_states - 1):
            fsm.add_transition(s, token_id=s % vocab_size, to_state=s + 1)
        fsm.set_goal_states([num_states - 1])
        fsm.build_reachability(max_steps=max_budget)

        for batch_size in batch_sizes:
            processor = GoalReachabilityLogitsProcessor(fsm=fsm, max_budget=max_budget)
            
            input_ids = torch.randint(0, vocab_size, (batch_size, 10), device=device)
            scores = torch.randn((batch_size, vocab_size), device=device)

            # Warmup
            for _ in range(warmup):
                _ = processor(input_ids, scores.clone())

            # Timed iterations
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_t = time.perf_counter()
            for _ in range(num_iterations):
                _ = processor(input_ids, scores)
            
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_t = time.perf_counter()

            total_time_ms = (end_t - start_t) * 1000
            per_step_us = (total_time_ms / num_iterations) * 1000
            per_step_ms = total_time_ms / num_iterations
            per_sample_us = per_step_us / batch_size

            results.append([
                f"{vocab_size:,}",
                batch_size,
                f"{per_step_us:.2f} us ({per_step_ms:.4f} ms)",
                f"{per_sample_us:.2f} us",
                "PASS (< 0.1 ms)" if per_step_ms < 0.1 else "FAIL (>= 0.1 ms)"
            ])

    headers = [
        "Vocab Size",
        "Batch Size",
        "Total Step Latency",
        "Per-Sample Latency",
        "O(1) Overhead Target"
    ]
    print("\n" + "=" * 80)
    print(" [BENCHMARK] GCLM Runtime Overhead (O(1) Verification)")
    print(f" (Iterations: {num_iterations:,}, States: {num_states}, Budget: {max_budget})")
    print("=" * 80)
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print("\n")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    benchmark_latency(device=dev)
