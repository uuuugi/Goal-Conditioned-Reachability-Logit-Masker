"""
Benchmark: Complexity and Scaling Analysis (Paper Experiment 2 & 3)
Evaluates:
1. Offline BFS Build Time (ms) vs Number of States |S|
2. Memory Footprint (MB) vs Number of States |S|
3. Online Token Masking Latency (us) vs Number of States |S| (Empirical O(1) Verification)

Generates publication-quality figure: 'paper_figure_scaling.png'
"""

import os
import time
import torch
import matplotlib.pyplot as plt
from tabulate import tabulate
from typing import List, Dict

from core.fsm_builder import ReachabilityFSM
from core.logit_processor import GoalReachabilityLogitsProcessor


def run_scaling_benchmark(output_plot_path: str = "paper_figure_scaling.png"):
    print("\n" + "=" * 80)
    print(" [EXPERIMENT 2 & 3] FSM Complexity & O(1) Latency Scaling Benchmark")
    print("=" * 80)

    device = "cpu"  # Run on CPU for strict baseline consistency
    vocab_sizes = [32000, 151643]  # Standard 32k vs Qwen 151k
    state_counts = [10, 50, 200, 1000, 5000, 10000]
    max_budget = 50
    num_latency_trials = 500

    results: Dict[int, Dict[str, List]] = {
        v: {"states": [], "build_time_ms": [], "memory_mb": [], "online_latency_us": []}
        for v in vocab_sizes
    }

    table_rows = []

    for vocab_size in vocab_sizes:
        print(f"\n--- Testing Vocab Size: {vocab_size:,} ---")
        for num_states in state_counts:
            # 1. Build FSM structure
            fsm = ReachabilityFSM(num_states=num_states, vocab_size=vocab_size, device=device)
            # Add chain transitions + some branches
            for s in range(num_states - 1):
                fsm.add_transition(s, token_id=s % vocab_size, to_state=s + 1)
                # Add branching transition
                if s + 2 < num_states:
                    fsm.add_transition(s, token_id=(s * 7 + 13) % vocab_size, to_state=s + 2)

            fsm.set_goal_states([num_states - 1])

            # 2. Measure Offline BFS Build Time
            start_b = time.perf_counter()
            fsm.build_reachability(max_steps=max_budget)
            end_b = time.perf_counter()
            build_time_ms = (end_b - start_b) * 1000

            # 3. Measure Memory Footprint
            mem_bytes = fsm.memory_footprint_bytes()
            mem_mb = mem_bytes / (1024 * 1024)

            # 4. Measure Online 1-Token Masking Latency
            processor = GoalReachabilityLogitsProcessor(fsm=fsm, max_budget=max_budget)
            dummy_input = torch.tensor([[0]], dtype=torch.long, device=device)
            dummy_scores = torch.randn((1, vocab_size), device=device)

            # Warmup
            for _ in range(50):
                _ = processor(dummy_input, dummy_scores.clone())

            start_l = time.perf_counter()
            for _ in range(num_latency_trials):
                _ = processor(dummy_input, dummy_scores)
            end_l = time.perf_counter()

            online_latency_us = ((end_l - start_l) / num_latency_trials) * 1e6

            results[vocab_size]["states"].append(num_states)
            results[vocab_size]["build_time_ms"].append(build_time_ms)
            results[vocab_size]["memory_mb"].append(mem_mb)
            results[vocab_size]["online_latency_us"].append(online_latency_us)

            table_rows.append([
                f"{vocab_size:,}",
                f"{num_states:,}",
                f"{build_time_ms:.2f} ms",
                f"{mem_mb:.2f} MB",
                f"{online_latency_us:.2f} us ({online_latency_us/1000:.4f} ms)",
            ])

    headers = [
        "Vocab Size",
        "States |S|",
        "Offline BFS Time",
        "Memory Footprint",
        "Online 1-Token Latency",
    ]
    print("\n" + tabulate(table_rows, headers=headers, tablefmt="grid"))

    # -------------------------------------------------------------
    # Generate Publication-Quality Figures (Matplotlib)
    # -------------------------------------------------------------
    print(f"\n[PLOT] Generating publication-quality figures -> {output_plot_path}...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    colors = {32000: "#1f77b4", 151643: "#ff7f0e"}
    labels = {32000: "Vocab: 32,000 (LLaMA/Mistral)", 151643: "Vocab: 151,643 (Qwen2.5)"}

    # Plot 1: Offline BFS Build Time
    for v in vocab_sizes:
        axes[0].plot(
            results[v]["states"],
            results[v]["build_time_ms"],
            marker="o",
            linewidth=2,
            color=colors[v],
            label=labels[v],
        )
    axes[0].set_title("(a) Offline BFS Precomputation Time", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("FSM State Count |S|", fontsize=11)
    axes[0].set_ylabel("Build Time (ms)", fontsize=11)
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Memory Footprint
    for v in vocab_sizes:
        axes[1].plot(
            results[v]["states"],
            results[v]["memory_mb"],
            marker="s",
            linewidth=2,
            color=colors[v],
            label=labels[v],
        )
    axes[1].set_title("(b) Memory Footprint (VRAM / RAM)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("FSM State Count |S|", fontsize=11)
    axes[1].set_ylabel("Memory (MB)", fontsize=11)
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Plot 3: Online 1-Token Latency (O(1) Verification)
    for v in vocab_sizes:
        axes[2].plot(
            results[v]["states"],
            results[v]["online_latency_us"],
            marker="^",
            linewidth=2,
            color=colors[v],
            label=labels[v],
        )
    axes[2].set_title("(c) Online Runtime Latency: Strict O(1)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("FSM State Count |S|", fontsize=11)
    axes[2].set_ylabel("Latency per Token (us)", fontsize=11)
    axes[2].set_xscale("log")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_plot_path, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Figure successfully saved to: {os.path.abspath(output_plot_path)}\n")


if __name__ == "__main__":
    run_scaling_benchmark()
