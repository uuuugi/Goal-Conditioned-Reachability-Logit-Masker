"""
Benchmark: Dead-end Trap Avoidance and Strict Budget Goal Guarantee
Compares:
1. Vanilla (Unconstrained Generation)
2. Standard 1-way DFA Masking (Outlines/SGLang style)
3. GCLM (Goal-Conditioned Reachability Logit Masker)
"""

import random
import torch
import numpy as np
from tabulate import tabulate

from core.fsm_builder import ReachabilityFSM
from core.logit_processor import GoalReachabilityLogitsProcessor


def run_deadend_simulation(num_trials: int = 1000, max_budget: int = 3, device: str = "cpu"):
    """
    Scenario:
    - Alphabet: {Token 1: 'B_path', Token 2: 'C_step', Token 3: 'Goal', Token 4: 'Deadend_entry', Token 5: 'Deadend_sink'}
    - Goal Path: 0 -> 1 -> 2 -> 3 (Goal) [Takes 3 steps: Token 1, 2, 3]
    - Dead-end Path: 0 -> 4 -> 5 (Sink) [Takes 2 steps: Token 4, 5, then stuck]
    - At State 0: Model has equal likelihood of choosing Token 1 (Success) or Token 4 (Dead-end).
    - Budget is exactly 3 tokens.
    """
    vocab_size = 10
    tok_succ_1, tok_succ_2, tok_succ_goal = 1, 2, 3
    tok_dead_1, tok_dead_2 = 4, 5

    # 1. Build FSM
    # States: 0(Start), 1, 2, 3(Goal), 4(Dead 1), 5(Dead 2 Sink)
    num_states = 6
    goal_state = 3

    fsm = ReachabilityFSM(num_states=num_states, vocab_size=vocab_size, device=device)
    # Success branch
    fsm.add_transition(0, tok_succ_1, 1)
    fsm.add_transition(1, tok_succ_2, 2)
    fsm.add_transition(2, tok_succ_goal, 3)
    fsm.add_transition(3, tok_succ_goal, 3)  # goal self-loop

    # Dead-end branch
    fsm.add_transition(0, tok_dead_1, 4)
    fsm.add_transition(4, tok_dead_2, 5)

    fsm.set_goal_states([goal_state])
    fsm.build_reachability(max_steps=max_budget)

    # 2. Simulate 3 strategies
    results = {"Vanilla (Unconstrained)": 0, "Forward DFA Masking": 0, "GCLM (Ours)": 0}

    for trial in range(num_trials):
        # -------------------------------------------------------------
        # 1. Vanilla (Random choice over all vocab or uniform logits)
        # -------------------------------------------------------------
        curr_s = 0
        reached_goal = False
        for step in range(max_budget):
            # Vanilla chooses randomly among valid tokens or any vocab
            token = random.choice([tok_succ_1, tok_succ_2, tok_succ_goal, tok_dead_1, tok_dead_2])
            next_s = fsm.transitions[curr_s, token].item()
            if next_s >= 0:
                curr_s = next_s
                if curr_s == goal_state:
                    reached_goal = True
                    break
            else:
                break
        if reached_goal:
            results["Vanilla (Unconstrained)"] += 1

        # -------------------------------------------------------------
        # 2. Forward DFA Masker (Only checks if transition >= 0 from current state)
        # -------------------------------------------------------------
        curr_s = 0
        reached_goal = False
        for step in range(max_budget):
            # Forward DFA allows any valid forward transition from curr_s
            valid_tokens = [v for v in range(vocab_size) if fsm.transitions[curr_s, v].item() >= 0]
            if not valid_tokens:
                break
            # Uniform random choice among valid forward transitions
            token = random.choice(valid_tokens)
            curr_s = fsm.transitions[curr_s, token].item()
            if curr_s == goal_state:
                reached_goal = True
                break
        if reached_goal:
            results["Forward DFA Masking"] += 1

        # -------------------------------------------------------------
        # 3. GCLM (Goal-Conditioned Reachability Logit Masker)
        # -------------------------------------------------------------
        processor = GoalReachabilityLogitsProcessor(fsm=fsm, max_budget=max_budget)
        curr_ids = torch.tensor([[0]], dtype=torch.long, device=device)
        reached_goal = False

        for step in range(max_budget):
            raw_logits = torch.randn((1, vocab_size), device=device)  # Random model logits
            masked_logits = processor(curr_ids, raw_logits)

            # Sample from masked logits
            valid_indices = torch.where(masked_logits[0] > -float("inf"))[0]
            if len(valid_indices) == 0:
                break

            # Pick randomly from valid options according to softmax
            probs = torch.softmax(masked_logits[0, valid_indices], dim=-1)
            selected_idx = torch.multinomial(probs, 1).item()
            selected_token = valid_indices[selected_idx].item()

            curr_ids = torch.cat([curr_ids, torch.tensor([[selected_token]], device=device)], dim=1)

            # Check up-to-date state after token appending
            curr_state = processor.get_state(curr_ids)[0].item()
            if curr_state == goal_state:
                reached_goal = True
                break

        if reached_goal:
            results["GCLM (Ours)"] += 1

    table_data = []
    for method, successes in results.items():
        rate = (successes / num_trials) * 100
        table_data.append([method, f"{successes}/{num_trials}", f"{rate:.2f}%"])

    print("\n" + "=" * 60)
    print(" [BENCHMARK] Dead-End Trap & Budget Constraint")
    print(f" (Trials: {num_trials}, Max Budget: {max_budget} tokens)")
    print("=" * 60)
    print(tabulate(table_data, headers=["Method", "Successes", "Goal Reach Rate (%)"], tablefmt="grid"))
    print("\nKey Insight: GCLM eliminates dead-end branches in advance by backward BFS reachability lookup.\n")


if __name__ == "__main__":
    run_deadend_simulation()
