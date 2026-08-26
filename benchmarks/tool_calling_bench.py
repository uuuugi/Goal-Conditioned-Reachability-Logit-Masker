"""
Benchmark: Multi-Step Agent Tool-Calling / ReAct Dead-end Benchmark
Evaluates whether GCLM prevents an LLM agent from getting trapped in retry loops
or branching into deep API sub-trees that cannot finish within the tool-call budget.
"""

import random
import torch
from tabulate import tabulate

from core.fsm_builder import ReachabilityFSM
from core.logit_processor import GoalReachabilityLogitsProcessor


def run_tool_calling_benchmark(num_trials: int = 500, device: str = "cpu"):
    print("\n" + "=" * 80)
    print(" [EXPERIMENT 4] Multi-Step Agent Tool-Calling & Action Budget Benchmark")
    print(f" (Trials per budget: {num_trials:,}, Device: {device.upper()})")
    print("=" * 80)

    # Tool Tokens:
    # 1: 'DB_Query', 2: 'Filter', 3: 'Summarize', 4: 'Finish_Submit' (Goal)
    # 5: 'Web_Search', 6: 'Parse_HTML', 7: 'Format_Data'
    # 8: 'Obsolete_API', 9: 'Retry_Loop_Sink'
    vocab_size = 20

    # Paths:
    # Short Path: 0 -> DB_Query(1) -> Summarize(3) -> Finish(4) [3 steps]
    # Standard Path: 0 -> DB_Query(1) -> Filter(2) -> Summarize(3) -> Finish(4) [4 steps]
    # Long Path: 0 -> Web_Search(5) -> Parse_HTML(6) -> Format_Data(7) -> Summarize(3) -> Finish(4) [5 steps]
    # Trap Path: 0 -> Obsolete_API(8) -> Retry_Loop_Sink(9) [Stuck in loop]

    num_states = 9
    goal_state = 8

    fsm = ReachabilityFSM(num_states=num_states, vocab_size=vocab_size, device=device)

    # DB Branch
    fsm.add_transition(0, 1, 1)  # 0 -> DB(1) -> 1
    fsm.add_transition(1, 2, 2)  # 1 -> Filter(2) -> 2
    fsm.add_transition(2, 3, 3)  # 2 -> Summarize(3) -> 3
    fsm.add_transition(1, 3, 3)  # Fast path: 1 -> Summarize(3) -> 3
    fsm.add_transition(3, 4, goal_state)  # 3 -> Finish(4) -> Goal

    # Web Branch
    fsm.add_transition(0, 5, 4)  # 0 -> Web(5) -> 4
    fsm.add_transition(4, 6, 5)  # 4 -> Parse(6) -> 5
    fsm.add_transition(5, 7, 6)  # 5 -> Format(7) -> 6
    fsm.add_transition(6, 3, 3)  # 6 -> Summarize(3) -> 3

    # Trap Branch
    fsm.add_transition(0, 8, 7)  # 0 -> Obsolete(8) -> 7
    fsm.add_transition(7, 9, 7)  # 7 -> Retry(9) -> 7 (infinite loop)

    # Goal self loop
    fsm.add_transition(goal_state, 4, goal_state)
    fsm.set_goal_states([goal_state])

    budgets = [3, 4, 5, 8]
    summary_rows = []

    for budget in budgets:
        fsm.build_reachability(max_steps=budget, allow_early_finish=True)
        results = {"Vanilla": 0, "Forward DFA": 0, "GCLM (Ours)": 0}

        # 1. Vanilla
        for _ in range(num_trials):
            curr_s = 0
            for _ in range(budget):
                tok = random.randint(1, 9)
                next_s = fsm.transitions[curr_s, tok].item()
                if next_s >= 0:
                    curr_s = next_s
                    if curr_s == goal_state:
                        results["Vanilla"] += 1
                        break
                else:
                    break

        # 2. Forward DFA
        for _ in range(num_trials):
            curr_s = 0
            for _ in range(budget):
                valid_toks = [v for v in range(vocab_size) if fsm.transitions[curr_s, v].item() >= 0]
                if not valid_toks:
                    break
                tok = random.choice(valid_toks)
                curr_s = fsm.transitions[curr_s, tok].item()
                if curr_s == goal_state:
                    results["Forward DFA"] += 1
                    break

        # 3. GCLM
        processor = GoalReachabilityLogitsProcessor(fsm=fsm, max_budget=budget)
        for _ in range(num_trials):
            curr_ids = torch.tensor([[0]], dtype=torch.long, device=device)
            processor.reset(batch_size=1, device=torch.device(device))
            for _ in range(budget):
                logits = torch.randn((1, vocab_size), device=device)
                masked_logits = processor(curr_ids, logits)

                valid_indices = torch.where(masked_logits[0] > -float("inf"))[0]
                if len(valid_indices) == 0:
                    break

                probs = torch.softmax(masked_logits[0, valid_indices], dim=-1)
                selected_idx = torch.multinomial(probs, 1).item()
                tok = valid_indices[selected_idx].item()

                curr_ids = torch.cat([curr_ids, torch.tensor([[tok]], device=device)], dim=1)
                curr_s = processor.get_state(curr_ids)[0].item()
                if curr_s == goal_state:
                    results["GCLM (Ours)"] += 1
                    break

        for m in ["Vanilla", "Forward DFA", "GCLM (Ours)"]:
            rate = (results[m] / num_trials) * 100
            summary_rows.append([
                f"Budget = {budget} actions",
                m,
                f"{results[m]}/{num_trials}",
                f"{rate:.2f}%",
            ])

    headers = ["Action Budget", "Method", "Completed Tasks", "Success Rate (%)"]
    print(tabulate(summary_rows, headers=headers, tablefmt="grid"))
    print("\nKey Finding: Under tight action budgets (3-4 steps), Forward DFA fails because it picks long/trap branches. GCLM dynamically restricts the search space to feasible shortest-paths only.\n")


if __name__ == "__main__":
    run_tool_calling_benchmark()
