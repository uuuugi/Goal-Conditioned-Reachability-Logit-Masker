from typing import Any, Dict, List, Optional, Sequence, Union
import torch

from .fsm_builder import ReachabilityFSM


class FSMCompiler:
    """
    Utility to compile high-level structure definitions into a ReachabilityFSM.
    Supports token matching with HuggingFace Tokenizers.
    """

    def __init__(self, vocab_size: int, tokenizer: Optional[Any] = None, device: str = "cpu"):
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.device = device

    def _get_token_ids(self, text_or_tokens: Union[str, int, Sequence[int]]) -> List[int]:
        """Convert string or token IDs into a list of token IDs."""
        if isinstance(text_or_tokens, int):
            return [text_or_tokens]
        if isinstance(text_or_tokens, (list, tuple)):
            return list(text_or_tokens)
        if isinstance(text_or_tokens, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer required to convert string to token IDs.")
            tokens = self.tokenizer.encode(text_or_tokens, add_special_tokens=False)
            return tokens
        raise TypeError(f"Unsupported token specification: {type(text_or_tokens)}")

    def build_synthetic_deadend_fsm(
        self,
        token_success_path: Sequence[int],
        token_deadend_path: Sequence[int],
        eos_token_id: int,
    ) -> ReachabilityFSM:
        """
        Creates a synthetic dead-end trap FSM:
        State 0 (A): Start
        Path 1 (Success): 0 -> 1 -> 2 -> ... -> Goal (accepts token_success_path, then eos)
        Path 2 (Dead-end): 0 -> D1 -> D2 -> ... -> Sink (accepts token_deadend_path, no goal)
        """
        len_succ = len(token_success_path)
        len_dead = len(token_deadend_path)

        # State 0: Start
        # States 1 to len_succ: Success path
        # Goal state: len_succ + 1 (emits EOS and self-loops)
        # Dead-end states: len_succ + 2 to len_succ + 1 + len_dead
        # Dead-end sink: len_succ + 2 + len_dead

        goal_state = len_succ + 1
        num_states = goal_state + len_dead + 2

        fsm = ReachabilityFSM(num_states=num_states, vocab_size=self.vocab_size, device=self.device)

        # Success path
        curr = 0
        for i, tid in enumerate(token_success_path):
            next_s = i + 1
            fsm.add_transition(curr, tid, next_s)
            curr = next_s

        # Success path final step -> Goal
        fsm.add_transition(curr, eos_token_id, goal_state)
        # Goal state self loop
        fsm.add_transition(goal_state, eos_token_id, goal_state)

        # Dead-end path
        curr = 0
        dead_start = len_succ + 2
        for i, tid in enumerate(token_deadend_path):
            next_s = dead_start + i
            fsm.add_transition(curr, tid, next_s)
            curr = next_s

        # Sink state for dead-end (no transition to goal)
        sink = num_states - 1
        fsm.add_transition(curr, eos_token_id, sink)

        fsm.set_goal_states([goal_state])
        return fsm

    def build_strict_budget_json_fsm(
        self,
        open_bracket_tokens: Sequence[int],
        key_value_tokens_list: List[Sequence[int]],
        close_bracket_tokens: Sequence[int],
        eos_token_id: int,
    ) -> ReachabilityFSM:
        """
        Builds a JSON schema FSM with multiple optional fields and guaranteed closing:
        - Must start with '{'
        - Can generate key-value pairs in sequence or loop
        - Can close with '}' and EOS at any point, but MUST close with '}' before EOS.
        """
        # States:
        # 0: Pre-open
        # 1: Inside object (after open bracket)
        # KV states: intermediate steps for generating keys & values
        # Close state: After '}'
        # Goal state: After EOS

        # Let's create an FSM where:
        # 0 --(open_bracket)--> 1
        # 1 --(close_bracket)--> Close
        # 1 --(KV_path)--> 1 (loop for next fields)
        # Close --(eos)--> Goal (Goal self-loops on eos)

        state_counter = 2
        kv_routes = []
        for kv in key_value_tokens_list:
            route = []
            for tid in kv:
                route.append((state_counter, tid))
                state_counter += 1
            kv_routes.append(route)

        close_state = state_counter
        state_counter += 1
        goal_state = state_counter
        state_counter += 1

        fsm = ReachabilityFSM(num_states=state_counter, vocab_size=self.vocab_size, device=self.device)

        # 0 -> 1 on open bracket
        for tok in open_bracket_tokens:
            fsm.add_transition(0, tok, 1)

        # 1 -> Close on close bracket
        for tok in close_bracket_tokens:
            fsm.add_transition(1, tok, close_state)

        # KV branches from 1 and returning to 1
        for route in kv_routes:
            curr = 1
            for i, (next_s, tid) in enumerate(route):
                target = next_s if i < len(route) - 1 else 1  # loop back to state 1
                fsm.add_transition(curr, tid, target)
                curr = next_s

        # Close -> Goal on EOS
        fsm.add_transition(close_state, eos_token_id, goal_state)
        # Goal self-loop
        fsm.add_transition(goal_state, eos_token_id, goal_state)

        fsm.set_goal_states([goal_state])
        return fsm
