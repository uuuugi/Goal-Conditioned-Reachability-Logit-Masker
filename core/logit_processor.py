from typing import Optional, Union
import torch
from transformers.generation.logits_process import LogitsProcessor

from .fsm_builder import ReachabilityFSM


class GoalReachabilityLogitsProcessor(LogitsProcessor):
    """
    Hugging Face compatible LogitsProcessor for O(1) Goal-Conditioned Reachability Masking.

    Guarantees that generated tokens stay on paths that can reach the goal state
    within the remaining token budget T_rem.

    Args:
        fsm (ReachabilityFSM): Compiled FSM with computed reachability table.
        max_budget (int): Maximum token budget (max_new_tokens) allocated for generation.
        allow_early_finish (bool): Whether finishing at goal state before budget is allowed.
    """

    def __init__(
        self,
        fsm: ReachabilityFSM,
        max_budget: int,
        allow_early_finish: bool = True,
    ):
        if fsm.reachability_table is None:
            fsm.build_reachability(max_steps=max_budget, allow_early_finish=allow_early_finish)

        self.fsm = fsm
        self.max_budget = max_budget
        self.allow_early_finish = allow_early_finish

        self.prompt_lengths: Optional[torch.Tensor] = None
        self.current_states: Optional[torch.Tensor] = None
        self.last_seq_lengths: Optional[torch.Tensor] = None

    def reset(self, batch_size: int = 1, initial_state: Optional[int] = None, device: Optional[torch.device] = None) -> None:
        """Reset internal state tracker for a new generation run."""
        dev = device if device is not None else self.fsm.device
        init_s = initial_state if initial_state is not None else self.fsm.initial_state
        self.current_states = torch.full((batch_size,), init_s, dtype=torch.long, device=dev)
        self.prompt_lengths = None
        self.last_seq_lengths = None

    def _initialize_tracker(self, input_ids: torch.LongTensor) -> None:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Ensure FSM tensors are on the same device as input_ids
        if self.fsm.device != device:
            self.fsm.to(device)

        self.prompt_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        self.last_seq_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        self.current_states = torch.full((batch_size,), self.fsm.initial_state, dtype=torch.long, device=device)

    def get_state(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Return the up-to-date state for input_ids."""
        if self.prompt_lengths is None:
            self._initialize_tracker(input_ids)
        else:
            self._update_states(input_ids)
        return self.current_states

    def _update_states(self, input_ids: torch.LongTensor) -> None:
        batch_size, seq_len = input_ids.shape
        if self.last_seq_lengths is None:
            return

        # Check if new tokens have been added since last call
        new_tokens_count = seq_len - self.last_seq_lengths
        if (new_tokens_count > 0).any():
            last_tokens = input_ids[:, -1]
            next_states = self.fsm.transitions[self.current_states, last_tokens]
            
            # If transition is valid (>= 0), update state
            valid_update = next_states >= 0
            self.current_states = torch.where(valid_update, next_states, self.current_states)
            self.last_seq_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=input_ids.device)

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Mask logits based on O(1) reachability lookup.

        Args:
            input_ids (torch.LongTensor): [batch_size, sequence_length]
            scores (torch.FloatTensor): [batch_size, vocab_size]

        Returns:
            torch.FloatTensor: Masked logits with unreachable transitions set to -inf.
        """
        batch_size, seq_len = input_ids.shape

        if self.prompt_lengths is None or self.current_states is None or len(self.current_states) != batch_size:
            self._initialize_tracker(input_ids)
        else:
            self._update_states(input_ids)

        # Remaining steps for each sample in the batch: [batch_size]
        generated_steps = seq_len - self.prompt_lengths
        t_rem = self.max_budget - generated_steps

        # Fast path for batch_size == 1 (common for interactive LLM generation)
        if batch_size == 1:
            s_curr = self.current_states[0]
            next_states = self.fsm.transitions[s_curr]  # [vocab_size]
            valid_trans = next_states >= 0
            clamped_next = torch.clamp(next_states, min=0)
            
            step_idx = max(0, min(t_rem[0].item() - 1, self.fsm.max_steps))
            reach_row = self.fsm.reachability_table[step_idx]  # [num_states]
            reachable = reach_row[clamped_next]  # [vocab_size]

            valid_mask = valid_trans & reachable
            if not valid_mask.any():
                valid_mask = valid_trans

            scores.masked_fill_(~valid_mask.unsqueeze(0), float("-inf"))
            return scores

        # General batch path
        next_states = self.fsm.transitions[self.current_states]  # [batch_size, vocab_size]
        valid_transitions = (next_states >= 0)
        clamped_next = torch.clamp(next_states, min=0)

        step_idx = torch.clamp(t_rem - 1, min=0, max=self.fsm.max_steps).unsqueeze(1)  # [batch_size, 1]
        reachable = self.fsm.reachability_table[step_idx, clamped_next]  # [batch_size, vocab_size]

        valid_mask = valid_transitions & reachable
        has_any_valid = valid_mask.any(dim=-1, keepdim=True)
        effective_mask = torch.where(has_any_valid, valid_mask, valid_transitions)

        scores.masked_fill_(~effective_mask, float("-inf"))
        return scores
