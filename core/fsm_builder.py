from typing import Iterable, List, Optional, Set, Union
import torch


class ReachabilityFSM:
    """
    Finite State Machine with backward BFS reachability bitmap for O(1) logit masking.
    
    Attributes:
        num_states (int): Total number of states in the FSM.
        vocab_size (int): Size of the token vocabulary.
        transitions (torch.Tensor): Tensor of shape [num_states, vocab_size] storing next state ID (-1 if invalid).
        goal_states (Set[int]): Set of target/accepting state IDs.
        reachability_table (torch.Tensor): Boolean tensor of shape [max_steps + 1, num_states].
        device (torch.device): Device on which tensors reside.
    """

    def __init__(
        self,
        num_states: int,
        vocab_size: int,
        goal_states: Optional[Iterable[int]] = None,
        initial_state: int = 0,
        device: Union[str, torch.device] = "cpu",
    ):
        self.num_states = num_states
        self.vocab_size = vocab_size
        self.initial_state = initial_state
        self.device = torch.device(device)

        # Transitions: [num_states, vocab_size] initialized with -1 (no transition)
        self.transitions = torch.full(
            (num_states, vocab_size),
            -1,
            dtype=torch.long,
            device=self.device,
        )

        self.goal_states: Set[int] = set(goal_states) if goal_states is not None else set()
        self.is_goal_state = torch.zeros(num_states, dtype=torch.bool, device=self.device)
        for g in self.goal_states:
            self.is_goal_state[g] = True

        self.reachability_table: Optional[torch.Tensor] = None
        self.max_steps: Optional[int] = None

    def add_transition(self, from_state: int, token_id: int, to_state: int) -> None:
        """Add a transition for a single token ID."""
        if not (0 <= from_state < self.num_states):
            raise ValueError(f"from_state {from_state} out of bounds [0, {self.num_states})")
        if not (0 <= to_state < self.num_states):
            raise ValueError(f"to_state {to_state} out of bounds [0, {self.num_states})")
        if not (0 <= token_id < self.vocab_size):
            raise ValueError(f"token_id {token_id} out of bounds [0, {self.vocab_size})")

        self.transitions[from_state, token_id] = to_state

    def add_transitions(self, from_state: int, token_ids: Iterable[int], to_state: int) -> None:
        """Add transitions for multiple token IDs simultaneously."""
        for tid in token_ids:
            self.add_transition(from_state, tid, to_state)

    def set_goal_states(self, goal_states: Iterable[int]) -> None:
        """Set target/accepting states."""
        self.goal_states = set()
        self.is_goal_state.zero_()
        for g in goal_states:
            if not (0 <= g < self.num_states):
                raise ValueError(f"goal_state {g} out of bounds [0, {self.num_states})")
            self.goal_states.add(g)
            self.is_goal_state[g] = True

    def build_reachability(self, max_steps: int, allow_early_finish: bool = True) -> torch.Tensor:
        """
        Compute backward BFS reachability table R[t, s] using vectorized PyTorch operations.
        
        R[t, s] == True iff state s can reach at least one goal state in:
          - <= t steps (if allow_early_finish=True)
          - exactly t steps (if allow_early_finish=False)

        Args:
            max_steps (int): Maximum token budget T_max.
            allow_early_finish (bool): If True, reaching goal in <= t steps is considered reachable.

        Returns:
            torch.Tensor: Boolean tensor of shape [max_steps + 1, num_states].
        """
        if not self.goal_states:
            raise ValueError("No goal states specified. Call set_goal_states() first.")

        self.max_steps = max_steps
        table = torch.zeros((max_steps + 1, self.num_states), dtype=torch.bool, device=self.device)

        # Base case t = 0: only goal states are reachable in 0 steps
        for g in self.goal_states:
            table[0, g] = True

        valid_trans = self.transitions >= 0
        clamped_trans = torch.clamp(self.transitions, min=0)

        for t in range(1, max_steps + 1):
            prev_reachable = table[t - 1]  # [num_states]
            
            # For each transition (s, v) -> next_s, check if next_s is reachable in t-1 steps
            # trans_reachable: [num_states, vocab_size]
            trans_reachable = prev_reachable[clamped_trans] & valid_trans
            
            # A state s can transition to a reachable state if any token v leads to a reachable next_s
            can_reach = trans_reachable.any(dim=1)  # [num_states]

            if allow_early_finish:
                table[t] = table[t - 1] | can_reach
            else:
                table[t] = can_reach

        self.reachability_table = table
        return self.reachability_table

    def to(self, device: Union[str, torch.device]) -> "ReachabilityFSM":
        """Move FSM tensors to specified device."""
        self.device = torch.device(device)
        self.transitions = self.transitions.to(self.device)
        self.is_goal_state = self.is_goal_state.to(self.device)
        if self.reachability_table is not None:
            self.reachability_table = self.reachability_table.to(self.device)
        return self

    def memory_footprint_bytes(self) -> int:
        """Calculate total memory usage of FSM tensors in bytes."""
        trans_bytes = self.transitions.numel() * self.transitions.element_size()
        reach_bytes = 0
        if self.reachability_table is not None:
            reach_bytes = self.reachability_table.numel() * self.reachability_table.element_size()
        return trans_bytes + reach_bytes
