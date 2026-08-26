import pytest
import torch
from core.fsm_builder import ReachabilityFSM


def test_fsm_basic_reachability():
    # Linear graph: 0 -> 1 -> 2 (Goal)
    vocab_size = 10
    fsm = ReachabilityFSM(num_states=3, vocab_size=vocab_size)
    fsm.add_transition(0, token_id=1, to_state=1)
    fsm.add_transition(1, token_id=2, to_state=2)
    fsm.set_goal_states([2])

    # Max steps = 2
    reach = fsm.build_reachability(max_steps=2, allow_early_finish=True)

    # t = 0: only state 2 is True
    assert reach[0, 2].item() is True
    assert reach[0, 1].item() is False
    assert reach[0, 0].item() is False

    # t = 1: state 1 and 2 are True
    assert reach[1, 2].item() is True
    assert reach[1, 1].item() is True
    assert reach[1, 0].item() is False

    # t = 2: state 0, 1, 2 are all True
    assert reach[2, 0].item() is True
    assert reach[2, 1].item() is True
    assert reach[2, 2].item() is True


def test_fsm_deadend_reachability():
    # Branching graph:
    # 0 -> 1 -> 2 (Goal) via token 1, 2 (needs 2 steps)
    # 0 -> 3 -> 4 (Dead-end) via token 3, 4 (sink)
    vocab_size = 10
    fsm = ReachabilityFSM(num_states=5, vocab_size=vocab_size)
    fsm.add_transition(0, token_id=1, to_state=1)
    fsm.add_transition(1, token_id=2, to_state=2)
    fsm.add_transition(0, token_id=3, to_state=3)
    fsm.add_transition(3, token_id=4, to_state=4)
    fsm.set_goal_states([2])

    reach = fsm.build_reachability(max_steps=5, allow_early_finish=True)

    # States 3 and 4 should NEVER be reachable to goal
    for t in range(6):
        assert reach[t, 3].item() is False
        assert reach[t, 4].item() is False

    # State 0 is reachable only when t >= 2
    assert reach[0, 0].item() is False
    assert reach[1, 0].item() is False
    assert reach[2, 0].item() is True
    assert reach[3, 0].item() is True


def test_fsm_multi_goal():
    # 0 -> 1 (Goal A), 0 -> 2 (Goal B)
    vocab_size = 5
    fsm = ReachabilityFSM(num_states=3, vocab_size=vocab_size)
    fsm.add_transition(0, 1, 1)
    fsm.add_transition(0, 2, 2)
    fsm.set_goal_states([1, 2])

    reach = fsm.build_reachability(max_steps=1)
    assert reach[0, 1].item() is True
    assert reach[0, 2].item() is True
    assert reach[0, 0].item() is False
    assert reach[1, 0].item() is True
