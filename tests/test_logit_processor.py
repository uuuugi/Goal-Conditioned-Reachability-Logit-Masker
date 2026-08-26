import pytest
import torch
from core.fsm_builder import ReachabilityFSM
from core.logit_processor import GoalReachabilityLogitsProcessor


def test_logits_processor_masking():
    # Setup graph:
    # 0 -> 1 -> 2 (Goal) via token 1, then token 2 (total 2 steps)
    # 0 -> 3 (Dead-end) via token 3
    vocab_size = 5
    fsm = ReachabilityFSM(num_states=4, vocab_size=vocab_size)
    fsm.add_transition(0, 1, 1)
    fsm.add_transition(1, 2, 2)
    fsm.add_transition(0, 3, 3)
    fsm.set_goal_states([2])

    max_budget = 2
    processor = GoalReachabilityLogitsProcessor(fsm=fsm, max_budget=max_budget)

    # Step 1: Prompt is [0] (length 1), scores shape [1, 5]
    prompt_ids = torch.tensor([[0]], dtype=torch.long)
    initial_scores = torch.zeros((1, vocab_size), dtype=torch.float)

    masked_scores = processor(prompt_ids, initial_scores.clone())

    # Token 1 leads to 1 (which can reach goal in 1 step): Should be VALID (0.0)
    # Token 3 leads to 3 (dead-end): Should be -inf
    # Other tokens: not defined in FSM, should be -inf
    assert masked_scores[0, 1].item() == 0.0
    assert masked_scores[0, 3].item() == -float("inf")
    assert masked_scores[0, 0].item() == -float("inf")
    assert masked_scores[0, 2].item() == -float("inf")
    assert masked_scores[0, 4].item() == -float("inf")


def test_logits_processor_step_progression():
    # Linear graph: 0 -> 1 -> 2 (Goal)
    vocab_size = 5
    fsm = ReachabilityFSM(num_states=3, vocab_size=vocab_size)
    fsm.add_transition(0, 1, 1)
    fsm.add_transition(1, 2, 2)
    fsm.set_goal_states([2])

    processor = GoalReachabilityLogitsProcessor(fsm=fsm, max_budget=2)

    # Step 1
    input_ids = torch.tensor([[0]], dtype=torch.long)
    scores = torch.zeros((1, vocab_size))
    s1 = processor(input_ids, scores.clone())
    assert s1[0, 1].item() == 0.0

    # Step 2: token 1 was appended
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    scores = torch.zeros((1, vocab_size))
    s2 = processor(input_ids, scores.clone())
    # State should now be 1, only token 2 is valid
    assert s2[0, 2].item() == 0.0
    assert s2[0, 1].item() == -float("inf")


def test_batch_logits_processor():
    vocab_size = 5
    fsm = ReachabilityFSM(num_states=4, vocab_size=vocab_size)
    fsm.add_transition(0, 1, 1)
    fsm.add_transition(0, 2, 2)
    fsm.add_transition(1, 3, 3)
    fsm.add_transition(2, 3, 3)
    fsm.set_goal_states([3])

    processor = GoalReachabilityLogitsProcessor(fsm=fsm, max_budget=2)

    # Batch of 2 samples
    input_ids = torch.tensor([[0], [0]], dtype=torch.long)
    scores = torch.zeros((2, vocab_size))
    masked = processor(input_ids, scores)

    # Both batch items in state 0, tokens 1 and 2 should be valid
    for b in range(2):
        assert masked[b, 1].item() == 0.0
        assert masked[b, 2].item() == 0.0
        assert masked[b, 0].item() == -float("inf")
        assert masked[b, 3].item() == -float("inf")
