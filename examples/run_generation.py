"""
Example: End-to-End Generation with GCLM LogitsProcessor
Demonstrates strict budget JSON completion and dead-end avoidance with Hugging Face Transformers.
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from core.fsm_builder import ReachabilityFSM
from core.logit_processor import GoalReachabilityLogitsProcessor
from core.compiler import FSMCompiler


def run_demo(model_name: str = "Qwen/Qwen2.5-0.5B", budget: int = 15):
    print(f"\n[DEMO] Loading Tokenizer and Model: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model.to("cpu")
    except Exception as e:
        print(f"[WARN] Could not load remote model ({e}). Using GPT-2 fallback or mock...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        except Exception:
            print("[WARN] Hugging Face offline or unavailable. Running in synthetic mode.")
            return

    vocab_size = model.config.vocab_size
    compiler = FSMCompiler(vocab_size=vocab_size, tokenizer=tokenizer, device=device)

    # 1. Define JSON structure with optional fields:
    # {"status": "ok", "code": 200}
    # Open: '{"' or '{'
    # Close: '}'
    # We want model to complete valid JSON before budget runs out.
    
    prompt = "Generate a JSON response for server status: "
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Compile a simple strict JSON FSM
    open_tokens = tokenizer.encode('{"', add_special_tokens=False)
    kv1_tokens = tokenizer.encode('status":"ok",', add_special_tokens=False)
    kv2_tokens = tokenizer.encode('status":"ok"', add_special_tokens=False)
    kv3_tokens = tokenizer.encode('code":200', add_special_tokens=False)
    close_tokens = tokenizer.encode('}', add_special_tokens=False)
    eos_id = tokenizer.eos_token_id or 0

    # Build FSM
    # 0: Start -> 1: Open
    # 1 -> 2: status:"ok", -> 1 (loop)
    # 1 -> 3: status:"ok" -> 4: Close
    # 1 -> 5: code":200 -> 4: Close
    # 1 -> 4: Close
    # 4 -> 6: Goal (on EOS)

    fsm = ReachabilityFSM(num_states=10, vocab_size=vocab_size, device=device)
    
    # 0 -> 1 on open
    for t in open_tokens:
        fsm.add_transition(0, t, 1)

    # 1 -> 4 on close
    for t in close_tokens:
        fsm.add_transition(1, t, 4)

    # 1 -> loop or 1 -> close via KV
    if len(kv1_tokens) > 0:
        fsm.add_transition(1, kv1_tokens[0], 1)
    if len(kv2_tokens) > 0:
        fsm.add_transition(1, kv2_tokens[0], 4)
    if len(kv3_tokens) > 0:
        fsm.add_transition(1, kv3_tokens[0], 4)

    # 4 -> 6 on eos
    fsm.add_transition(4, eos_id, 6)
    fsm.add_transition(6, eos_id, 6)  # goal self-loop
    fsm.set_goal_states([6])

    fsm.build_reachability(max_steps=budget)

    # Setup GCLM LogitsProcessor
    gclm_processor = GoalReachabilityLogitsProcessor(fsm=fsm, max_budget=budget)
    logits_processors = LogitsProcessorList([gclm_processor])

    print(f"\nPrompt: '{prompt}'")
    print(f"Token Budget: {budget} tokens\n")

    # Generate with GCLM
    print("--- 1. Generation with GCLM (Guaranteed Completion) ---")
    output_gclm = model.generate(
        input_ids,
        max_new_tokens=budget,
        logits_processor=logits_processors,
        do_sample=True,
        temperature=0.7,
        pad_token_id=eos_id,
    )
    generated_text_gclm = tokenizer.decode(output_gclm[0], skip_special_tokens=False)
    print(f"Output:\n{generated_text_gclm}\n")

    # Generate with Vanilla (Unconstrained)
    print("--- 2. Generation with Vanilla (Unconstrained) ---")
    output_vanilla = model.generate(
        input_ids,
        max_new_tokens=budget,
        do_sample=True,
        temperature=0.7,
        pad_token_id=eos_id,
    )
    generated_text_vanilla = tokenizer.decode(output_vanilla[0], skip_special_tokens=False)
    print(f"Output:\n{generated_text_vanilla}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GCLM generation demo.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Model name or path")
    parser.add_argument("--budget", type=int, default=15, help="Max new tokens budget")
    args = parser.parse_args()

    run_demo(model_name=args.model, budget=args.budget)
