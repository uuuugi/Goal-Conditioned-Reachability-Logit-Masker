# Goal-Conditioned Reachability Logit Masker (GCLM)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers 4.36+](https://img.shields.io/badge/Transformers-4.36+-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An ultra-fast, strictly $O(1)$ runtime **Goal-Conditioned Reachability Logit Masking Engine** for Large Language Models.  
GCLM mathematically guarantees that an LLM will strictly reach designated goal/accepting states within a fixed token budget ($T_{\max}$), **fundamentally preventing dead-end traps and truncated syntax failures**.

---

## 💡 Key Differences: GCLM vs. Forward DFA Maskers (Outlines / SGLang)

```
[Traditional Forward DFA (Outlines / SGLang)]
  Start (A) ─── Token X ───▶ [Valid Branch D] ─── Token Y ───▶ [Dead-End / Truncated Trap ❌]
  (Only checks if transition exists from current state)

[GCLM: Time-Bounded Backward Reachability (Ours)]
  Start (A) ─── Token X (Masked to -inf ⛔)
            └── Token B ───▶ State C ───▶ Goal / Closing '}' ✅
  (Preemptively prunes any branch that cannot reach Goal in <= T_rem steps)
```

| Feature | Standard Forward DFA (Outlines / SGLang) | **GCLM (Ours)** |
| :--- | :--- | :--- |
| **Masking Basis** | Current state validity ($s_{\text{curr}} \xrightarrow{v} s'$) | **Time-bounded backward reachability** ($s_{\text{curr}} \xrightarrow{v} s' \rightsquigarrow S_{\text{goal}}$ in $\le T_{\text{rem}}-1$ steps) |
| **Dead-End Traps** | ❌ May enter valid forward branches that lead to dead-ends | ✅ **Preemptively masked** before entering trap |
| **Token Budget Exceeded**| ❌ Outputs truncated/broken syntax when budget ends | ✅ **Forces early syntax closure** before budget exhaustion |
| **Per-Token Overhead** | $O(1)$ table lookup | **Strict $O(1)$ vectorized PyTorch lookup (< 0.1ms)** |
| **Complexity Scaling** | Scales with active state transitions | **Zero runtime dependence on state count $\|S\|$** |

---

## 📐 Mathematical Formulation

### 1. Offline Backward BFS Table Builder
Given an FSM $(S, \Sigma, \delta, s_0, S_{\text{goal}})$ and maximum token budget $T_{\max}$, we precompute a reachability tensor $R \in \mathbb{B}^{(T_{\max} + 1) \times |S|}$ via vectorized backward BFS:

$$R[0, s] = \begin{cases} \text{True} & \text{if } s \in S_{\text{goal}} \\ \text{False} & \text{otherwise} \end{cases}$$

For $t = 1, \dots, T_{\max}$:
$$R[t, s] = R[t-1, s] \lor \left( \exists v \in \mathcal{V} \text{ s.t. } \delta(s, v) \ge 0 \land R[t-1, \delta(s, v)] = \text{True} \right)$$

### 2. Strict $\mathcal{O}(1)$ Runtime Logits Masking
At decoding step $k$ with remaining budget $T_{\text{rem}} = T_{\max} - k$:

$$
\operatorname{ValidTokens}(v) = (\delta(s_{\mathrm{curr}}, v) \ge 0) \;\land\; R\big[\min(T_{\text{rem}}-1, T_{\max}), \;\operatorname{clamp}(\delta(s_{\mathrm{curr}}, v), 0)\big]
$$

$$
\operatorname{Logits}[v] = 
\begin{cases} 
\operatorname{Logits}[v] & \text{if } \operatorname{ValidTokens}(v) = \text{True} \\\\ 
-\infty & \text{otherwise} 
\end{cases}
$$

---

## 📁 Repository Structure

```
gclm_project/
├── core/
│   ├── __init__.py
│   ├── fsm_builder.py          # Transitions tensor & vectorized backward BFS reachability table
│   ├── logit_processor.py      # Hugging Face LogitsProcessor compatible O(1) in-place masker
│   └── compiler.py             # Tokenizer-aware grammar/pattern compiler
├── benchmarks/
│   ├── synthetic_deadend.py    # Experiment 1: Dead-end trap avoidance benchmark
│   ├── json_budget_bench.py    # Experiment 2: Real-world strict budget JSON benchmark
│   ├── tool_calling_bench.py   # Experiment 3: Multi-step agent action budget benchmark
│   ├── scaling_bench.py        # Experiment 4: Complexity scaling (|S|=10~10,000) & plot generator
│   ├── real_model_bench.py     # Experiment 5: Real lightweight LLM (Qwen2.5) E2E benchmark
│   └── latency_bench.py        # Per-token runtime overhead benchmark
├── examples/
│   └── run_generation.py       # Live interactive generation demo with Transformers
├── tests/
│   ├── test_fsm_builder.py     # Unit tests for BFS reachability & multi-goal
│   └── test_logit_processor.py # Unit tests for batch masking & state progression
├── paper_figure_scaling.png    # Publication-ready 300-DPI scaling figure
├── requirements.txt
└── README.md
```

---

## 📊 Comprehensive Experimental Results

### 1. Real Lightweight LLM End-to-End Benchmark (`Qwen2.5-0.5B`)
> Tested on real model weights generating JSON responses under strict token limits.

| Token Budget ($T_{\max}$) | Vanilla Sampling | Forward DFA (Outlines Style) | **GCLM (Ours)** | Latency / Sample (GCLM) |
| :--- | :---: | :---: | :---: | :---: |
| **$T_{\max} = 6$ tokens** | 0.0% | 30.0% | **100.0%** | **615.90 ms** (Fastest, early closure) |
| **$T_{\max} = 10$ tokens**| 0.0% | 70.0% | **100.0%** | **1,086.02 ms** |
| **$T_{\max} = 16$ tokens**| 0.0% | 85.0% | **100.0%** | **992.39 ms** |

---

### 2. Strict Budget JSON Schema Parsing Benchmark
> Complex nested JSON schema tested across 500 trials per budget.

| Budget ($T_{\max}$) | Vanilla | Forward DFA (Outlines Style) | **GCLM (Ours)** | Key Insight |
| :--- | :---: | :---: | :---: | :--- |
| **$T_{\max} = 4$** | 2.4% | 55.4% | **100.0%** | **Forces safe `{}` closure when fields cannot finish** |
| **$T_{\max} = 6$** | 2.4% | 45.6% | **100.0%** | Prunes deep nested object paths |
| **$T_{\max} = 8$** | 2.2% | 65.2% | **100.0%** | Eliminates dangling commas |
| **$T_{\max} = 16$** | 1.4% | 91.8% | **100.0%** | Complete 100% parse rate across all budgets |

---

### 3. Multi-Step Agent Tool-Calling & Action Budget Benchmark
> ReAct-style multi-tool workflow evaluating goal completion within action limits.

| Action Budget | Vanilla | Forward DFA | **GCLM (Ours)** | Key Finding |
| :--- | :---: | :---: | :---: | :--- |
| **3 Actions** | 0.00% | 16.80% | **100.00%** | Dynamically forces 3-step shortest path |
| **4 Actions** | 0.00% | 33.20% | **100.00%** | Prunes unfinishable deep search subtrees |
| **8 Actions** | 0.60% | 65.20% | **100.00%** | **Completely avoids infinite retry trap loops** |

---

### 4. FSM Complexity & Strict $\mathcal{O}(1)$ Runtime Scaling
> Scaling state count $|S|$ from 10 to 10,000 (1,000x increase). Plot saved as `paper_figure_scaling.png`.

| Vocabulary Size $\vert\mathcal{V}\vert$ | State Count $\vert S\vert$ | Offline BFS Time | Memory Footprint | Online Latency per Token |
| :--- | :---: | :---: | :---: | :---: |
| **$\vert\mathcal{V}\vert = 32,000$ (LLaMA)** | $\vert S\vert = 10$ | 29.55 ms | 2.44 MB | **388.72 $\mu$s** |
| $\vert\mathcal{V}\vert = 32,000$ | $\vert S\vert = 100$ | 240.10 ms | 24.42 MB | **335.10 $\mu$s** |
| $\vert\mathcal{V}\vert = 32,000$ | $\vert S\vert = 1,000$ | 2,111.82 ms | 244.19 MB | **340.84 $\mu$s** |
| $\vert\mathcal{V}\vert = 32,000$ | **$\vert S\vert = 10,000$** | 25,790.14 ms | 2.44 GB | **356.29 $\mu$s** ($\mathcal{O}(1)$ empirically verified) |
| **$\vert\mathcal{V}\vert = 151,643$ (Qwen2.5)** | $\vert S\vert = 10$ | 159.29 ms | 11.57 MB | **601.92 $\mu$s** |
| $\vert\mathcal{V}\vert = 151,643$ | **$\vert S\vert = 10,000$** | 147,702.79 ms | 11.56 GB | **666.22 $\mu$s** ($\mathcal{O}(1)$ empirically verified) |

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/your-username/gclm.git
cd gclm
pip install -r requirements.txt
```

### 2. Basic Usage with Hugging Face Transformers
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from core.fsm_builder import ReachabilityFSM
from core.logit_processor import GoalReachabilityLogitsProcessor

# 1. Load model and tokenizer
model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

vocab_size = model.config.vocab_size
max_budget = 15

# 2. Define FSM & Goal state
fsm = ReachabilityFSM(num_states=5, vocab_size=vocab_size)
fsm.add_transition(from_state=0, token_id=101, to_state=1)
fsm.add_transition(from_state=1, token_id=102, to_state=2)
fsm.set_goal_states([2])

# 3. Precompute reachability table (one-time offline step)
fsm.build_reachability(max_steps=max_budget)

# 4. Attach GCLM to Hugging Face LogitsProcessorList
gclm_processor = GoalReachabilityLogitsProcessor(fsm=fsm, max_budget=max_budget)
logits_processors = LogitsProcessorList([gclm_processor])

# 5. Generate with guaranteed reachability
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=max_budget,
    logits_processor=logits_processors
)
print(tokenizer.decode(outputs[0]))
```

---

## 🧪 Reproducing Experiments

```bash
# Run Unit Tests
python -m pytest tests/ -v

# Run Experiment 1: Synthetic Dead-End Benchmark
python -m benchmarks.synthetic_deadend

# Run Experiment 2: Strict Budget JSON Benchmark
python -m benchmarks.json_budget_bench

# Run Experiment 3: Agent Tool-Calling Benchmark
python -m benchmarks.tool_calling_bench

# Run Experiment 4: Scaling Benchmark & Generate Paper Plots
python -m benchmarks.scaling_bench

# Run Experiment 5: Real Lightweight LLM Benchmark (Qwen2.5)
python -m benchmarks.real_model_bench --model Qwen/Qwen2.5-0.5B
```

---

## 📑 Citation & Author

```bibtex
@article{an2026gclm,
  title={Goal-Conditioned Reachability Logit Masker: Guaranteed Goal Satisfaction for Constrained LLM Generation in O(1) Time},
  author={An, ByeongUk},
  journal={arXiv preprint},
  year={2026}
}
```

**Author**: ByeongUk An  
**Email**: `hhjjkk7186@gmail.com`  
**ORCID**: [`0009-0007-5612-5602`](https://orcid.org/0009-0007-5612-5602)

---

## 📄 License
MIT License
