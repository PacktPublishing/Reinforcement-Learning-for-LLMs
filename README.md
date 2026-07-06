# Reinforcement Learning for Large Language Models
### *A Complete Guide*: Companion Code

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part1_foundations/01_math_toolkit.ipynb)
![Runs on a free Colab T4](https://img.shields.io/badge/runs%20on-free%20Colab%20T4-F9AB00)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB)
![License Apache 2.0](https://img.shields.io/badge/code-Apache%202.0-green)

> **Book:** *Reinforcement Learning for Large Language Models, A Complete Guide*
> **Authors:** Arun Shankar & Michael Chertushkin · Packt, 2026
> **Repo:** [github.com/PacktPublishing/Reinforcement-Learning-for-LLMs](https://github.com/PacktPublishing/Reinforcement-Learning-for-LLMs)

Nineteen self-contained notebooks that build modern LLM alignment **from the ground up**, from the essential math to end-to-end production recipes. Every method (PPO, DPO, GRPO, reward modeling, verifiers, self-play, constitutional AI, tool-use agents) is implemented in **plain PyTorch** where seeing the mechanics matters, and **every notebook runs on a single free Google Colab T4 GPU**. No API keys, no paid compute, no local setup.

---

## Quick start

1. Click any **Open in Colab** badge in the tables below.
2. ⚠️ **Set the runtime to a T4 GPU first**, **Runtime → Change runtime type → T4 GPU → Save**. This is the single most important step: on a CPU runtime the training notebooks are 10–100× slower and can run out of memory. The first cell prints `Device: cuda` and `GPU: Tesla T4` when you're set correctly.
3. **Runtime → Run all.** Each notebook installs its own pinned dependencies (~60–90 s) and runs top to bottom.

That's it, nothing to clone or `pip install` locally.

---

## Part 1: Foundations
*The math, the problem, and the setup.*

| # | Chapter | Open |
|---|---------|:----:|
| 1 | Essential Math Toolkit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part1_foundations/01_math_toolkit.ipynb) |
| 2 | Why LLMs Need RL: The Alignment Gap | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part1_foundations/02_alignment_gap.ipynb) |
| 3 | RL Fundamentals: The Complete Picture | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part1_foundations/03_rl_fundamentals.ipynb) |
| 4 | Setting Up Your Free Environment | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part1_foundations/04_environment_setup.ipynb) |

## Part 2: Core Methods
*The alignment toolkit, each built from scratch: SFT → RLHF → DPO → reward models → GRPO.*

| # | Chapter | Open |
|---|---------|:----:|
| 5 | Supervised Fine-Tuning & The Cold Start | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part2_core/05_sft.ipynb) |
| 6 | RLHF: The Three-Step Dance (PPO) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part2_core/06_rlhf_ppo.ipynb) |
| 7 | Direct Preference Optimization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part2_core/07_dpo.ipynb) |
| 8 | Online DPO & Iterative Alignment | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part2_core/08_online_dpo.ipynb) |
| 9 | Reward Modeling & The Critic | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part2_core/09_reward_modeling.ipynb) |
| 10 | Modern RL Algorithms: GRPO, RLOO, KTO | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part2_core/10_grpo_rloo_kto.ipynb) |

## Part 3: Advanced Techniques
*Verifiable rewards, reasoning models (the DeepSeek-R1 recipe), test-time compute, and self-improvement.*

| # | Chapter | Open |
|---|---------|:----:|
| 11 | Verifiers & Outcome Rewards | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part3_advanced/11_verifiers_outcome_rewards.ipynb) |
| 12a | Reasoning with GRPO: Concepts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part3_advanced/12a_reasoning_grpo_concepts.ipynb) |
| 12b | Reasoning with GRPO: The DeepSeek Recipe | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part3_advanced/12b_reasoning_grpo_deepseek.ipynb) |
| 13 | Test-Time Compute: Scaling at Inference | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part3_advanced/13_test_time_compute.ipynb) |
| 14 | Self-Play & Constitutional AI | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part3_advanced/14_selfplay_constitutional_ai.ipynb) |
| 15 | Multi-Objective RL & Agentic Systems | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part3_advanced/15_multiobjective_agentic.ipynb) |
| 16 | Domain-Specific RL: Code, Math, Tools | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part3_advanced/16_domain_specific_rl.ipynb) |

## Part 4: Production Recipes
*Three end-to-end pipelines, each combining the earlier building blocks.*

| # | Chapter | What it builds | Open |
|---|---------|----------------|:----:|
| 17 | Recipe: Chatbot | SFT → reward model → DPO (helpful **and** safe) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part4_recipes/17_recipe_chatbot.ipynb) |
| 18 | Recipe: Reasoner | SFT (chain-of-thought) → GRPO → test-time scaling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part4_recipes/18_recipe_reasoner.ipynb) |
| 19 | Recipe: Agent | Tool-use SFT + trajectory rewards + constitutional safety | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Reinforcement-Learning-for-LLMs/blob/main/notebooks/part4_recipes/19_recipe_agent.ipynb) |

---

## Models

Everything trains on a **0.5B** model so the full run fits a free T4. Larger models appear only as *inference-only* comparisons or as illustrative scale/cost references in the prose, they are never fine-tuned.

| Model | Used in | Role |
|-------|---------|------|
| `Qwen/Qwen2.5-0.5B` | Ch 2, 4–6, 8–12a, 15–17 | trainable base policy / reward-model backbone |
| `Qwen/Qwen2.5-0.5B-Instruct` | Ch 7, 13, 14, 18, 19 | instruction-tuned policy |
| `lvwerra/distilbert-imdb` | Ch 6, 8 | small pretrained sentiment reward model |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Ch 12b, 13 | inference-only comparison (loaded in fp16) |

---

## Repository structure

```
Reinforcement-Learning-for-LLMs/
├── data/                      # Sample data and artifacts
├── notebooks/
│   ├── part1_foundations/     # Chapters 1–4
│   ├── part2_core/            # Chapters 5–10
│   ├── part3_advanced/        # Chapters 11–16
│   └── part4_recipes/         # Chapters 17–19
├── scripts/                   # Maintenance and setup scripts
├── utils/                     # Custom helper modules (data, eval, viz)
├── LICENSE
├── README.md                  # (this file)
├── pre-commit-config.yaml
└── requirements.txt           # pinned dependency versions (source of truth)
```

---

## How it works

- **Self-contained.** Each notebook's first cell installs exactly the dependencies it needs, pinned to a validated Colab run, you don't manage a shared environment.
- **Pinned versions.** [`requirements.txt`](requirements.txt) is the single source of truth for the core stack (`transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`). PyTorch/NumPy are intentionally left to Colab's GPU-matched builds.
- **T4-safe.** Every training notebook is tuned to fit the ~15 GB T4 budget, fp32 where update precision matters, gradient checkpointing, and careful freeing of optimizer/gradient state between stages.
- **Honest results.** Plots and metrics are produced by the code you see, not hard-coded. Where a 0.5B model has genuine limits, the notebooks say so rather than faking a curve.

---

## Hardware & runtime

All notebooks target the **free Colab T4 GPU** (Runtime → Change runtime type → **T4 GPU**).

| Part | Typical runtime on a T4 |
|------|-------------------------|
| Part 1: Foundations | < 5 min (CPU is fine here) |
| Part 2: Core methods | 10–30 min |
| Part 3: Advanced | 20–60 min |
| Part 4: Recipes | 20–45 min |

---

## License

Code: **Apache 2.0** (see [`LICENSE`](LICENSE)).
Book text & figures: **© 2026 Arun Shankar & Michael Chertushkin. All rights reserved.**
