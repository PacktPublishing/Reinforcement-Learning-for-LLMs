#!/usr/bin/env python3
"""
sync_notebook_setup.py: standardize every companion notebook's top cells.

For each notebook it (idempotently) sets:
  1. cell 0  -> a pastel "banner" header (per-part theme), correct Packt Colab badge,
               book attribution, tested-on badge, and a "What this notebook covers" box.
  2. cell 1  -> a SELF-CONTAINED Setup cell that pip-installs ONLY the packages this
               notebook actually imports, pinned from code/requirements.txt.

Versions live ONLY in code/requirements.txt (the source of truth). Edit there, re-run this.
Run:  python code/scripts/sync_notebook_setup.py
"""
import json, glob, os, re, sys

REPO = "/Users/arunprasathshankar/Desktop/repos/packt-final"
NB_ROOT = f"{REPO}/code/notebooks"
REQ = f"{REPO}/code/requirements.txt"
PACKT = "PacktPublishing/Reinforcement-Learning-for-LLMs"
TESTED = "Google Colab (T4), 2026-06-29"

# ---- 1. version pins from requirements.txt (name -> "name==x" or "name") ----
PIN = {}
for line in open(REQ):
    line = line.split("#")[0].strip()
    if not line:
        continue
    name = re.split(r"[=<>~]", line)[0].strip()
    PIN[name.lower()] = line

# ---- 2. import-name -> pip-name map + things we never pip install on Colab ----
IMPORT_TO_PIP = {
    "sklearn": "scikit-learn", "PIL": "pillow", "cv2": "opencv-python",
    "yaml": "pyyaml", "bs4": "beautifulsoup4", "skimage": "scikit-image",
    "dotenv": "python-dotenv", "hf_hub": "huggingface_hub",
}
# Colab already ships these (and pinning them can break the CUDA runtime) -> skip install
COLAB_PREINSTALLED = {
    "torch", "numpy", "pandas", "matplotlib", "scipy", "sklearn", "requests",
    "tqdm", "PIL", "seaborn", "IPython", "google", "sympy", "networkx", "plotly",
}
STDLIB = set(getattr(sys, "stdlib_module_names", set())) | {"__future__"}

# (roman, shields-label, hex-color, emoji): Colab strips CSS, so colour comes
# from shields.io badge IMAGES, which render in both Colab and GitHub.
# Authoritative chapter map (basename -> (chapter_number, title)). Source of truth
# for the header, immune to re-runs transforming the title cell.
CANON = {
    "01_math_toolkit.ipynb": (1, "Essential Math Toolkit"),
    "02_alignment_gap.ipynb": (2, "Why LLMs Need RL: The Alignment Gap"),
    "03_rl_fundamentals.ipynb": (3, "RL Fundamentals: The Complete Picture"),
    "04_environment_setup.ipynb": (4, "Setting Up Your Free Training Environment"),
    "05_sft.ipynb": (5, "Supervised Fine-Tuning & The Cold Start"),
    "06_rlhf_ppo.ipynb": (6, "RLHF: The Three-Step Dance"),
    "07_dpo.ipynb": (7, "Direct Preference Optimization"),
    "08_online_dpo.ipynb": (8, "Online DPO & Iterative Alignment"),
    "09_reward_modeling.ipynb": (9, "Reward Modeling & The Critic"),
    "10_grpo_rloo_kto.ipynb": (10, "Modern RL Algorithms: GRPO, RLOO, KTO & More"),
    "11_verifiers_outcome_rewards.ipynb": (11, "Verifiers & Outcome Rewards: Beyond PRMs"),
    "12a_reasoning_grpo_concepts.ipynb": (12, "Reasoning with GRPO: The DeepSeek Recipe (Part A)"),
    "12b_reasoning_grpo_deepseek.ipynb": (12, "Reasoning with GRPO: The DeepSeek Recipe (Part B)"),
    "13_test_time_compute.ipynb": (13, "Test-Time Compute: Scaling at Inference"),
    "14_selfplay_constitutional_ai.ipynb": (14, "Self-Play & Constitutional AI"),
    "15_multiobjective_agentic.ipynb": (15, "Multi-Objective RL & Agentic Systems"),
    "16_domain_specific_rl.ipynb": (16, "Domain-Specific RL: Code, Math, Tools, Dialogue"),
    "17_recipe_chatbot.ipynb": (17, "Three Recipes, Recipe 1: The Chatbot"),
    "18_recipe_reasoner.ipynb": (17, "Three Recipes, Recipe 2: The Reasoner"),
    "19_recipe_agent.ipynb": (17, "Three Recipes, Recipe 3: The Agent"),
}

# Chapter 4 is the provisioning chapter: it must install (and then verify) the FULL
# core stack the rest of the book needs, not merely what it imports itself.
FULL_STACK = {"04_environment_setup.ipynb"}
CORE_STACK = ["transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes"]

# RL notebooks written against the OLD TRL API (PPOTrainer.step(), old SFTTrainer /
# PPOConfig). Per the "pin the old TRL they were written for" decision, install the
# exact stack each was authored against, so the existing code runs unchanged.
# (torch is NOT pinned -- Colab provides it. These old pkgs on a 2026 Colab may need
# nudging; verify on Colab.)
# 06 was rewritten TRL-free (manual SFT + from-scratch PPO) -> uses the current stack.
OLD_TRL_PINS = {
    # (none) - 06/08/10 are from-scratch PyTorch implementations and do NOT import TRL.
    # They install only what they import (the modern pinned stack from requirements.txt),
    # so pinning the old trl==0.8.6 stack here was wrong and has been removed.
}

PART_THEME = {
    "part1_foundations": ("I",   "Foundations",   "6D5DD3", "🧭"),
    "part2_core":        ("II",  "Core%20Methods", "1F9D6B", "⚙️"),
    "part3_advanced":    ("III", "Advanced",      "E07B39", "🚀"),
    "part4_recipes":     ("IV",  "Recipes",       "2F76C9", "🍳"),
}

def imported_modules(nb):
    mods = set()
    pat = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z_][\w]*)", re.M)
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        for m in pat.finditer(src):
            mods.add(m.group(1))
    return mods

def build_install_list(mods):
    pip_names = []
    for m in sorted(mods):
        if m in STDLIB or m in COLAB_PREINSTALLED:
            continue
        pip = IMPORT_TO_PIP.get(m, m)
        pip_names.append(pip)
    pip_names = sorted(set(pip_names))
    # Unsloth pulls its own transformers/trl/peft/etc -> let it resolve those.
    if "unsloth" in pip_names:
        managed = {"transformers", "trl", "peft", "datasets", "accelerate",
                   "bitsandbytes", "xformers"}
        pip_names = [p for p in pip_names if p not in managed]
        if "torchao" not in pip_names:
            pip_names.append("torchao")
    # apply pins
    specs = [PIN.get(p.lower(), p) for p in pip_names]
    return specs

def setup_cell(num, specs, check_mods):
    if specs:
        # wrap install across lines for readability
        body = " ".join(specs)
        install = f'    %pip install -q --progress-bar off {body}'
    else:
        install = '    pass  # all dependencies are pre-installed on Colab'
    checks = [m for m in sorted(check_mods)
              if m not in STDLIB and m in {*PIN, "torch", "numpy",
              "transformers", "trl", "peft", "datasets", "accelerate",
              "bitsandbytes", "unsloth", "torchao"}]
    check_list = ", ".join(f'"{c}"' for c in checks) if checks else ""
    src = [
        f"# === Setup: Chapter {num} · tested on {TESTED} ===\n",
        "# Self-contained: installs ONLY what this notebook imports (pinned).\n",
        "# Versions are managed centrally in code/requirements.txt.\n",
        "import os, sys\n",
        'os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # quiet HF download bars (keeps GitHub able to render the saved notebook)\n',
        'IN_COLAB = "google.colab" in sys.modules\n',
        "if IN_COLAB:\n",
        install + "\n",
    ]
    if check_list:
        src += [
            "\n",
            "import importlib\n",
            f"for _p in [{check_list}]:\n",
            "    try:\n",
            '        print(f"{_p:<16}", importlib.import_module(_p).__version__)\n',
            "    except Exception as _e:\n",
            '        print(f"{_p:<16} (not importable here)")\n',
        ]
    return {"cell_type": "code", "metadata": {"packt_ui": "setup"}, "id": "packt-setup",
            "execution_count": None, "outputs": [], "source": src}

def header_cell(part, num, title, relpath, covers):
    roman, label, hexc, emoji = PART_THEME[part]
    colab = f"https://colab.research.google.com/github/{PACKT}/blob/main/code/notebooks/{relpath}"
    part_badge = f"https://img.shields.io/badge/Part%20{roman}-{label}-{hexc}?style=for-the-badge"
    tested_badge = "https://img.shields.io/badge/Tested-Colab%20T4-2FAE7E?style=for-the-badge"
    # Pure markdown + shields.io badge IMAGES -> renders in Colab AND GitHub.
    lines = [
        f"# {emoji} Chapter {num}: {title}",
        "",
        f"![Part]({part_badge}) &nbsp; ![Tested]({tested_badge}) &nbsp; "
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})",
        "",
        "> 📘 **Book:** *Reinforcement Learning for Large Language Models*, "
        "Arun Shankar & Michael Chertushkin (Packt, 2026)  ",
        f"> 📓 **Notebook:** `{relpath}`  ",
        f"> ✅ **Tested on:** {TESTED}",
        "",
        "> 📌 **What this notebook covers**  ",
        f"> {covers}",
        "",
        "---",
    ]
    return {"cell_type": "markdown", "metadata": {"packt_ui": "header"}, "id": "packt-header",
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}

def parse_old_title(cell):
    src = "".join(cell["source"])
    m = re.search(r"#\s*Chapter\s+(\d+):\s*(.+)", src)
    num = m.group(1) if m else "?"
    title = m.group(2).strip() if m else "Companion Notebook"
    cov = re.search(r"What this notebook covers\s*\n+([^\n]+)", src)
    covers = cov.group(1).strip() if cov else (
        f"Companion notebook for Chapter {num}. Runs on a free Colab T4; "
        "all models are small enough for the free tier.")
    return num, title, covers

def process(path):
    nb = json.load(open(path))
    relpath = os.path.relpath(path, NB_ROOT)
    part = relpath.split("/")[0]
    cells = nb["cells"]

    # find old title cell (first markdown starting with "# Chapter") or existing header
    title_idx = next((i for i, c in enumerate(cells)
                      if c["cell_type"] == "markdown"
                      and (c.get("metadata", {}).get("packt_ui") == "header"
                           or "".join(c["source"]).lstrip().startswith("# Chapter"))), None)
    if title_idx is None:
        return f"{relpath}: SKIP (no title cell)"
    base = os.path.basename(path)
    if base in CANON:
        n, title = CANON[base]
        num = str(n)
        covers = (f"This is the companion notebook for Chapter {num} of the book. "
                  "Run it on a free Colab T4 GPU. All code uses small, publicly "
                  "available models (under 500 MB) that fit within the free-tier "
                  "memory limit.")
    else:
        num, title, covers = parse_old_title(cells[title_idx])
    new_header = header_cell(part, num, title, relpath, covers)

    mods = imported_modules(nb)
    if base in FULL_STACK:                 # provisioning chapter: install full stack
        mods |= set(CORE_STACK)
    specs = build_install_list(mods)
    if base in OLD_TRL_PINS:               # RL notebooks on the old TRL API
        specs = OLD_TRL_PINS[base]
    new_setup = setup_cell(num, specs, mods)

    # remove any existing tagged header/setup cells, and the first old pip/IN_COLAB code cell
    keep = []
    removed_setup = False
    for i, c in enumerate(cells):
        if i == title_idx:
            continue
        tag = c.get("metadata", {}).get("packt_ui")
        if tag in ("header", "setup"):
            continue
        if (not removed_setup and c["cell_type"] == "code"
                and re.search(r"%?pip install|IN_COLAB", "".join(c["source"]))):
            removed_setup = True
            continue
        keep.append(c)
    nb["cells"] = [new_header, new_setup] + keep
    json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
    open(path, "a").write("\n")
    return f"{relpath:52s} ch{num:>2}  install: {specs if specs else '(none)'}"

if __name__ == "__main__":
    for f in sorted(glob.glob(NB_ROOT + "/part*/*.ipynb")):
        print(process(f))
