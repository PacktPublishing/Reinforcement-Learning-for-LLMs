"""
data.py: Dataset loading and preference pair utilities.
Shared helpers imported by notebooks that need them.
"""


def load_preferences(path: str):
    """Load a .jsonl file of (prompt, chosen, rejected) preference pairs."""
    import json
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_prompts(path: str):
    """Load a .jsonl file of prompts."""
    import json
    with open(path) as f:
        return [json.loads(line)["prompt"] for line in f if line.strip()]
