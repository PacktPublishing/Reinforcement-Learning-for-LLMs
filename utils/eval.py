"""
eval.py: Evaluation helpers: reward scoring, KL divergence, win rate.
"""


def win_rate(chosen_rewards: list[float], rejected_rewards: list[float]) -> float:
    """Fraction of pairs where chosen reward exceeds rejected reward."""
    assert len(chosen_rewards) == len(rejected_rewards)
    wins = sum(c > r for c, r in zip(chosen_rewards, rejected_rewards))
    return wins / len(chosen_rewards)
