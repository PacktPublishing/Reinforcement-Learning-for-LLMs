"""
viz.py: Plotting helpers for training curves and reward distributions.
"""


def plot_training_curve(steps: list[int], values: list[float], title: str = "Training"):
    """Plot a single training metric over steps."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(steps, values)
    plt.xlabel("Step")
    plt.title(title)
    plt.tight_layout()
    plt.show()
