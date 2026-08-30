import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_history(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_training(log_path, save_path=None):
    history = load_history(log_path)
    epochs = [e['epoch'] for e in history]

    loss_keys = [k for k in history[0] if 'loss' in k.lower()]
    metric_keys = [
        k for k in history[0]
        if k != 'epoch' and isinstance(history[0][k], (int, float)) and k not in loss_keys
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key in loss_keys:
        axes[0].plot(epochs, [e[key] for e in history], marker='o', label=key)
    axes[0].set_title('Loss')
    axes[0].set_xlabel('epoch')
    axes[0].legend()
    axes[0].grid(True)

    for key in metric_keys:
        axes[1].plot(epochs, [e[key] for e in history], marker='o', label=key)
    axes[1].set_title('Metrics')
    axes[1].set_xlabel('epoch')
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    return fig
