import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def _as_uint8_rgb(image):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    return image


def colorize(mask, palette):
    mask = np.asarray(mask)
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in palette.items():
        rgb[mask == int(cls)] = color
    return rgb


def save_mask(mask, palette, save_path):
    rgb = colorize(mask, palette)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    Image.fromarray(rgb).save(save_path)
    return save_path


def save_overlay(image, mask, palette, save_path, alpha=0.5):
    image = _as_uint8_rgb(image)
    color = colorize(mask, palette)
    blend = (image * (1 - alpha) + color * alpha).astype(np.uint8)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    Image.fromarray(blend).save(save_path)
    return save_path


def plot_sample(image, gt, pred, palette=None, names=None, save_path=None):
    image = _as_uint8_rgb(image)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')

    if palette is not None:
        axes[1].imshow(colorize(gt, palette))
        axes[2].imshow(colorize(pred, palette))
    else:
        axes[1].imshow(gt, cmap='gray')
        axes[2].imshow(pred, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    if palette is not None and names is not None:
        handles = []
        for cls, color in palette.items():
            label = names[int(cls)] if int(cls) < len(names) else str(cls)
            handles.append(mpatches.Patch(color=np.asarray(color) / 255.0, label=label))
        fig.legend(handles=handles, loc='lower center', ncol=len(handles))

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path
    return fig
