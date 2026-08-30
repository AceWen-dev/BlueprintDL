import os

import torch


def export_onnx(model, input_size, save_path, opset=11, dynamic_batch=True, device='cpu'):
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, input_size[0], input_size[1], device=device)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.onnx.export(
        model, dummy, save_path,
        input_names=['input'], output_names=['output'],
        opset_version=opset, dynamic_axes=dynamic_axes,
    )
    return save_path


def export_torchscript(model, input_size, save_path, device='cpu'):
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, input_size[0], input_size[1], device=device)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    traced = torch.jit.trace(model, dummy)
    torch.jit.save(traced, save_path)
    return save_path
